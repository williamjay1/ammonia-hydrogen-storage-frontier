"""Run the EU-27 public country-reference two-inventory frontier.

Renewable and electrolyser nameplates are sized from 2006-2014 local-year
weather. For each 2015-2023 evaluation year, the hydrogen and ammonia
inventories are re-optimized with perfect foresight and a cyclic annual
boundary. This is not a fixed-storage-capacity holdout or reliability test.
Every profile is a single reference point, not a national average, grid-bidding
zone, production facility, or complete RFNBO certificate assessment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

try:
    from .rfnbo_inventory_frontier import (
        H2_PER_NH3_KG_PER_T, HIGHS_LP_OPTIONS, LPSettings, solve_frontier_point,
    )
except ImportError:  # Support direct execution as ``python src/run_eu27_rfnbo_frontier.py``.
    from rfnbo_inventory_frontier import (
        H2_PER_NH3_KG_PER_T, HIGHS_LP_OPTIONS, LPSettings, solve_frontier_point,
    )


PROJECT = Path(r"D:\MLWork\project12_h2flex_eu")
RAW = Path(
    r"F:\AcademicData\project12_h2flex_eu\raw\eu27_reference_profiles_20260929_2005_2023"
)
POINTS = PROJECT / "datasets" / "eu27_reference_points_2024.csv"
OUT_ROOT = PROJECT / "results"
COUNTRY_CODES = (
    "AT", "BE", "BG", "HR", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI",
    "FR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO",
    "SE", "SI", "SK",
)
TRAIN_YEARS = tuple(range(2006, 2015))
EVALUATION_YEARS = tuple(range(2015, 2024))
BUFFER_HOURS = (0.0, 24.0, 72.0, 168.0, 336.0)
H2_REDUCTION_TARGETS = (0.25, 0.50, 0.75)
MAX_NH3_BUFFER_HOURS_FOR_THRESHOLD = 336.0
SEC_KWH_PER_KG = 52.0
DEMAND_TPH = 1.0
HB_NAMEPLATE_TPH = 1.5
HB_MIN_LOAD_FRACTION = 0.40
HB_RAMP_FRACTION_NAMEPLATE_PER_HOUR = 0.60
ELECTROLYSER_MULTIPLE = 3.0
RENEWABLE_OVERBUILD = 1.50
WIND_ENERGY_SHARE = 0.50


def buffer_frontier_cases() -> tuple[tuple[float, str], ...]:
    """Return sampled buffer ceilings, including the flexible zero-buffer control."""
    return tuple(
        (hours, "flexible_no_product_buffer" if hours == 0.0
         else "flexible_plus_nh3_buffer")
        for hours in BUFFER_HOURS
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_country_points() -> dict[str, dict[str, object]]:
    if not POINTS.exists():
        raise FileNotFoundError(f"Missing GISCO-derived coordinate table: {POINTS}")
    frame = pd.read_csv(POINTS, dtype={"country_code": str})
    if set(frame["country_code"]) != set(COUNTRY_CODES) or len(frame) != 27:
        raise ValueError("GISCO-derived point file must contain exactly one record per EU-27 member")
    if frame["country_code"].duplicated().any() or not frame[["latitude", "longitude"]].notna().all().all():
        raise ValueError("Country reference coordinates contain duplicates or missing values")
    return {str(row["country_code"]): row.to_dict() for _, row in frame.iterrows()}


def pvgis_series(country_code: str) -> pd.Series:
    path = RAW / "pv" / f"{country_code.lower()}_pvgis_era5_2005_2023.json"
    data = read_json(path)
    meteo = data.get("inputs", {}).get("meteo_data", {})
    if meteo.get("meteo_db") != "ERA5":
        raise ValueError(f"Unexpected PVGIS database metadata: {path}")
    items = data.get("outputs", {}).get("hourly", [])
    stamps = pd.to_datetime([item["time"] for item in items], format="%Y%m%d:%H%M", utc=True)
    # PVGIS reports hourly interval midpoints (:30); convert to interval starts.
    index = stamps - pd.Timedelta(minutes=30)
    cf = np.asarray([float(item["P"]) for item in items], dtype=float) / 1000.0
    if len(index) != 166536 or not np.isfinite(cf).all() or np.any(cf < 0) or np.any(cf > 1.05):
        raise ValueError(f"Invalid or incomplete PVGIS series: {path}")
    return pd.Series(cf, index=index, name="pv_cf").sort_index()


def wind_speed_series(country_code: str) -> pd.Series:
    path = RAW / "wind" / f"{country_code.lower()}_openmeteo_era5_wind100m_2005_2023.json"
    data = read_json(path)
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    speed = hourly.get("wind_speed_100m", [])
    if len(times) != 166536 or len(times) != len(speed):
        raise ValueError(f"Malformed or incomplete Open-Meteo series: {path}")
    values = np.asarray(speed, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"Invalid wind speed values: {path}")
    index = pd.to_datetime(times, utc=True)
    return pd.Series(values, index=index, name="wind_speed_100m").sort_index()


def local_year_index(year: int, timezone_name: str) -> pd.DatetimeIndex:
    """UTC interval starts spanning one complete local civil calendar year."""
    start = pd.Timestamp(year=year, month=1, day=1, tz=timezone_name).tz_convert("UTC")
    end = pd.Timestamp(year=year + 1, month=1, day=1, tz=timezone_name).tz_convert("UTC")
    return pd.date_range(start, end, freq="h", inclusive="left")


def reindex_complete(series: pd.Series, index: pd.DatetimeIndex, label: str) -> np.ndarray:
    aligned = series.reindex(index)
    missing = int(aligned.isna().sum())
    if missing:
        raise ValueError(f"{label} missing {missing} of {len(index)} UTC-year hourly records")
    return aligned.to_numpy(dtype=float)


def wind_cf_from_speed(
    speed: np.ndarray, cut_in: float = 3.0, rated: float = 12.0, cut_out: float = 25.0
) -> np.ndarray:
    speed = np.asarray(speed, dtype=float)
    if not np.isfinite(speed).all() or np.any(speed < 0):
        raise ValueError("Wind speed must be finite and non-negative")
    if not (0 <= cut_in < rated < cut_out):
        raise ValueError("Invalid generic wind-curve points")
    cf = np.zeros_like(speed)
    ramp = (speed >= cut_in) & (speed < rated)
    cf[ramp] = (speed[ramp] ** 3 - cut_in**3) / (rated**3 - cut_in**3)
    cf[(speed >= rated) & (speed < cut_out)] = 1.0
    return cf


def profile_for_year(
    year: int,
    timezone_name: str,
    pv: pd.Series,
    wind_speed: pd.Series,
    pv_mw: float,
    wind_mw: float,
    *,
    cut_in: float = 3.0,
    rated: float = 12.0,
    cut_out: float = 25.0,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    index = local_year_index(year, timezone_name)
    pv_cf = reindex_complete(pv, index, "PVGIS")
    speed = reindex_complete(wind_speed, index, "Open-Meteo ERA5 wind")
    wind_cf = wind_cf_from_speed(speed, cut_in, rated, cut_out)
    generation = pv_mw * pv_cf + wind_mw * wind_cf
    # RFNBO matching is grouped by complete civil calendar month at the
    # reference point. The local-year index prevents UTC year boundaries from
    # leaking an adjacent year's partial January/December into the budget.
    local = index.tz_convert(timezone_name)
    month_labels = np.asarray(local.strftime("%Y-%m"), dtype=str)
    if not np.isfinite(generation).all():
        raise ValueError(f"Non-finite renewable generation for {year}")
    return index, generation, month_labels, pv_cf, wind_cf


def design_capacities(
    timezone_name: str,
    pv: pd.Series,
    wind_speed: pd.Series,
    *,
    cut_in: float = 3.0,
    rated: float = 12.0,
    cut_out: float = 25.0,
    renewable_overbuild: float = RENEWABLE_OVERBUILD,
    wind_energy_share: float = WIND_ENERGY_SHARE,
) -> dict[str, float]:
    if (not np.isfinite(renewable_overbuild) or renewable_overbuild <= 0
            or not np.isfinite(wind_energy_share) or not 0 <= wind_energy_share <= 1):
        raise ValueError("renewable overbuild must be positive and wind energy share must be in [0, 1]")
    pv_train: list[np.ndarray] = []
    wind_train: list[np.ndarray] = []
    for year in TRAIN_YEARS:
        index = local_year_index(year, timezone_name)
        pv_train.append(reindex_complete(pv, index, "PVGIS training"))
        speed = reindex_complete(wind_speed, index, "ERA5 wind training")
        wind_train.append(wind_cf_from_speed(speed, cut_in, rated, cut_out))
    pv_mean = float(np.mean(np.concatenate(pv_train)))
    wind_mean = float(np.mean(np.concatenate(wind_train)))
    electricity_mwh_per_t = H2_PER_NH3_KG_PER_T * SEC_KWH_PER_KG / 1000.0
    total_design_energy_mw = renewable_overbuild * DEMAND_TPH * electricity_mwh_per_t
    pv_energy_mw = total_design_energy_mw * (1.0 - wind_energy_share)
    wind_energy_mw = total_design_energy_mw * wind_energy_share
    if pv_mean <= 0 or wind_mean <= 0:
        raise ValueError(f"Non-positive training resource factor at point ({pv_mean}, {wind_mean})")
    return {
        "pv_capacity_mw": pv_energy_mw / pv_mean,
        "wind_capacity_mw": wind_energy_mw / wind_mean,
        "pv_mean_cf_training": pv_mean,
        "wind_mean_cf_training": wind_mean,
        "electrolyser_mw": ELECTROLYSER_MULTIPLE * DEMAND_TPH * electricity_mwh_per_t,
        "grid_connection_mw": ELECTROLYSER_MULTIPLE * DEMAND_TPH * electricity_mwh_per_t,
        "renewable_overbuild": renewable_overbuild,
        "wind_energy_share": wind_energy_share,
    }


def point_result_row(
    point: dict[str, object],
    year: int,
    rule: str,
    frontier_kind: str,
    process_mode: str,
    buffer_hours: float,
    result: object,
    capacities: dict[str, float],
    generation: np.ndarray,
    pv_cf: np.ndarray,
    wind_cf: np.ndarray,
    *,
    target_h2_reduction: float | None = None,
    status: str = "optimal",
    error: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "country_code": point["country_code"],
        "country": point["country"],
        "timezone": point["timezone"],
        "latitude": point["latitude"],
        "longitude": point["longitude"],
        "local_weather_year": year,
        "temporal_matching_rule": rule,
        "frontier_kind": frontier_kind,
        "process_mode": process_mode,
        "nh3_buffer_limit_hours": buffer_hours,
        "target_h2_reduction_fraction": target_h2_reduction,
        "status": status,
        "error": error,
        "renewable_generation_mwh": float(np.sum(generation)),
        "pv_capacity_mw": capacities["pv_capacity_mw"],
        "wind_capacity_mw": capacities["wind_capacity_mw"],
        "electrolyser_mw": capacities["electrolyser_mw"],
        "pv_cf_mean_train": capacities["pv_mean_cf_training"],
        "wind_cf_mean_train": capacities["wind_mean_cf_training"],
        "pv_cf_mean_test": float(np.mean(pv_cf)),
        "wind_cf_mean_test": float(np.mean(wind_cf)),
    }
    if result is not None:
        row.update(result.__dict__)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", choices=COUNTRY_CODES, help="One country for smoke tests")
    parser.add_argument("--year", type=int, help="One evaluation year for smoke tests")
    parser.add_argument("--run-label", required=True,
                        help="Unique versioned result-folder suffix; existing non-empty folders are refused")
    parser.add_argument("--resume", action="store_true",
                        help="Resume only when run configuration, source hashes and code hashes match")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_label):
        parser.error("--run-label may contain only letters, digits, hyphens and underscores")
    out_dir = OUT_ROOT / f"eu27_rfnbo_frontier_{args.run_label}"
    if out_dir.exists() and any(out_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite non-empty results directory: {out_dir}")
    years = [args.year] if args.year is not None else list(EVALUATION_YEARS)
    if not set(years).issubset(EVALUATION_YEARS):
        parser.error(
            f"--year must be in the evaluation period "
            f"{EVALUATION_YEARS[0]}-{EVALUATION_YEARS[-1]}"
        )

    points = read_country_points()
    selected = {args.country: points[args.country]} if args.country else points
    run_configuration: dict[str, object] = {
        "project": "12",
        "scope": "EU-27 single country-reference points; not national means or plant sites",
        "train_years_local": list(TRAIN_YEARS),
        "evaluation_weather_years_local": years,
        "countries": sorted(selected),
        "monthly_matching_timezone_rule": (
            "full local YYYY-MM at the reference point; complete local calendar year with cyclic inventory boundary"
        ),
        "buffer_levels_hours": list(BUFFER_HOURS),
        "h2_reduction_targets_for_threshold": list(H2_REDUCTION_TARGETS),
        "max_nh3_buffer_hours_for_threshold": MAX_NH3_BUFFER_HOURS_FOR_THRESHOLD,
        "fixed_scenario": {
            "ammonia_delivery_tph": DEMAND_TPH,
            "hb_nameplate_tph": HB_NAMEPLATE_TPH,
            "hb_min_load_fraction_flexible": HB_MIN_LOAD_FRACTION,
            "hb_ramp_fraction_nameplate_per_hour": HB_RAMP_FRACTION_NAMEPLATE_PER_HOUR,
            "ramp_interpretation": "No additional binding ramp limit at hourly resolution; 0.60 allows the full 40%-to-100% load swing per hour. Strict-ramp stress tests are separate.",
            "specific_electricity_kwh_per_kg_h2": SEC_KWH_PER_KG,
            "hydrogen_kg_per_t_nh3": H2_PER_NH3_KG_PER_T,
            "electrolyser_nameplate_multiple_of_reference_h2_load": ELECTROLYSER_MULTIPLE,
            "renewable_overbuild_relative_to_reference_electricity": RENEWABLE_OVERBUILD,
            "wind_share_of_annual_renewable_energy": WIND_ENERGY_SHARE,
        },
        "solver": {
            "method": "scipy-highs",
            "highs_options": HIGHS_LP_OPTIONS,
        },
        "wind_curve": {"cut_in_m_s": 3.0, "rated_m_s": 12.0, "cut_out_m_s": 25.0},
    }
    code_hashes = {
        str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
        str(Path(__file__).with_name("rfnbo_inventory_frontier.py").resolve()):
            sha256_file(Path(__file__).with_name("rfnbo_inventory_frontier.py").resolve()),
        str(Path(__file__).with_name("acquire_eu27_reference_profiles.py").resolve()):
            sha256_file(Path(__file__).with_name("acquire_eu27_reference_profiles.py").resolve()),
        str(Path(__file__).with_name("audit_eu27_reference_profiles.py").resolve()):
            sha256_file(Path(__file__).with_name("audit_eu27_reference_profiles.py").resolve()),
    }
    countries: list[dict[str, object]] = []
    input_hashes: dict[str, str] = {}
    prepared: dict[str, tuple[dict[str, object], pd.Series, pd.Series, dict[str, float], Path, Path]] = {}
    for code, point in selected.items():
        pv_path = RAW / "pv" / f"{code.lower()}_pvgis_era5_2005_2023.json"
        wind_path = RAW / "wind" / f"{code.lower()}_openmeteo_era5_wind100m_2005_2023.json"
        pv = pvgis_series(code)
        wind_speed = wind_speed_series(code)
        capacities = design_capacities(str(point["timezone"]), pv, wind_speed)
        input_hashes[str(pv_path)] = sha256_file(pv_path)
        input_hashes[str(wind_path)] = sha256_file(wind_path)
        countries.append({**point, **capacities})
        prepared[code] = (point, pv, wind_speed, capacities, pv_path, wind_path)

    acquisition_manifests = sorted(
        (PROJECT / "data_manifest" / "eu27_reference_profiles").glob("acquisition_*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not acquisition_manifests:
        raise FileNotFoundError("No completed acquisition manifest is available")
    acquisition_manifest_path = acquisition_manifests[-1]
    points_hash = sha256_file(POINTS)
    boundary_path = RAW / "geography" / "CNTR_RG_01M_2024_4326.geojson"
    quality_audit_dir = PROJECT / "results" / "eu27_reference_profile_audit"
    quality_audit_paths = (
        quality_audit_dir / "eu27_input_quality_audit.json",
        quality_audit_dir / "eu27_input_quality_audit.csv",
    )
    if not all(path.is_file() for path in quality_audit_paths):
        raise FileNotFoundError("The completed EU-27 hourly profile quality audit is required")
    input_hashes[str(POINTS)] = points_hash
    input_hashes[str(boundary_path)] = sha256_file(boundary_path)
    input_hashes[str(acquisition_manifest_path)] = sha256_file(acquisition_manifest_path)
    input_hashes.update({str(path): sha256_file(path) for path in quality_audit_paths})

    manifest_path = out_dir / "model_run_manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Cannot resume without run manifest: {manifest_path}")
        manifest = read_json(manifest_path)
        if manifest.get("run_configuration") != run_configuration:
            raise ValueError("Resume refused: run configuration differs from the existing manifest")
        if manifest.get("input_sha256") != input_hashes:
            raise ValueError("Resume refused: public input hashes differ from the existing manifest")
        if manifest.get("code_sha256") != code_hashes:
            raise ValueError("Resume refused: executable source hashes differ from the existing manifest")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_root": str(RAW),
            "acquisition_manifest": str(acquisition_manifest_path),
            "run_configuration": run_configuration,
            "countries": countries,
            "input_sha256": input_hashes,
            "code_sha256": code_hashes,
            "completed_chunks": {},
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
            },
            "data_attribution_and_licence": {
                "PVGIS": "European Commission Joint Research Centre, PVGIS 5.3 / PVGIS-ERA5; cite and attribute JRC.",
                "Open-Meteo": "ERA5 archive API, CC BY 4.0 attribution; free endpoint limited to non-commercial use under the recorded terms.",
                "GISCO": "Eurostat/GISCO Countries 2024; non-commercial use; publication/maps must include © EuroGeographics for the administrative boundaries.",
            },
            "boundary": (
                "Conditional one-hour reference-plant screen. RFNBO temporal correlation is represented only as an energy-matching budget. "
                "No full additionality/geographical-correlation/GHG certification, observed plant operation, HB/ASU parasitic load, "
                "start-up/failure/reliability, network linepack, or total-plant cost is modeled."
            ),
        }
        atomic_json(manifest_path, manifest)

    for code, (point, pv, wind_speed, capacities, _, _) in prepared.items():
        timezone_name = str(point["timezone"])
        for year in years:
            chunk_key = f"{code}_{year}"
            chunk_path = out_dir / f"chunk_{code}_{year}.csv"
            existing_record = manifest["completed_chunks"].get(chunk_key)
            if chunk_path.exists():
                if not args.resume or not existing_record:
                    raise FileExistsError(f"Unregistered checkpoint found: {chunk_path}")
                chunk_hash = sha256_file(chunk_path)
                if chunk_hash != existing_record.get("sha256"):
                    raise ValueError(f"Checkpoint hash mismatch: {chunk_path}")
                print(f"Reused verified chunk {chunk_key}", flush=True)
                continue
            if existing_record:
                raise FileNotFoundError(f"Manifest checkpoint is missing: {chunk_path}")
            rows: list[dict[str, object]] = []
            index, generation, months, pv_cf, wind_cf = profile_for_year(
                year, timezone_name, pv, wind_speed,
                capacities["pv_capacity_mw"], capacities["wind_capacity_mw"],
            )
            settings = LPSettings(
                electrolyser_mw=capacities["electrolyser_mw"],
                grid_connection_mw=capacities["grid_connection_mw"],
                electrolyser_kwh_per_kg=SEC_KWH_PER_KG,
                ammonia_demand_tph=DEMAND_TPH,
                hb_nameplate_tph=HB_NAMEPLATE_TPH,
                hb_min_load_fraction=HB_MIN_LOAD_FRACTION,
                hb_ramp_fraction_nameplate_per_hour=HB_RAMP_FRACTION_NAMEPLATE_PER_HOUR,
            )
            for rule in ("monthly", "hourly"):
                rigid = None
                try:
                    rigid, _ = solve_frontier_point(
                        generation, months, rule=rule, flexible=False,
                        ammonia_buffer_limit_t=0.0, settings=settings,
                    )
                    rows.append(point_result_row(
                        point, year, rule, "rigid_reference", "rigid", 0.0,
                        rigid, capacities, generation, pv_cf, wind_cf,
                    ))
                except RuntimeError as exc:
                    rows.append(point_result_row(
                        point, year, rule, "rigid_reference", "rigid", 0.0,
                        None, capacities, generation, pv_cf, wind_cf,
                        status="infeasible_or_solver_error", error=str(exc),
                    ))
                    print(f"Failed rigid {code} {year} {rule}: {exc}", flush=True)

                for buffer_hours, process_mode in buffer_frontier_cases():
                    try:
                        result, _ = solve_frontier_point(
                            generation, months, rule=rule, flexible=True,
                            ammonia_buffer_limit_t=buffer_hours * DEMAND_TPH,
                            settings=settings,
                        )
                        rows.append(point_result_row(
                            point, year, rule, "sampled_inventory_frontier", process_mode,
                            buffer_hours, result, capacities, generation, pv_cf, wind_cf,
                        ))
                    except RuntimeError as exc:
                        rows.append(point_result_row(
                            point, year, rule, "sampled_inventory_frontier", process_mode,
                            buffer_hours, None, capacities, generation, pv_cf, wind_cf,
                            status="infeasible_or_solver_error", error=str(exc),
                        ))

                for target_reduction in H2_REDUCTION_TARGETS:
                    if rigid is None:
                        rows.append(point_result_row(
                            point, year, rule, "minimum_product_buffer_for_target",
                            "flexible_plus_nh3_buffer", MAX_NH3_BUFFER_HOURS_FOR_THRESHOLD,
                            None, capacities, generation, pv_cf, wind_cf,
                            target_h2_reduction=target_reduction,
                            status="not_applicable_rigid_reference_infeasible",
                        ))
                    elif rigid.h2_capacity_kg > 1e-6:
                        h2_target = rigid.h2_capacity_kg * (1.0 - target_reduction)
                        try:
                            threshold, _ = solve_frontier_point(
                                generation, months, rule=rule, flexible=True,
                                ammonia_buffer_limit_t=MAX_NH3_BUFFER_HOURS_FOR_THRESHOLD * DEMAND_TPH,
                                settings=settings,
                                objective_mode="min_nh3",
                                h2_capacity_limit_kg=h2_target,
                            )
                            rows.append(point_result_row(
                                point, year, rule, "minimum_product_buffer_for_target",
                                "flexible_plus_nh3_buffer", MAX_NH3_BUFFER_HOURS_FOR_THRESHOLD,
                                threshold, capacities, generation, pv_cf, wind_cf,
                                target_h2_reduction=target_reduction,
                            ))
                        except RuntimeError as exc:
                            rows.append(point_result_row(
                                point, year, rule, "minimum_product_buffer_for_target",
                                "flexible_plus_nh3_buffer", MAX_NH3_BUFFER_HOURS_FOR_THRESHOLD,
                                None, capacities, generation, pv_cf, wind_cf,
                                target_h2_reduction=target_reduction,
                                status="target_infeasible_within_buffer_limit", error=str(exc),
                            ))
                    else:
                        rows.append(point_result_row(
                            point, year, rule, "minimum_product_buffer_for_target",
                            "flexible_plus_nh3_buffer", MAX_NH3_BUFFER_HOURS_FOR_THRESHOLD,
                            None, capacities, generation, pv_cf, wind_cf,
                            target_h2_reduction=target_reduction,
                            status="not_applicable_rigid_h2_capacity_is_zero",
                        ))
            expected_chunk_rows = 2 * (1 + len(BUFFER_HOURS) + len(H2_REDUCTION_TARGETS))
            if len(rows) != expected_chunk_rows:
                raise RuntimeError(
                    f"Incomplete country-year result grid for {code}_{year}: "
                    f"expected {expected_chunk_rows} rows, got {len(rows)}"
                )
            temporary_chunk = chunk_path.with_suffix(".csv.tmp")
            pd.DataFrame(rows).to_csv(temporary_chunk, index=False, float_format="%.10g")
            temporary_chunk.replace(chunk_path)
            manifest["completed_chunks"][chunk_key] = {
                "path": str(chunk_path),
                "sha256": sha256_file(chunk_path),
                "result_rows": len(rows),
            }
            atomic_json(manifest_path, manifest)
            print(f"Completed {code} {year}: {len(rows)} rows; "
                  f"{len(manifest['completed_chunks'])}/{len(selected) * len(years)} chunks", flush=True)

    expected_chunks = {f"{code}_{year}" for code in selected for year in years}
    if set(manifest["completed_chunks"]) != expected_chunks:
        raise RuntimeError("Checkpoint set does not match the requested country-year grid")
    chunk_frames = [
        pd.read_csv(out_dir / f"chunk_{code}_{year}.csv")
        for code in selected for year in years
    ]
    final_frame = pd.concat(chunk_frames, ignore_index=True)
    expected_result_rows = len(selected) * len(years) * 2 * (
        1 + len(BUFFER_HOURS) + len(H2_REDUCTION_TARGETS)
    )
    if len(final_frame) != expected_result_rows:
        raise RuntimeError(
            f"Incomplete final result grid: expected {expected_result_rows} rows, "
            f"got {len(final_frame)}"
        )
    solver_error_rows = int((
        final_frame["status"].astype(str) == "infeasible_or_solver_error"
    ).sum())
    target_infeasible_rows = int((
        final_frame["status"].astype(str) == "target_infeasible_within_buffer_limit"
    ).sum())
    csv_path = out_dir / "eu27_rfnbo_frontier_results.csv"
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    final_frame.to_csv(temporary_csv, index=False, float_format="%.10g")
    temporary_csv.replace(csv_path)
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["result_rows"] = len(final_frame)
    manifest["expected_result_rows"] = expected_result_rows
    manifest["solver_error_rows"] = solver_error_rows
    manifest["target_infeasible_rows"] = target_infeasible_rows
    manifest["result_csv"] = str(csv_path)
    manifest["result_sha256"] = sha256_file(csv_path)
    manifest["status"] = (
        "complete" if solver_error_rows == 0 else "complete_with_solver_errors"
    )
    atomic_json(manifest_path, manifest)
    print(f"Completed chunks: {len(expected_chunks)}/{len(expected_chunks)}")
    print(f"Results: {csv_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

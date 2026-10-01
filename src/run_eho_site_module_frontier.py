"""Run a public, site-scale conditional cavern-module frontier.

The modeled reference throughput is derived from EHO's estimated 2024 annual
hydrogen output for ammonia end-use. The optimization replays historical
weather and does not estimate plant operation, outages, or cavern availability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import warnings
from functools import partial
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT = Path(r"D:\MLWork\project12_h2flex_eu")
RAW_ROOT = Path(
    r"F:\AcademicData\project12_h2flex_eu\raw\eho_ammonia_site_weather_20260930"
)
SAMPLE_PATH = PROJECT / "datasets" / "eho_ammonia_site_clusters_v1_20260930.csv"
OUT_ROOT = PROJECT / "results" / "eho_site_module_frontier"
sys.path.insert(0, str(PROJECT / "src"))
import run_eu27_cavern_module_frontier as model  # noqa: E402
import run_eu27_rfnbo_frontier as weather  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def pvgis_series(path: Path) -> pd.Series:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("inputs", {}).get("meteo_data", {}).get("meteo_db") != "ERA5":
        raise ValueError(f"PVGIS source metadata is not ERA5: {path}")
    items = data.get("outputs", {}).get("hourly", [])
    if len(items) != 166_536:
        raise ValueError(f"Unexpected PVGIS hourly record count {len(items)}: {path}")
    stamps = pd.to_datetime([item["time"] for item in items], format="%Y%m%d:%H%M", utc=True)
    if not np.all(stamps.minute == 30):
        raise ValueError(f"PVGIS ERA5 hourly timestamps are not interval midpoints: {path}")
    # PVGIS ERA5 stamps hourly means at :30; the dispatch index uses interval starts.
    index = stamps - pd.Timedelta(minutes=30)
    output = np.asarray([float(item["P"]) for item in items], dtype=float) / 1000.0
    if not np.isfinite(output).all() or np.any(output < 0) or np.any(output > 1.05):
        raise ValueError(f"Invalid PVGIS per-unit profile: {path}")
    index = pd.DatetimeIndex(index)
    if len(index) > 1 and not np.all(index[1:] - index[:-1] == pd.Timedelta(hours=1)):
        raise ValueError(f"PVGIS ERA5 timestamps are not a complete hourly sequence: {path}")
    return pd.Series(output, index=index, name="pv_cf").sort_index()


def wind_series(path: Path) -> pd.Series:
    data = json.loads(path.read_text(encoding="utf-8"))
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    values = hourly.get("wind_speed_100m", [])
    if len(times) != 166_536 or len(values) != 166_536:
        raise ValueError(f"Unexpected Open-Meteo hourly record count: {path}")
    speed = np.asarray(values, dtype=float)
    if not np.isfinite(speed).all() or np.any(speed < 0):
        raise ValueError(f"Invalid Open-Meteo wind profile: {path}")
    index = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    if len(index) > 1 and not np.all(index[1:] - index[:-1] == pd.Timedelta(hours=1)):
        raise ValueError(f"Open-Meteo ERA5 timestamps are not a complete hourly sequence: {path}")
    return pd.Series(speed, index=index, name="wind_speed_100m").sort_index()


_APPEND_COLUMNS: dict[Path, list[str]] = {}


def append_row(path: Path, row: dict[str, Any]) -> None:
    """Append records with a stable, expanding CSV schema.

    Solver diagnostics add fields only for optimal/lexicographic rows. When a
    new field first appears, prior rows are reindexed and the header is
    rewritten once; subsequent records are aligned to the saved column order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    key = path.resolve()
    if key not in _APPEND_COLUMNS:
        _APPEND_COLUMNS[key] = (
            list(pd.read_csv(path, nrows=0).columns) if path.exists() else list(row)
        )
    columns = _APPEND_COLUMNS[key]
    additions = [name for name in row if name not in columns]
    if not path.exists():
        pd.DataFrame([row]).reindex(columns=columns).to_csv(
            path, index=False, lineterminator="\n"
        )
        return
    if additions:
        previous = pd.read_csv(path)
        columns = [*columns, *additions]
        _APPEND_COLUMNS[key] = columns
        previous = previous.reindex(columns=columns)
        current = pd.DataFrame([row]).reindex(columns=columns)
        pd.concat((previous, current), ignore_index=True).to_csv(
            path, index=False, lineterminator="\n"
        )
        return
    pd.DataFrame([row]).reindex(columns=columns).to_csv(
        path, mode="a", header=False, index=False, lineterminator="\n"
    )


def task_key(row: dict[str, Any]) -> tuple[str, int, str, str, float]:
    return (
        str(row["site_id"]),
        int(row["weather_year"]),
        str(row["process_case"]),
        str(row["cavern_archetype"]),
        float(row["withdrawal_to_injection_ratio"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--solver-method", choices=["highs-ds", "highs-ipm"], default="highs-ds")
    parser.add_argument("--lp-formulation", choices=["expanded", "compact"], default="expanded")
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--site-ids", nargs="*", default=None)
    parser.add_argument("--limit-sites", type=int, default=None)
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2015, 2024)))
    parser.add_argument("--cavern-cases", nargs="+",
                        default=["high_investment_small_cavern"],
                        choices=[case.case_id for case in model.CAVERN_CASES])
    parser.add_argument("--wtir", nargs="+", type=float, default=[2.0])
    parser.add_argument("--include-rigid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-load-fractions", nargs="+", type=float, default=[0.60])
    parser.add_argument("--buffer-hours", nargs="+", type=float, default=[0.0, 24.0, 72.0])
    parser.add_argument("--hb-nameplate-multiple", type=float, default=1.50)
    parser.add_argument("--ramp-fraction-nameplate-per-hour", type=float, default=0.60)
    parser.add_argument("--storage-loss-per-hour", type=float, default=0.0)
    parser.add_argument("--compression-kwh-per-kg", type=float,
                        default=model.STORAGE_COMPRESSION_KWH_PER_KG)
    parser.add_argument("--renewable-overbuild", type=float, default=weather.RENEWABLE_OVERBUILD)
    parser.add_argument("--wind-energy-share", type=float, default=weather.WIND_ENERGY_SHARE)
    parser.add_argument("--boundary", choices=["cyclic", "zero"], default="cyclic")
    args = parser.parse_args()
    model.SOLVER_METHOD = args.solver_method
    if args.lp_formulation == "compact":
        from compact_cavern_lp import solve_compact
        model.linprog = solve_compact
    else:
        from scipy.optimize import OptimizeWarning
        warnings.filterwarnings("ignore", category=OptimizeWarning,
                                message="Unrecognized options detected.*")
        model.linprog = partial(model.linprog, options={"threads": args.solver_threads})
    if args.solver_threads < 1:
        parser.error("solver threads must be positive")

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_label):
        parser.error("run label may contain only letters, digits, hyphens and underscores")
    if not SAMPLE_PATH.is_file():
        raise FileNotFoundError(f"Missing EHO-derived site sample: {SAMPLE_PATH}")
    if not args.years or any(year not in range(2005, 2024) for year in args.years):
        parser.error("weather years must lie within the acquired 2005–2023 series")
    if len(set(args.years)) != len(args.years):
        parser.error("weather years must be unique")
    if (not args.min_load_fractions
            or any(not 0 <= value <= 1 for value in args.min_load_fractions)
            or not args.buffer_hours
            or any(value < 0 or not math.isfinite(value) for value in args.buffer_hours)
            or args.hb_nameplate_multiple < 1
            or any(value * args.hb_nameplate_multiple > 1 for value in args.min_load_fractions)
            or args.ramp_fraction_nameplate_per_hour < 0
            or not 0 <= args.storage_loss_per_hour < 1
            or args.compression_kwh_per_kg < 0
            or not math.isfinite(args.hb_nameplate_multiple)
            or not math.isfinite(args.ramp_fraction_nameplate_per_hour)
            or not math.isfinite(args.storage_loss_per_hour)
            or not math.isfinite(args.compression_kwh_per_kg)
            or not math.isfinite(args.renewable_overbuild)
            or args.renewable_overbuild <= 0
            or not math.isfinite(args.wind_energy_share)
            or not 0 <= args.wind_energy_share <= 1
            or any(not 1 <= ratio <= 5 for ratio in args.wtir)):
        parser.error("invalid process, loss, compression, or WTIR parameter")
    if args.limit_sites is not None and args.limit_sites < 1:
        parser.error("--limit-sites must be positive")

    sites = pd.read_csv(SAMPLE_PATH)
    if args.site_ids:
        missing = sorted(set(args.site_ids) - set(sites["site_id"]))
        if missing:
            parser.error(f"unknown site IDs: {missing}")
        sites = sites.loc[sites["site_id"].isin(args.site_ids)]
    if args.limit_sites is not None:
        sites = sites.head(args.limit_sites)
    if sites.empty:
        parser.error("site selection is empty")

    process_specs: list[dict[str, Any]] = []
    if args.include_rigid:
        process_specs.append({
            "name": "rigid", "flexible": False, "min_load_fraction": 1.0,
            "buffer_hours": 0.0,
        })
    for min_load in args.min_load_fractions:
        for buffer_hours in args.buffer_hours:
            load_label = f"{min_load:g}".replace(".", "p")
            buffer_label = f"{buffer_hours:g}".replace(".", "p")
            process_specs.append({
                "name": f"flex_ml{load_label}_b{buffer_label}",
                "flexible": True,
                "min_load_fraction": float(min_load),
                "buffer_hours": float(buffer_hours),
            })
    labels = [spec["name"] for spec in process_specs]
    if len(labels) != len(set(labels)):
        parser.error("duplicate process flexibility scenarios")
    if not process_specs:
        parser.error("select at least one process scenario")

    output_dir = OUT_ROOT / args.run_label
    output_path = output_dir / "site_weather_frontier.csv"
    trace_path = output_dir / "module_feasibility_trace.csv"
    manifest_path = output_dir / "run_manifest.json"
    case_map = {case.case_id: case for case in model.CAVERN_CASES}
    config = {
        "run_label": args.run_label,
        "solver_method": args.solver_method,
        "lp_formulation": args.lp_formulation,
        "solver_threads": 1 if args.lp_formulation == "compact" else args.solver_threads,
        "site_ids": sites["site_id"].tolist(),
        "years": args.years,
        "cavern_cases": args.cavern_cases,
        "wtir": args.wtir,
        "process_scenarios": process_specs,
        "hb_nameplate_multiple": args.hb_nameplate_multiple,
        "ramp_fraction_nameplate_per_hour": args.ramp_fraction_nameplate_per_hour,
        "storage_loss_per_hour": args.storage_loss_per_hour,
        "compression_kwh_per_kg": args.compression_kwh_per_kg,
        "renewable_overbuild": args.renewable_overbuild,
        "wind_energy_share": args.wind_energy_share,
        "matching_rule": "hourly electrolyser eligibility; direct grid support for site auxiliaries/compression is bounded by model connection and separately reported",
        "boundary": f"independent complete local civil year; {args.boundary} H2 and NH3 stock boundary; perfect foresight",
        "boundary_mode": args.boundary,
        "weather_timestamp_convention": (
            "PVGIS ERA5 hourly means stamped at UTC :30 are shifted to interval starts; "
            "Open-Meteo ERA5 hourly timestamps are used at UTC hour starts."
        ),
        "capacity_design_training_years": list(weather.TRAIN_YEARS),
    }
    if output_dir.exists():
        if not args.resume:
            raise FileExistsError(f"Run directory already exists; pass --resume to continue: {output_dir}")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Cannot resume without a run manifest: {manifest_path}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("configuration") != config:
            raise ValueError("Resume configuration differs from the saved run configuration")
    else:
        output_dir.mkdir(parents=True)
        code_files = [Path(__file__).resolve(), Path(model.__file__).resolve(),
                      Path(weather.__file__).resolve(),
                      Path(__import__("run_eu27_rfnbo_frontier_aux").__file__).resolve(),
                      Path(__import__("rfnbo_inventory_frontier").__file__).resolve()]
        if args.lp_formulation == "compact":
            code_files.append(Path(__import__("compact_cavern_lp").__file__).resolve())
        atomic_json(manifest_path, {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": config,
            "input_hashes": {str(SAMPLE_PATH): sha256_file(SAMPLE_PATH)},
            "code_hashes": {str(path): sha256_file(path) for path in code_files},
            "weather_raw_hashes": {},
            "completed_rows": 0,
            "status": "running",
            "interpretation_boundary": (
                "EHO annual H2 outputs are estimates; sample coverage is 15 city clusters in five EU states. "
                "Weather-year adequacy is not measured plant reliability, cavern availability, or EU-27 census inference."
            ),
        })

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for recorded_path, recorded_hash in manifest["code_hashes"].items():
        if sha256_file(Path(recorded_path)) != recorded_hash:
            raise ValueError(f"Resume code differs from recorded source: {recorded_path}")
    for recorded_path, recorded_hash in manifest["input_hashes"].items():
        if sha256_file(Path(recorded_path)) != recorded_hash:
            raise ValueError(f"Resume input differs from recorded input: {recorded_path}")
    completed: set[tuple[str, int, str, str, float]] = set()
    if output_path.exists():
        old = pd.read_csv(output_path)
        completed = {task_key(row) for row in old.to_dict(orient="records")}
    task_total = len(sites) * len(args.years) * len(process_specs) * len(args.cavern_cases) * len(args.wtir)
    print(f"Starting/resuming {task_total} site-year-process-storage scenarios", flush=True)

    for site in sites.to_dict(orient="records"):
        site_id = str(site["site_id"])
        pv_path = RAW_ROOT / "pvgis" / f"{site_id}_pvgis_era5_2005_2023.json"
        wind_path = RAW_ROOT / "open_meteo" / f"{site_id}_wind_era5_2005_2023.json"
        if not pv_path.is_file() or not wind_path.is_file():
            raise FileNotFoundError(f"Missing raw weather profile for {site_id}")
        manifest["weather_raw_hashes"][str(pv_path)] = sha256_file(pv_path)
        manifest["weather_raw_hashes"][str(wind_path)] = sha256_file(wind_path)
        atomic_json(manifest_path, manifest)

        pv = pvgis_series(pv_path)
        wind = wind_series(wind_path)
        timezone_name = str(site["timezone"])
        capacities = model.design_capacities_with_aux(
            timezone_name, pv, wind, model.AUX_MWH_PER_T_DEFAULT,
            renewable_overbuild=args.renewable_overbuild,
            wind_energy_share=args.wind_energy_share,
        )
        site_tph = float(site["nh3_equivalent_service_tph"])
        if not math.isfinite(site_tph) or site_tph <= 0:
            raise ValueError(f"Invalid EHO-derived NH3-equivalent throughput for {site_id}")

        for year in args.years:
            _, generation, month_labels, pv_cf, wind_cf = weather.profile_for_year(
                year,
                timezone_name,
                pv,
                wind,
                capacities["pv_capacity_mw"],
                capacities["wind_capacity_mw"],
            )
            period_lengths = [len(generation)]
            free_results: dict[str, dict[str, Any]] = {}
            for process_spec in process_specs:
                process_case = str(process_spec["name"])
                if all((site_id, int(year), process_case, case_id, float(wtir)) in completed
                       for case_id in args.cavern_cases for wtir in args.wtir):
                    continue
                flexible = bool(process_spec["flexible"])
                min_load = float(process_spec["min_load_fraction"])
                buffer_hours = float(process_spec["buffer_hours"])
                free_results[process_case] = model.solve_shared_horizon(
                    generation,
                    month_labels,
                    period_lengths,
                    electrolyser_mw=capacities["electrolyser_mw"],
                    grid_connection_mw=capacities["grid_connection_mw"],
                    min_hb_load_fraction=min_load,
                    flexible=flexible,
                    nh3_buffer_limit_hours=buffer_hours,
                    matching_rule="hourly",
                    boundary=args.boundary,
                    auxiliary_mwh_per_t_nh3=model.AUX_MWH_PER_T_DEFAULT,
                    storage_loss_per_hour=args.storage_loss_per_hour,
                    compression_kwh_per_kg=args.compression_kwh_per_kg,
                    hb_nameplate_tph=args.hb_nameplate_multiple * weather.DEMAND_TPH,
                    ramp_fraction_of_nameplate_per_hour=args.ramp_fraction_nameplate_per_hour,
                    cavern=None,
                    cavern_count=None,
                    # Only the minimum H2 inventory is used from this unconstrained baseline.
                    # The module-constrained run below performs the full lexicographic solve.
                    lexicographic=False,
                )
                if free_results[process_case].get("status") != "optimal":
                    raise RuntimeError(f"Free-inventory benchmark failed: {site_id}/{year}/{process_case}")

                for case_id in args.cavern_cases:
                    reference = case_map[case_id]
                    site_cavern = model.CavernCase(
                        reference.case_id,
                        reference.working_gas_million_sm3,
                        reference.peak_withdrawal_million_sm3_day,
                        reference.reference_site_cavern_count,
                        service_reference_nh3_tph=site_tph,
                    )
                    for wtir in args.wtir:
                        key = (site_id, int(year), process_case, case_id, float(wtir))
                        if key in completed:
                            continue
                        minimum, trace, final = model.minimum_feasible_modules(
                            generation,
                            month_labels,
                            period_lengths,
                            cavern=site_cavern,
                            max_modules=reference.reference_site_cavern_count,
                            electrolyser_mw=capacities["electrolyser_mw"],
                            grid_connection_mw=capacities["grid_connection_mw"],
                            min_load_fraction=min_load,
                            flexible=flexible,
                            buffer_hours=buffer_hours,
                            matching_rule="hourly",
                            country=site_id,
                            boundary=args.boundary,
                            auxiliary_mwh_per_t_nh3=model.AUX_MWH_PER_T_DEFAULT,
                            storage_loss_per_hour=args.storage_loss_per_hour,
                            compression_kwh_per_kg=args.compression_kwh_per_kg,
                            withdrawal_to_injection_ratio=float(wtir),
                            hb_nameplate_tph=args.hb_nameplate_multiple * weather.DEMAND_TPH,
                            ramp_fraction_of_nameplate_per_hour=args.ramp_fraction_nameplate_per_hour,
                        )
                        if any(item.get("status") == "solver_failed" for item in trace):
                            raise RuntimeError(f"Module search solver failure: {site_id}/{year}/{process_case}")
                        if final is not None and final.get("status") != "optimal":
                            raise RuntimeError(f"Final lexicographic solve failed: {site_id}/{year}/{process_case}")
                        no_cavern = free_results[process_case]
                        row: dict[str, Any] = {
                            "site_id": site_id,
                            "country": site["country"],
                            "city_cluster": site["city_cluster"],
                            "project_names": site["project_names"],
                            "process_types": site["process_types"],
                            "eho_record_count": int(site["eho_record_count"]),
                            "eho_reporting_year": int(site["eho_year"]),
                            "eho_h2_output_t_per_year_estimate": float(site["h2_output_t_per_year_estimate"]),
                            "service_reference_nh3_tph": site_tph,
                            "weather_year": int(year),
                            "local_year_hours": len(generation),
                            "process_case": process_case,
                            "flexible_process": flexible,
                            "min_hb_load_fraction_of_nameplate": min_load,
                            "hb_nameplate_multiple": args.hb_nameplate_multiple,
                            "ramp_fraction_nameplate_per_hour": args.ramp_fraction_nameplate_per_hour,
                            "nh3_buffer_limit_hours": buffer_hours,
                            "cavern_archetype": case_id,
                            "module_count_limit": reference.reference_site_cavern_count,
                            "withdrawal_to_injection_ratio": float(wtir),
                            "minimum_feasible_modules": minimum,
                            "status": ("optimal" if final is not None else "not_feasible_within_reference_module_limit"),
                            "module_reference_service_hours_at_site_scale": site_cavern.reference_service_hours,
                            "free_storage_status": no_cavern.get("status"),
                            "free_h2_storage_service_hours": no_cavern.get("h2_storage_capacity_reference_service_hours"),
                            "free_h2_storage_kg_per_normalized_tph": no_cavern.get("h2_storage_capacity_kg_per_1_tph_nh3"),
                            "pv_capacity_mw_per_normalized_tph": capacities["pv_capacity_mw"],
                            "wind_capacity_mw_per_normalized_tph": capacities["wind_capacity_mw"],
                            "electrolyser_capacity_mw_per_normalized_tph": capacities["electrolyser_mw"],
                            "grid_connection_mw_per_normalized_tph": capacities["grid_connection_mw"],
                            "renewable_overbuild": capacities["renewable_overbuild"],
                            "wind_energy_share": capacities["wind_energy_share"],
                            "training_pv_cf_mean": capacities["pv_mean_cf_training"],
                            "training_wind_cf_mean": capacities["wind_mean_cf_training"],
                            "evaluation_pv_cf_mean": float(np.mean(pv_cf)),
                            "evaluation_wind_cf_mean": float(np.mean(wind_cf)),
                        }
                        if final is not None:
                            h2_kg = float(final["h2_storage_capacity_kg_per_1_tph_nh3"])
                            nh3_t = float(final["nh3_buffer_capacity_t_per_1_tph_nh3"])
                            row.update({
                                "h2_storage_service_hours": float(final["h2_storage_capacity_reference_service_hours"]),
                                "h2_storage_kg_at_site_scale": h2_kg * site_tph,
                                "h2_storage_tonnes_at_site_scale": h2_kg * site_tph / 1000.0,
                                "nh3_buffer_tonnes_at_site_scale": nh3_t * site_tph,
                            "annual_nh3_equivalent_t_at_site_scale": site_tph * len(generation),
                                "grid_import_mwh_per_normalized_tph": float(final["grid_import_mwh"]),
                                "grid_import_mwh_at_site_scale": float(final["grid_import_mwh"]) * site_tph,
                                "max_h2_balance_residual_kgph": float(final["max_h2_balance_residual_kgph"]),
                                "max_h2_storage_balance_residual_kg": float(final["max_h2_storage_balance_residual_kg"]),
                                "max_nh3_balance_residual_tph": float(final["max_nh3_balance_residual_tph"]),
                                "max_annual_nh3_service_residual_t_per_normalized_tph": float(final["max_annual_nh3_service_residual_t"]),
                                "max_hb_ramp_violation_tph_per_hour": float(final["max_hb_ramp_violation_tph_per_hour"]),
                                "max_site_power_violation_mw": float(final["max_site_power_violation_mw"]),
                                "max_hourly_matching_violation_mw": float(final["max_matching_violation_native"]),
                                "max_h2_charge_kgph_per_normalized_tph": float(final["max_h2_charge_kgph"]),
                                "max_h2_discharge_kgph_per_normalized_tph": float(final["max_h2_discharge_kgph"]),
                                "peak_nh3_buffer_use_t_per_normalized_tph": float(final["peak_nh3_buffer_use_t_per_1_tph_nh3"]),
                                "solver_method": final["solver_method"],
                            })
                        append_row(output_path, row)
                        completed.add(key)
                        for item in trace:
                            trace_row = {
                                "site_id": site_id,
                                "weather_year": int(year),
                                "process_case": process_case,
                                "cavern_archetype": case_id,
                                "withdrawal_to_injection_ratio": float(wtir),
                                "service_reference_nh3_tph": site_tph,
                                **item,
                            }
                            append_row(trace_path, trace_row)
                        manifest["completed_rows"] = len(completed)
                        manifest["last_completed_key"] = list(key)
                        manifest["status"] = "running"
                        atomic_json(manifest_path, manifest)
                        print(
                            f"{site_id} {year} {process_case} {case_id} WTIR={wtir:g}: "
                            f"min modules={minimum}",
                            flush=True,
                        )
            del generation, month_labels, pv_cf, wind_cf
        del pv, wind

    manifest["status"] = "complete" if len(completed) == task_total else "partial"
    manifest["completed_rows"] = len(completed)
    manifest["task_total"] = task_total
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(manifest_path, manifest)
    print(json.dumps({
        "output_dir": str(output_dir),
        "result_csv": str(output_path),
        "trace_csv": str(trace_path),
        "completed_rows": len(completed),
        "task_total": task_total,
        "status": manifest["status"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

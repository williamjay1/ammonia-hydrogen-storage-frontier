"""Run the EU-27 reference frontier with explicit HB/ASU power consumption.

This is a separately versioned corrective model. It does not alter the active
v1 baseline run or its source files. Data preparation helpers are imported from
the v1 runner; all LP optimization is performed by the auxiliary-aware solver.
"""

from __future__ import annotations

import argparse
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
    from . import run_eu27_rfnbo_frontier as base
    from .rfnbo_inventory_frontier_aux import SOLVER_METHOD, solve_frontier_point_aux
    from .rfnbo_inventory_frontier import H2_PER_NH3_KG_PER_T
except ImportError:
    import run_eu27_rfnbo_frontier as base
    from rfnbo_inventory_frontier_aux import SOLVER_METHOD, solve_frontier_point_aux
    from rfnbo_inventory_frontier import H2_PER_NH3_KG_PER_T


PROJECT = base.PROJECT
RAW = base.RAW
OUT_ROOT = base.OUT_ROOT
AUX_MWH_PER_T_DEFAULT = (1.3 + 0.3) / 3.6  # 1.6 GJ/t = 0.4444 MWh/t
AUX_SOURCE_DOI = "10.1038/s44286-025-00207-9"


def design_capacities_with_aux(
    timezone_name: str,
    pv: pd.Series,
    wind_speed: pd.Series,
    auxiliary_mwh_per_t_nh3: float,
    *,
    renewable_overbuild: float = base.RENEWABLE_OVERBUILD,
    wind_energy_share: float = base.WIND_ENERGY_SHARE,
) -> dict[str, float]:
    """Size the weather-profile renewable system on total modeled site load.

    Electrolyser nameplate remains based on the H2-conversion load; renewable
    generation and grid interconnection scale with H2 plus HB/ASU average load.
    Annual average q equals fixed NH3 delivery under the cyclic inventory
    balance, so the q-proportional auxiliary energy has a transparent basis.
    """
    h2_only = base.design_capacities(
        timezone_name, pv, wind_speed,
        renewable_overbuild=renewable_overbuild,
        wind_energy_share=wind_energy_share,
    )
    h2_mwh_per_t = H2_PER_NH3_KG_PER_T * base.SEC_KWH_PER_KG / 1000.0
    site_mwh_per_t = h2_mwh_per_t + float(auxiliary_mwh_per_t_nh3)
    scale = site_mwh_per_t / h2_mwh_per_t
    return {
        **h2_only,
        "pv_capacity_mw": h2_only["pv_capacity_mw"] * scale,
        "wind_capacity_mw": h2_only["wind_capacity_mw"] * scale,
        "grid_connection_mw": base.ELECTROLYSER_MULTIPLE * base.DEMAND_TPH * site_mwh_per_t,
        "auxiliary_mwh_per_t_nh3": float(auxiliary_mwh_per_t_nh3),
        "reference_h2_electricity_mwh_per_t_nh3": h2_mwh_per_t,
        "reference_total_site_electricity_mwh_per_t_nh3": site_mwh_per_t,
        "renewable_capacity_scaling_vs_h2_only": scale,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", choices=base.COUNTRY_CODES,
                        help="One country for pilot or sensitivity calculations")
    parser.add_argument("--year", type=int, help="One evaluation year for pilot runs")
    parser.add_argument("--run-label", required=True,
                        help="Unique versioned result folder; non-empty folders are not overwritten")
    parser.add_argument("--resume", action="store_true",
                        help="Resume only when manifest, input and code hashes match")
    parser.add_argument("--min-load", type=float, default=0.40,
                        help="Flexible HB minimum as a fraction of nameplate (default 0.40)")
    parser.add_argument("--aux-mwh-per-t", type=float, default=AUX_MWH_PER_T_DEFAULT,
                        help="HB/ASU throughput-proportional electric load, MWh/t NH3")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_label):
        parser.error("--run-label may contain only letters, digits, hyphens and underscores")
    if not 0 <= args.min_load <= 1 or args.aux_mwh_per_t < 0:
        parser.error("minimum load must be in [0,1] and auxiliary load must be non-negative")
    out_dir = OUT_ROOT / f"eu27_rfnbo_frontier_{args.run_label}"
    if out_dir.exists() and any(out_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite non-empty results directory: {out_dir}")
    years = [args.year] if args.year is not None else list(base.EVALUATION_YEARS)
    if not set(years).issubset(base.EVALUATION_YEARS):
        parser.error(f"--year must be in {base.EVALUATION_YEARS[0]}-{base.EVALUATION_YEARS[-1]}")

    points = base.read_country_points()
    selected = {args.country: points[args.country]} if args.country else points
    run_configuration: dict[str, object] = {
        "project": "12",
        "model_version": "v3_aux_power_electrolyser_matching",
        "solver_method": SOLVER_METHOD,
        "scope": "EU-27 one public country reference point; not plant observations or national means",
        "train_years_local": list(base.TRAIN_YEARS),
        "evaluation_weather_years_local": years,
        "countries": sorted(selected),
        "temporal_matching": (
            "monthly or hourly temporal-eligibility proxy applied to electrolyser input only; "
            "HB/ASU auxiliary electricity remains in the site power balance but is not treated "
            "as electrolyser input; not a full RFNBO certification calculation"
        ),
        "buffer_levels_hours": list(base.BUFFER_HOURS),
        "h2_reduction_targets": list(base.H2_REDUCTION_TARGETS),
        "maximum_product_buffer_hours_for_threshold": base.MAX_NH3_BUFFER_HOURS_FOR_THRESHOLD,
        "fixed_scenario": {
            "ammonia_delivery_tph": base.DEMAND_TPH,
            "hb_nameplate_tph": base.HB_NAMEPLATE_TPH,
            "flexible_hb_minimum_load_fraction": args.min_load,
            "hourly_ramp_fraction_of_nameplate": base.HB_RAMP_FRACTION_NAMEPLATE_PER_HOUR,
            "h2_specific_electricity_kwh_per_kg": base.SEC_KWH_PER_KG,
            "h2_kg_per_t_nh3": H2_PER_NH3_KG_PER_T,
            "hb_asu_auxiliary_electricity_mwh_per_t_nh3": args.aux_mwh_per_t,
            "auxiliary_source": {
                "doi": AUX_SOURCE_DOI,
                "basis": "1.3 GJ/t N2 production + 0.3 GJ/t HB loop, converted to 0.4444 MWh/t",
                "status": "literature-derived modeled parameter; not facility measurement",
            },
            "electrolyser_nameplate_multiple": base.ELECTROLYSER_MULTIPLE,
            "renewable_overbuild_on_total_site_energy": base.RENEWABLE_OVERBUILD,
            "wind_share_of_annual_renewable_energy": base.WIND_ENERGY_SHARE,
        },
        "wind_curve": {"cut_in_m_s": 3.0, "rated_m_s": 12.0, "cut_out_m_s": 25.0},
    }

    code_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("rfnbo_inventory_frontier_aux.py").resolve(),
        Path(__file__).with_name("run_eu27_rfnbo_frontier.py").resolve(),
        Path(__file__).with_name("rfnbo_inventory_frontier.py").resolve(),
        Path(__file__).with_name("acquire_eu27_reference_profiles.py").resolve(),
        Path(__file__).with_name("audit_eu27_reference_profiles.py").resolve(),
    ]
    code_hashes = {str(path): base.sha256_file(path) for path in code_paths}

    countries: list[dict[str, object]] = []
    input_hashes: dict[str, str] = {}
    prepared: dict[str, tuple[dict[str, object], pd.Series, pd.Series, dict[str, float], Path, Path]] = {}
    for code, point in selected.items():
        pv_path = RAW / "pv" / f"{code.lower()}_pvgis_era5_2005_2023.json"
        wind_path = RAW / "wind" / f"{code.lower()}_openmeteo_era5_wind100m_2005_2023.json"
        pv = base.pvgis_series(code)
        wind_speed = base.wind_speed_series(code)
        capacities = design_capacities_with_aux(
            str(point["timezone"]), pv, wind_speed, args.aux_mwh_per_t
        )
        input_hashes[str(pv_path)] = base.sha256_file(pv_path)
        input_hashes[str(wind_path)] = base.sha256_file(wind_path)
        countries.append({**point, **capacities})
        prepared[code] = (point, pv, wind_speed, capacities, pv_path, wind_path)

    acquisition_manifests = sorted(
        (PROJECT / "data_manifest" / "eu27_reference_profiles").glob("acquisition_*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not acquisition_manifests:
        raise FileNotFoundError("No completed acquisition manifest is available")
    acquisition_manifest_path = acquisition_manifests[-1]
    points_path = base.POINTS
    boundary_path = RAW / "geography" / "CNTR_RG_01M_2024_4326.geojson"
    audit_dir = PROJECT / "results" / "eu27_reference_profile_audit"
    audit_paths = (audit_dir / "eu27_input_quality_audit.json",
                   audit_dir / "eu27_input_quality_audit.csv")
    if not all(path.is_file() for path in audit_paths):
        raise FileNotFoundError("A complete EU-27 input profile audit is required")
    input_hashes[str(points_path)] = base.sha256_file(points_path)
    input_hashes[str(boundary_path)] = base.sha256_file(boundary_path)
    input_hashes[str(acquisition_manifest_path)] = base.sha256_file(acquisition_manifest_path)
    input_hashes.update({str(path): base.sha256_file(path) for path in audit_paths})

    manifest_path = out_dir / "model_run_manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Cannot resume without run manifest: {manifest_path}")
        manifest = base.read_json(manifest_path)
        if manifest.get("run_configuration") != run_configuration:
            raise ValueError("Resume refused: run configuration differs from manifest")
        if manifest.get("input_sha256") != input_hashes:
            raise ValueError("Resume refused: public input hashes differ from manifest")
        if manifest.get("code_sha256") != code_hashes:
            raise ValueError("Resume refused: source hashes differ from manifest")
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
            "software": {"python": sys.version, "platform": platform.platform(),
                         "numpy": np.__version__, "pandas": pd.__version__,
                         "scipy": scipy.__version__},
            "data_attribution_and_licence": {
                "PVGIS": "European Commission JRC, PVGIS 5.3 / PVGIS-ERA5; attribution required",
                "Open-Meteo": "ERA5 Historical Weather API, CC BY 4.0 attribution and non-commercial-use condition recorded",
                "GISCO": "Eurostat/GISCO Countries 2024; non-commercial use; © EuroGeographics for boundaries",
            },
            "boundary": (
                "The modeled site load includes electrolyser electricity plus literature-bounded, throughput-proportional HB/ASU electricity. "
                "H2 compression beyond the SEC, full process heat, start-up/shutdown, failures, reliability, linepack, storage power/pressure, "
                "additionality/geographical/GHG RFNBO certification, observed plant operation, and total-plant cost are excluded. "
                "Each weather year is cyclic and independently re-optimized with perfect foresight."
            ),
        }
        base.atomic_json(manifest_path, manifest)

    for code, (point, pv, wind_speed, capacities, _, _) in prepared.items():
        zone = str(point["timezone"])
        for year in years:
            key = f"{code}_{year}"
            chunk_path = out_dir / f"chunk_{code}_{year}.csv"
            registered = manifest["completed_chunks"].get(key)
            if chunk_path.exists():
                if not args.resume or not registered:
                    raise FileExistsError(f"Unregistered checkpoint: {chunk_path}")
                if base.sha256_file(chunk_path) != registered.get("sha256"):
                    raise ValueError(f"Checkpoint hash mismatch: {chunk_path}")
                print(f"Reused verified chunk {key}", flush=True)
                continue
            if registered:
                raise FileNotFoundError(f"Manifest checkpoint missing: {chunk_path}")
            index, generation, months, pv_cf, wind_cf = base.profile_for_year(
                year, zone, pv, wind_speed,
                capacities["pv_capacity_mw"], capacities["wind_capacity_mw"],
            )
            settings = base.LPSettings(
                electrolyser_mw=capacities["electrolyser_mw"],
                grid_connection_mw=capacities["grid_connection_mw"],
                electrolyser_kwh_per_kg=base.SEC_KWH_PER_KG,
                ammonia_demand_tph=base.DEMAND_TPH,
                hb_nameplate_tph=base.HB_NAMEPLATE_TPH,
                hb_min_load_fraction=args.min_load,
                hb_ramp_fraction_nameplate_per_hour=base.HB_RAMP_FRACTION_NAMEPLATE_PER_HOUR,
            )
            rows: list[dict[str, object]] = []
            for rule in ("monthly", "hourly"):
                rigid = None
                try:
                    rigid, _ = solve_frontier_point_aux(
                        generation, months, rule=rule, flexible=False,
                        ammonia_buffer_limit_t=0.0, settings=settings,
                        auxiliary_mwh_per_t_nh3=args.aux_mwh_per_t,
                    )
                    rows.append(base.point_result_row(
                        point, year, rule, "rigid_reference", "rigid", 0.0,
                        rigid, capacities, generation, pv_cf, wind_cf,
                    ))
                except RuntimeError as exc:
                    rows.append(base.point_result_row(
                        point, year, rule, "rigid_reference", "rigid", 0.0,
                        None, capacities, generation, pv_cf, wind_cf,
                        status="infeasible_or_solver_error", error=str(exc),
                    ))
                for buffer_hours, process_mode in base.buffer_frontier_cases():
                    try:
                        result, _ = solve_frontier_point_aux(
                            generation, months, rule=rule, flexible=True,
                            ammonia_buffer_limit_t=buffer_hours * base.DEMAND_TPH,
                            settings=settings,
                            auxiliary_mwh_per_t_nh3=args.aux_mwh_per_t,
                        )
                        rows.append(base.point_result_row(
                            point, year, rule, "sampled_inventory_frontier", process_mode,
                            buffer_hours, result, capacities, generation, pv_cf, wind_cf,
                        ))
                    except RuntimeError as exc:
                        rows.append(base.point_result_row(
                            point, year, rule, "sampled_inventory_frontier", process_mode,
                            buffer_hours, None, capacities, generation, pv_cf, wind_cf,
                            status="infeasible_or_solver_error", error=str(exc),
                        ))
                for target in base.H2_REDUCTION_TARGETS:
                    if rigid is None or rigid.h2_capacity_kg <= 1e-6:
                        state = ("not_applicable_rigid_reference_infeasible" if rigid is None
                                 else "not_applicable_rigid_h2_capacity_is_zero")
                        rows.append(base.point_result_row(
                            point, year, rule, "minimum_product_buffer_for_target",
                            "flexible_plus_nh3_buffer",
                            base.MAX_NH3_BUFFER_HOURS_FOR_THRESHOLD,
                            None, capacities, generation, pv_cf, wind_cf,
                            target_h2_reduction=target, status=state,
                        ))
                        continue
                    h2_target = rigid.h2_capacity_kg * (1.0 - target)
                    try:
                        threshold, _ = solve_frontier_point_aux(
                            generation, months, rule=rule, flexible=True,
                            ammonia_buffer_limit_t=(base.MAX_NH3_BUFFER_HOURS_FOR_THRESHOLD
                                                    * base.DEMAND_TPH),
                            settings=settings, auxiliary_mwh_per_t_nh3=args.aux_mwh_per_t,
                            objective_mode="min_nh3", h2_capacity_limit_kg=h2_target,
                        )
                        rows.append(base.point_result_row(
                            point, year, rule, "minimum_product_buffer_for_target",
                            "flexible_plus_nh3_buffer",
                            base.MAX_NH3_BUFFER_HOURS_FOR_THRESHOLD,
                            threshold, capacities, generation, pv_cf, wind_cf,
                            target_h2_reduction=target,
                        ))
                    except RuntimeError as exc:
                        rows.append(base.point_result_row(
                            point, year, rule, "minimum_product_buffer_for_target",
                            "flexible_plus_nh3_buffer",
                            base.MAX_NH3_BUFFER_HOURS_FOR_THRESHOLD,
                            None, capacities, generation, pv_cf, wind_cf,
                            target_h2_reduction=target,
                            status="target_infeasible_within_buffer_limit", error=str(exc),
                        ))
            for row in rows:
                row["grid_connection_mw"] = capacities["grid_connection_mw"]
                row["hb_asu_auxiliary_mwh_per_t_nh3"] = args.aux_mwh_per_t
                row["reference_total_site_electricity_mwh_per_t_nh3"] = capacities[
                    "reference_total_site_electricity_mwh_per_t_nh3"
                ]
            expected = 2 * (1 + len(base.BUFFER_HOURS) + len(base.H2_REDUCTION_TARGETS))
            if len(rows) != expected:
                raise RuntimeError(f"Incomplete {key}: expected {expected} rows, got {len(rows)}")
            pd.DataFrame(rows).to_csv(chunk_path, index=False, float_format="%.10g")
            manifest["completed_chunks"][key] = {
                "path": str(chunk_path), "sha256": base.sha256_file(chunk_path),
                "result_rows": len(rows),
            }
            base.atomic_json(manifest_path, manifest)
            print(f"Completed {key}: {len(rows)} rows; "
                  f"{len(manifest['completed_chunks'])}/{len(selected) * len(years)} chunks", flush=True)

    expected_chunks = {f"{code}_{year}" for code in selected for year in years}
    if set(manifest["completed_chunks"]) != expected_chunks:
        raise RuntimeError("Checkpoint set does not match requested country-year grid")
    frame = pd.concat([
        pd.read_csv(out_dir / f"chunk_{code}_{year}.csv")
        for code in selected for year in years
    ], ignore_index=True)
    expected_rows = len(selected) * len(years) * 2 * (
        1 + len(base.BUFFER_HOURS) + len(base.H2_REDUCTION_TARGETS)
    )
    if len(frame) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} final rows, got {len(frame)}")
    solver_errors = int((frame["status"].astype(str) == "infeasible_or_solver_error").sum())
    target_infeasible = int((frame["status"].astype(str) == "target_infeasible_within_buffer_limit").sum())
    csv_path = out_dir / "eu27_rfnbo_frontier_results.csv"
    frame.to_csv(csv_path, index=False, float_format="%.10g")
    manifest.update({
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_rows": len(frame), "expected_result_rows": expected_rows,
        "solver_error_rows": solver_errors, "target_infeasible_rows": target_infeasible,
        "result_csv": str(csv_path), "result_sha256": base.sha256_file(csv_path),
        "status": "complete" if solver_errors == 0 else "complete_with_solver_errors",
    })
    base.atomic_json(manifest_path, manifest)
    print(f"Completed chunks: {len(expected_chunks)}/{len(expected_chunks)}")
    print(f"Results: {csv_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

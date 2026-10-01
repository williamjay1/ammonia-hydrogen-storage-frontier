"""One-factor robustness of the predeclared single-reference-module threshold.

Four scenarios vary the assumed injection/withdrawal ratio or the NH3-buffer
ceiling. Site anchors and weather years are fixed ex ante by EHO scale rank.
Results are conditional screening thresholds, not plant operating limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import run_eho_site_module_frontier as site_model


ROOT = site_model.PROJECT
OUT = ROOT / "results" / "eho_threshold_sensitivity" / "load_threshold_robustness_v1"
SAMPLE = site_model.SAMPLE_PATH
YEARS = (2015, 2019, 2023)
MIN_LOAD, MAX_LOAD, TOL = 0.40, 2.0 / 3.0, 0.025
RAMP = 0.60
CAVERN_ID = "high_investment_small_cavern"
SCENARIOS = (
    {"scenario": "central_wtir2_buffer24", "wtir": 2.0, "buffer_hours": 24.0},
    {"scenario": "wtir_1_buffer24", "wtir": 1.0, "buffer_hours": 24.0},
    {"scenario": "wtir_5_buffer24", "wtir": 5.0, "buffer_hours": 24.0},
    {"scenario": "wtir_2_buffer0", "wtir": 2.0, "buffer_hours": 0.0},
    {"scenario": "wtir_2_buffer48", "wtir": 2.0, "buffer_hours": 48.0},
    {"scenario": "exact_table14_rate", "wtir": 2.0, "buffer_hours": 24.0,
     "withdrawal_rate_million_sm3_day": 1.3625},
    {"scenario": "mid_cavern", "wtir": 2.0, "buffer_hours": 24.0,
     "cavern_id": "mid_investment_cavern"},
    {"scenario": "large_cavern", "wtir": 2.0, "buffer_hours": 24.0,
     "cavern_id": "low_investment_large_cavern"},
    {"scenario": "slow_ramp", "wtir": 2.0, "buffer_hours": 24.0,
     "ramp_fraction_nameplate_per_hour": 0.03},
    {"scenario": "overbuild2", "wtir": 2.0, "buffer_hours": 24.0,
     "renewable_overbuild": 2.0},
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def feasible(
    generation: np.ndarray,
    months: np.ndarray,
    capacities: dict[str, float],
    cavern: site_model.model.CavernCase,
    floor: float,
    wtir: float,
    buffer_hours: float,
    ramp: float = RAMP,
) -> tuple[bool, str]:
    c, a_ub, b_ub, a_eq, b_eq, bounds, idx = site_model.model.build_lp(
        generation,
        months,
        [len(generation)],
        electrolyser_mw=capacities["electrolyser_mw"],
        grid_connection_mw=capacities["grid_connection_mw"],
        min_hb_load_fraction=floor,
        flexible=True,
        nh3_buffer_limit_hours=buffer_hours,
        matching_rule="hourly",
        boundary="cyclic",
        auxiliary_mwh_per_t_nh3=site_model.model.AUX_MWH_PER_T_DEFAULT,
        storage_loss_per_hour=0.0,
        compression_kwh_per_kg=site_model.model.STORAGE_COMPRESSION_KWH_PER_KG,
        withdrawal_to_injection_ratio=wtir,
        hb_nameplate_tph=1.5 * site_model.weather.DEMAND_TPH,
        ramp_fraction_of_nameplate_per_hour=ramp,
        cavern=cavern,
        cavern_count=1,
    )
    # A threshold asks whether any admissible schedule exists. It does not
    # require a secondary minimum-inventory objective at every bisection step.
    result = site_model.model.linprog(
        np.zeros_like(c), A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
        bounds=bounds, method=site_model.model.SOLVER_METHOD)
    status = "optimal" if result.success else ("infeasible" if result.status == 2 else "solver_failed")
    if status not in {"optimal", "infeasible"}:
        raise RuntimeError(f"Threshold solve failed: {result}")
    if status == "optimal":
        residuals = [float(np.max(np.abs(a_eq @ result.x - b_eq))),
                     float(max(0, np.max(a_ub @ result.x - b_ub)))]
        for value, (lower, upper) in zip(result.x, bounds):
            if lower is not None:
                residuals.append(max(0.0, lower-value))
            if upper is not None:
                residuals.append(max(0.0, value-upper))
        if max(residuals, default=0.0) > 1e-5:
            raise RuntimeError(f"Threshold primal residual failed: {residuals}")
    return status == "optimal", status


def threshold(
    generation: np.ndarray,
    months: np.ndarray,
    capacities: dict[str, float],
    cavern: site_model.model.CavernCase,
    wtir: float,
    buffer_hours: float,
    ramp: float = RAMP,
) -> dict[str, object]:
    at_low, status_low = feasible(
        generation, months, capacities, cavern, MIN_LOAD, wtir, buffer_hours, ramp
    )
    at_high, status_high = feasible(
        generation, months, capacities, cavern, MAX_LOAD, wtir, buffer_hours, ramp
    )
    calls = 2
    lo = hi = np.nan
    if not at_low:
        classification = "infeasible_at_lower_test_bound"
    elif at_high:
        lo, hi = MAX_LOAD, np.nan
        classification = "feasible_at_upper_test_bound"
    else:
        lo, hi = MIN_LOAD, MAX_LOAD
        while hi - lo > TOL:
            mid = (lo + hi) / 2
            ok, _ = feasible(generation, months, capacities, cavern, mid, wtir, buffer_hours, ramp)
            calls += 1
            if ok:
                lo = mid
            else:
                hi = mid
        classification = "crossing_bracketed"
    return {
        "minimum_load_fraction_lower_feasible": lo,
        "minimum_load_fraction_upper_infeasible": hi,
        "feasible_at_40pct_minimum_load": at_low,
        "feasible_at_66p7pct_minimum_load": at_high,
        "status_at_40pct": status_low,
        "status_at_66p7pct": status_high,
        "threshold_classification": classification,
        "solver_calls": calls,
    }


def atomic_manifest(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + f".{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(12):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            time.sleep(0.25 * (attempt + 1))
    # Keep the completed rows and a recoverable manifest if another process
    # briefly holds the destination. This non-atomic fallback is small JSON.
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.unlink(missing_ok=True)


def main() -> int:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-label", default="cyclic_compact_threshold_v2")
    parser.add_argument("--anchor", choices=["low", "median", "high"], default=None)
    args = parser.parse_args()
    if not args.run_label.replace("_", "").replace("-", "").isalnum():
        parser.error("Invalid run label")
    OUT = ROOT / "results" / "eho_threshold_sensitivity" / args.run_label
    from compact_cavern_lp import solve_compact
    site_model.model.linprog = solve_compact
    site_model.model.SOLVER_METHOD = "highs-ipm"
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "load_threshold_robustness.csv"
    manifest_path = OUT / "run_manifest.json"

    sites = pd.read_csv(SAMPLE).sort_values("nh3_equivalent_service_tph").reset_index(drop=True)
    selected = (sites.iloc[0], sites.iloc[len(sites) // 2], sites.iloc[-1])
    anchors = [row.to_dict() for row in selected]
    anchor_labels = {str(anchors[0]["site_id"]): "low",
                     str(anchors[1]["site_id"]): "median",
                     str(anchors[2]["site_id"]): "high"}
    if args.anchor:
        anchors = [row for row in anchors if anchor_labels[row["site_id"]] == args.anchor]
    expected_rows = len(SCENARIOS) * len(YEARS) * len(anchors)
    code_paths = [Path(__file__).resolve(), Path(site_model.__file__).resolve(),
                  Path(site_model.model.__file__).resolve(), Path(site_model.weather.__file__).resolve(),
                  Path(__import__("compact_cavern_lp").__file__).resolve(),
                  Path(__import__("run_eu27_rfnbo_frontier_aux").__file__).resolve(),
                  Path(__import__("rfnbo_inventory_frontier").__file__).resolve()]
    code_hashes = {str(p): sha256(p) for p in code_paths}
    configuration = {
        "anchor_selection": "minimum, median-order, and maximum EHO-derived service scale among 15 clusters",
        "selected_site_ids": [row["site_id"] for row in anchors],
        "weather_years": list(YEARS),
        "scenarios": list(SCENARIOS),
        "tested_minimum_load_fraction_range": [MIN_LOAD, MAX_LOAD],
        "bracket_tolerance": TOL,
        "solver": {"method": "highs-ipm", "formulation": "compact", "threads": 1,
                   "objective": "feasibility only; all original matrix and bound residuals checked"},
        "fixed": {"reference_caverns": 1, "ramp_fraction_nameplate_per_hour": RAMP,
                  "boundary": "cyclic", "renewable_design_years": list(site_model.weather.TRAIN_YEARS),
                  "renewable_overbuild": 1.5, "wind_energy_share": 0.5},
    }
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(f"Pass --resume to continue {OUT}")
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("configuration") != configuration:
            raise ValueError("Saved threshold-robustness configuration differs")
        if prior.get("code_sha256") != code_hashes:
            raise ValueError("Saved threshold source hashes differ")
        for path, digest in prior.get("input_sha256", {}).items():
            if sha256(Path(path)) != digest:
                raise ValueError(f"Saved threshold input differs: {path}")
        rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    else:
        if args.resume:
            raise FileNotFoundError("Cannot resume; no manifest exists")
        rows = []
    completed = {(r["scenario"], r["site_id"], int(r["weather_year"])) for r in rows}

    case = next(c for c in site_model.model.CAVERN_CASES if c.case_id == CAVERN_ID)
    input_hashes: dict[str, str] = {str(SAMPLE): sha256(SAMPLE)}
    for site in anchors:
        site_id = str(site["site_id"])
        pv_path = site_model.RAW_ROOT / "pvgis" / f"{site_id}_pvgis_era5_2005_2023.json"
        wind_path = site_model.RAW_ROOT / "open_meteo" / f"{site_id}_wind_era5_2005_2023.json"
        input_hashes[str(pv_path)] = sha256(pv_path)
        input_hashes[str(wind_path)] = sha256(wind_path)
        pv, wind = site_model.pvgis_series(pv_path), site_model.wind_series(wind_path)
        tz = str(site["timezone"])
        capacities = site_model.model.design_capacities_with_aux(
            tz, pv, wind, site_model.model.AUX_MWH_PER_T_DEFAULT,
            renewable_overbuild=1.5, wind_energy_share=0.5,
        )
        cavern = site_model.model.CavernCase(
            case.case_id, case.working_gas_million_sm3, case.peak_withdrawal_million_sm3_day,
            case.reference_site_cavern_count,
            service_reference_nh3_tph=float(site["nh3_equivalent_service_tph"]),
        )
        for year in YEARS:
            _, generation, months, _, _ = site_model.weather.profile_for_year(
                year, tz, pv, wind, capacities["pv_capacity_mw"], capacities["wind_capacity_mw"]
            )
            for scenario in SCENARIOS:
                key = (scenario["scenario"], site_id, int(year))
                if key in completed:
                    continue
                chosen = next(c for c in site_model.model.CAVERN_CASES
                              if c.case_id == scenario.get("cavern_id", CAVERN_ID))
                scenario_cavern = site_model.model.CavernCase(
                    chosen.case_id, chosen.working_gas_million_sm3,
                    scenario.get("withdrawal_rate_million_sm3_day", chosen.peak_withdrawal_million_sm3_day),
                    chosen.reference_site_cavern_count,
                    service_reference_nh3_tph=float(site["nh3_equivalent_service_tph"]))
                scenario_caps = capacities
                scenario_generation = generation
                overbuild = float(scenario.get("renewable_overbuild", 1.5))
                if overbuild != 1.5:
                    scenario_caps = site_model.model.design_capacities_with_aux(
                        tz, pv, wind, site_model.model.AUX_MWH_PER_T_DEFAULT,
                        renewable_overbuild=overbuild, wind_energy_share=0.5)
                    _, scenario_generation, _, _, _ = site_model.weather.profile_for_year(
                        year, tz, pv, wind, scenario_caps["pv_capacity_mw"], scenario_caps["wind_capacity_mw"])
                ramp = float(scenario.get("ramp_fraction_nameplate_per_hour", RAMP))
                result = threshold(scenario_generation, months, scenario_caps, scenario_cavern,
                                   float(scenario["wtir"]), float(scenario["buffer_hours"]), ramp)
                row = {
                    "scenario": scenario["scenario"],
                    "site_id": site_id,
                    "country": site["country"],
                    "city_cluster": site["city_cluster"],
                    "scale_anchor": anchor_labels[site_id],
                    "nh3_equivalent_service_tph": float(site["nh3_equivalent_service_tph"]),
                    "weather_year": int(year),
                    "wtir": float(scenario["wtir"]),
                    "product_buffer_cap_hours": float(scenario["buffer_hours"]),
                    "cavern_id": scenario_cavern.case_id,
                    "withdrawal_rate_million_sm3_day": scenario_cavern.peak_withdrawal_million_sm3_day,
                    "ramp_fraction_nameplate_per_hour": ramp,
                    "renewable_overbuild": overbuild,
                    **result,
                    "interpretation": "Conditional one-reference-cavern scenario threshold; not a measured plant operating limit or geological feasibility result.",
                }
                pd.DataFrame([row]).to_csv(
                    result_path, mode="a" if result_path.exists() else "w",
                    header=not result_path.exists(), index=False, lineterminator="\n",
                )
                rows.append(row)
                completed.add(key)
                print(f"{scenario['scenario']} {site_id} {year}: {result['threshold_classification']} "
                      f"bracket=({result['minimum_load_fraction_lower_feasible']}, "
                      f"{result['minimum_load_fraction_upper_infeasible']})", flush=True)

                manifest = {
                    "status": "running", "configuration": configuration,
                    "completed_rows": len(rows), "expected_rows": expected_rows,
                    "input_sha256": input_hashes,
                    "source_code_sha256": sha256(Path(__file__).resolve()),
                    "code_sha256": code_hashes,
                    "output": str(result_path),
                    "interpretation_boundary": "Scenario screening only; no facility operation, geology, cavern availability, or population inference.",
                }
                atomic_manifest(manifest_path, manifest)
            del generation, months
        del pv, wind

    manifest = {
        "status": "complete" if len(completed) == expected_rows else "partial",
        "configuration": configuration,
        "completed_rows": len(completed), "expected_rows": expected_rows,
        "input_sha256": input_hashes,
        "source_code_sha256": sha256(Path(__file__).resolve()),
        "code_sha256": code_hashes,
        "output": str(result_path),
        "interpretation_boundary": "Scenario screening only; no facility operation, geology, cavern availability, or population inference.",
    }
    atomic_manifest(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

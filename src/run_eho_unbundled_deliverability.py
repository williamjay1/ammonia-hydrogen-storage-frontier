"""Counterfactual minimum deliverability at identical fixed working-gas volume.

This separates the rate requirement from the assumption that every reference
module bundles a fixed volume with one fixed directional rate. It does not
price extra wells/compressors or establish that their rates are achievable.
"""
from pathlib import Path
import argparse
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog, OptimizeWarning
import run_eho_site_module_frontier as site


def minimum_deliverability(generation, months, *, capacities, q_floor,
                          flexible, buffer_hours, fixed_volume, wtir=2.,
                          test_short=False):
    c, ub, b, eq, beq, bounds, idx = site.model.build_lp(
        generation, months, [len(generation)],
        electrolyser_mw=capacities["electrolyser_mw"],
        grid_connection_mw=capacities["grid_connection_mw"],
        min_hb_load_fraction=q_floor, flexible=flexible,
        nh3_buffer_limit_hours=buffer_hours, matching_rule="hourly",
        boundary="cyclic", hb_nameplate_tph=1.5,
        ramp_fraction_of_nameplate_per_hour=.6,
        withdrawal_to_injection_ratio=wtir,
        allow_short_periods_for_test=test_short)
    bounds[idx.h2_capacity] = (0., fixed_volume)
    n = idx.hours
    old_size = idx.size
    rows = np.r_[np.arange(n), np.arange(n), np.arange(n, 2*n), np.arange(n, 2*n)]
    cols = np.r_[np.arange(idx.starts["h2_charge_kgph"], idx.starts["h2_charge_kgph"]+n),
                 np.full(n, old_size),
                 np.arange(idx.starts["h2_discharge_kgph"], idx.starts["h2_discharge_kgph"]+n),
                 np.full(n, old_size)]
    vals = np.r_[np.full(n, wtir), -np.ones(n), np.ones(n), -np.ones(n)]
    rate_rows = sparse.coo_matrix((vals, (rows, cols)), shape=(2*n, old_size+1)).tocsr()
    ub = sparse.vstack([sparse.hstack([ub, sparse.csr_matrix((ub.shape[0], 1))]), rate_rows], format="csr")
    b = np.r_[b, np.zeros(2*n)]
    eq = sparse.hstack([eq, sparse.csr_matrix((eq.shape[0], 1))], format="csr")
    objective = np.zeros(old_size+1)
    objective[-1] = 1.
    bounds = [*bounds, (0., None)]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=OptimizeWarning)
        solution = linprog(objective, A_ub=ub, b_ub=b, A_eq=eq, b_eq=beq,
                           bounds=bounds, method="highs-ds", options={"threads": 1})
    if not solution.success:
        raise RuntimeError(f"Fixed-volume rate solve failed: {solution.message}")
    residuals = [np.max(np.abs(eq @ solution.x - beq)), max(0., np.max(ub @ solution.x - b))]
    for value, (lower, upper) in zip(solution.x, bounds):
        if lower is not None:
            residuals.append(max(0., lower-value))
        if upper is not None:
            residuals.append(max(0., value-upper))
    maximum_residual = float(max(residuals))
    if maximum_residual > 1e-5:
        raise RuntimeError(f"Rate solve residual {maximum_residual}")
    return {"minimum_withdrawal_kgph_per_normalized_tph": float(solution.fun),
            "minimum_injection_kgph_per_normalized_tph": float(solution.fun/wtir),
            "fixed_working_gas_kg_per_normalized_tph": fixed_volume,
            "maximum_original_matrix_and_bound_residual": maximum_residual,
            "status": "optimal"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "run_manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError("Requires complete primary results")
    frame = pd.read_csv(args.run_dir / "site_weather_frontier.csv")
    sample = pd.read_csv(site.SAMPLE_PATH).sort_values("nh3_equivalent_service_tph")
    anchors = [sample.iloc[0], sample.iloc[len(sample)//2], sample.iloc[-1]]
    reference = site.model.CAVERN_CASES[-1]
    rows = []
    for point in anchors:
        sid = point.site_id
        pv = site.pvgis_series(site.RAW_ROOT / "pvgis" / f"{sid}_pvgis_era5_2005_2023.json")
        wind = site.wind_series(site.RAW_ROOT / "open_meteo" / f"{sid}_wind_era5_2005_2023.json")
        scale = point.nh3_equivalent_service_tph
        for year in (2015, 2019, 2023):
            selected = frame[(frame.site_id == sid) & (frame.weather_year == year)]
            rigid = selected[selected.process_case == "rigid"].iloc[0]
            count = max(1, int(np.ceil(scale*rigid.free_h2_storage_kg_per_normalized_tph/reference.working_gas_kg-1e-9)))
            volume = count * reference.working_gas_kg / scale
            capacities = {"electrolyser_mw": rigid.electrolyser_capacity_mw_per_normalized_tph,
                          "grid_connection_mw": rigid.grid_connection_mw_per_normalized_tph}
            _, generation, months, _, _ = site.weather.profile_for_year(
                year, point.timezone, pv, wind, rigid.pv_capacity_mw_per_normalized_tph,
                rigid.wind_capacity_mw_per_normalized_tph)
            for case, flexible, floor, buffer in (("rigid", False, 1., 0.),
                                                  ("flex_ml0p6_b24", True, .6, 24.),
                                                  ("flex_ml0p4_b24", True, .4, 24.)):
                result = minimum_deliverability(generation, months, capacities=capacities,
                                               q_floor=floor, flexible=flexible, buffer_hours=buffer,
                                               fixed_volume=volume)
                rows.append({"site_id": sid, "weather_year": year, "process_case": case,
                             "fixed_reference_volume_modules": count,
                             "service_scale_tph": scale,
                             "minimum_withdrawal_kgph_at_scale": scale*result["minimum_withdrawal_kgph_per_normalized_tph"],
                             "minimum_withdrawal_reference_rate_equivalents": scale*result["minimum_withdrawal_kgph_per_normalized_tph"]/reference.peak_withdrawal_kg_per_hour,
                             **result})
                print(sid, year, case, result, flush=True)
                pd.DataFrame(rows).to_csv(args.run_dir / "fixed_volume_deliverability_counterfactual.csv", index=False)
    report = {"status": "complete", "rows": len(rows), "same_volume_across_process_cases": True,
              "input_csv_sha256": hashlib.sha256((args.run_dir / "site_weather_frontier.csv").read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "interpretation": "Minimum rate at fixed generic working-gas volume; no engineering availability or cost for extra wells/compressors is inferred."}
    (args.run_dir / "fixed_volume_deliverability_audit.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

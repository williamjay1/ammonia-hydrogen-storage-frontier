"""Independent necessary bounds for the EHO storage-module screen.

The interval bounds relax auxiliary electricity and process ramping. They are
necessary conditions, not sufficient schedules. They supplement the exact
no-flow-bounds volume lower bound with transparent service and rate cuts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_eho_site_module_frontier as site_model


DEFAULT_LENGTHS = (1, 6, 12, 24, 48, 72, 168, 336, 720, 1440, 2160, 4380)
H2_PER_T = site_model.model.H2_PER_NH3_KG_PER_T
H2_PER_MWH = site_model.model.H2_KG_PER_MWH_ELECTROLYSER


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integer_ceiling(value: float) -> int:
    if not np.isfinite(value) or value < -1e-9:
        raise ValueError("module ratio must be finite and nonnegative")
    return max(0, int(np.ceil(max(0.0, value) - 1e-9)))


def cyclic_minimum_sum(values: np.ndarray, length: int) -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("finite, nonempty hourly values are required")
    if not 1 <= length <= len(values):
        raise ValueError("interval length must lie within the annual horizon")
    extended = np.concatenate((values, values[: length - 1]))
    cumulative = np.concatenate(([0.0], np.cumsum(extended)))
    sums = cumulative[length:] - cumulative[:-length]
    return float(np.min(sums))


def service_rate_bounds(
    maximum_h2_production: np.ndarray,
    *,
    minimum_nh3_tph: float,
    product_buffer_hours: float,
    service_scale_tph: float,
    working_gas_kg: float,
    withdrawal_kgph: float,
    wtir: float,
    free_inventory_kg_per_tph: float,
    lengths: tuple[int, ...] = DEFAULT_LENGTHS,
) -> dict[str, float | int]:
    """Bounds implied by fixed hourly product service and cyclic stocks.

    For any L-hour interval, NH3 synthesis must be at least max(q_min L,
    L-B). Eligible H2 production cannot exceed the supplied hourly cap. Any
    positive difference must fit in working gas and cross the withdrawal
    interface. Cyclic inventory requires replenishment over the complement.
    """
    supply = np.asarray(maximum_h2_production, dtype=float)
    if (supply.ndim != 1 or supply.size < 2 or not np.isfinite(supply).all()
            or np.any(supply < 0) or not 0 <= minimum_nh3_tph <= 1
            or product_buffer_hours < 0 or service_scale_tph <= 0
            or working_gas_kg <= 0 or withdrawal_kgph <= 0 or wtir <= 0
            or free_inventory_kg_per_tph < 0):
        raise ValueError("invalid service, storage or supply bounds")
    n = len(supply)
    considered = sorted({int(x) for x in lengths if 1 <= x < n})
    if not considered:
        raise ValueError("at least one proper interval is required")
    capacity_ratio = rate_ratio = recharge_ratio = 0.0
    capacity_length = rate_length = recharge_length = 0
    service_only_one_module_violation = 0.0
    injection_kgph = withdrawal_kgph / wtir
    for length in considered:
        eligible_supply = cyclic_minimum_sum(supply, length)
        required_synthesis = max(minimum_nh3_tph * length,
                                 length - product_buffer_hours)
        deficit = max(0.0, H2_PER_T * required_synthesis - eligible_supply)
        ratios = (
            service_scale_tph * deficit / working_gas_kg,
            service_scale_tph * deficit / (withdrawal_kgph * length),
            service_scale_tph * deficit / (injection_kgph * (n - length)),
        )
        if ratios[0] > capacity_ratio:
            capacity_ratio, capacity_length = ratios[0], length
        if ratios[1] > rate_ratio:
            rate_ratio, rate_length = ratios[1], length
        if ratios[2] > recharge_ratio:
            recharge_ratio, recharge_length = ratios[2], length
        service_deficit = max(0.0, H2_PER_T * max(0, length - product_buffer_hours)
                              - eligible_supply)
        service_only_one_module_violation = max(
            service_only_one_module_violation,
            service_scale_tph * service_deficit / working_gas_kg - 1,
            service_scale_tph * service_deficit / (withdrawal_kgph * length) - 1,
            service_scale_tph * service_deficit / (injection_kgph * (n - length)) - 1,
        )
    exact_volume = integer_ceiling(
        service_scale_tph * free_inventory_kg_per_tph / working_gas_kg
    )
    rate_lower = integer_ceiling(rate_ratio)
    recharge_lower = integer_ceiling(recharge_ratio)
    cut_volume = integer_ceiling(capacity_ratio)
    return {
        "exact_volume_module_lower_bound": exact_volume,
        "interval_volume_module_lower_bound": cut_volume,
        "withdrawal_cut_module_lower_bound": rate_lower,
        "cyclic_recharge_module_lower_bound": recharge_lower,
        "combined_necessary_module_lower_bound": max(exact_volume, rate_lower,
                                                      recharge_lower, cut_volume),
        "certified_deliverability_above_volume_bound": int(
            max(rate_lower, recharge_lower) > exact_volume
        ),
        "withdrawal_cut_continuous_ratio": rate_ratio,
        "withdrawal_cut_interval_hours": rate_length,
        "recharge_cut_continuous_ratio": recharge_ratio,
        "recharge_cut_interval_hours": recharge_length,
        "volume_cut_continuous_ratio": capacity_ratio,
        "volume_cut_interval_hours": capacity_length,
        "one_module_service_cut_violation_ratio": max(0.0,
                                                     service_only_one_module_violation),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.run_dir / "run_manifest.json"
    csv_path = args.run_dir / "site_weather_frontier.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("A complete merged run is required")
    frame = pd.read_csv(csv_path)
    reconstruction_import = (
        site_model.model.AUX_MWH_PER_T_DEFAULT
        + site_model.model.STORAGE_COMPRESSION_KWH_PER_KG / 1000.
        * H2_PER_MWH * frame['electrolyser_capacity_mw_per_normalized_tph'])
    if not (frame['grid_connection_mw_per_normalized_tph'] >= reconstruction_import).all():
        raise RuntimeError('Rigid combined-stock reconstruction not guaranteed by grid cap')
    rigid = frame[frame.process_case == 'rigid'].set_index(['site_id','weather_year'])
    transfer_rows = []
    for row in frame[frame.process_case != 'rigid'].to_dict('records'):
        baseline = rigid.loc[(row['site_id'],row['weather_year'])]
        relief = (baseline.free_h2_storage_kg_per_normalized_tph
                  - row['free_h2_storage_kg_per_normalized_tph']) / H2_PER_T
        buffer = float(row['nh3_buffer_limit_hours'])
        if relief < -1e-5 or relief > buffer + 1e-5:
            raise RuntimeError(f'Combined-stock relief bound violated: {row["site_id"]}/{row["weather_year"]}')
        transfer_rows.append({'site_id':row['site_id'],'weather_year':row['weather_year'],
                              'process_case':row['process_case'],
                              'free_inventory_relief_service_hours':relief,
                              'product_buffer_limit_service_hours':buffer})
    pd.DataFrame(transfer_rows).to_csv(args.run_dir/'inventory_transfer_bound_checks.csv',index=False)
    sites = pd.read_csv(site_model.SAMPLE_PATH).set_index("site_id")
    output = []
    for site_id, site_rows in frame.groupby("site_id", sort=True):
        site = sites.loc[site_id]
        pv_path = site_model.RAW_ROOT / "pvgis" / f"{site_id}_pvgis_era5_2005_2023.json"
        wind_path = site_model.RAW_ROOT / "open_meteo" / f"{site_id}_wind_era5_2005_2023.json"
        pv, wind = site_model.pvgis_series(pv_path), site_model.wind_series(wind_path)
        for year, year_rows in site_rows.groupby("weather_year", sort=True):
            first = year_rows.iloc[0]
            _, generation, _, _, _ = site_model.weather.profile_for_year(
                int(year), str(site["timezone"]), pv, wind,
                float(first["pv_capacity_mw_per_normalized_tph"]),
                float(first["wind_capacity_mw_per_normalized_tph"]),
            )
            h2_max = H2_PER_MWH * np.minimum(
                generation, float(first["electrolyser_capacity_mw_per_normalized_tph"])
            )
            for row in year_rows.to_dict("records"):
                if row["free_storage_status"] != "optimal":
                    raise RuntimeError("Missing free-storage solution")
                cavern = next(c for c in site_model.model.CAVERN_CASES
                              if c.case_id == row["cavern_archetype"])
                flexible = str(row["flexible_process"]).lower() == "true"
                q_min = (float(row["min_hb_load_fraction_of_nameplate"])
                         * float(row["hb_nameplate_multiple"])) if flexible else 1.0
                values = service_rate_bounds(
                    h2_max, minimum_nh3_tph=q_min,
                    product_buffer_hours=float(row["nh3_buffer_limit_hours"]),
                    service_scale_tph=float(row["service_reference_nh3_tph"]),
                    working_gas_kg=cavern.working_gas_kg,
                    withdrawal_kgph=cavern.peak_withdrawal_kg_per_hour,
                    wtir=float(row["withdrawal_to_injection_ratio"]),
                    free_inventory_kg_per_tph=float(row["free_h2_storage_kg_per_normalized_tph"]),
                )
                if (row["status"] == "optimal"
                        and values["combined_necessary_module_lower_bound"]
                        > int(row["minimum_feasible_modules"])):
                    raise RuntimeError(f"Necessary bound exceeds optimized count: {site_id}/{year}")
                output.append({"site_id": site_id, "weather_year": int(year),
                               "process_case": row["process_case"],
                               "optimized_reference_modules": row["minimum_feasible_modules"],
                               **values})
    result = pd.DataFrame(output)
    result.to_csv(args.run_dir / "capacity_deliverability_bounds.csv", index=False)
    summary = {"rows": len(result), "all_necessary_bounds_satisfied": True,
               'combined_stock_bound_pairs':len(transfer_rows),
               'all_combined_stock_relief_bounds_satisfied':True,
               'grid_reconstruction_sufficient_condition_satisfied':True,
               'maximum_free_inventory_relief_service_hours':max(r['free_inventory_relief_service_hours'] for r in transfer_rows),
               "interval_lengths_hours": list(DEFAULT_LENGTHS),
               "by_process_case": {},
               "source_sha256": sha256(Path(__file__)),
               "input_sha256": {str(csv_path): sha256(csv_path),
                                  str(manifest_path): sha256(manifest_path)},
               "interpretation": "Necessary bounds over stated interval lengths; eligibility supply is an optimistic cap. No sufficiency, geology or facility reliability claim."}
    for case, rows in result.groupby("process_case"):
        summary["by_process_case"][case] = {
            "rows": len(rows),
            "certified_rate_bound_above_volume_bound": int(
                rows["certified_deliverability_above_volume_bound"].sum()),
            "combined_bound_equals_optimized_count": int(
                (rows["combined_necessary_module_lower_bound"]
                 == rows["optimized_reference_modules"]).sum()),
        }
    (args.run_dir / "capacity_deliverability_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

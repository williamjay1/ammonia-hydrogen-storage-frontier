"""Validate and summarize the frozen public EHO cavern-frontier run.

The bootstrap is descriptive for the selected city-cluster sample; it is not
an EU-27 population confidence interval. This script refuses partial runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    ROOT
    / "results"
    / "eho_site_module_frontier"
    / "primary_cyclic_v5_ds1_ml06_ml04_b24_holdout2015_2023"
)
SAMPLE = ROOT / "datasets" / "eho_ammonia_site_clusters_v1_20260930.csv"
YEARS = tuple(range(2015, 2024))
RIGID_CASE = "rigid"


def percentile_interval(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or values.size == 0:
        return [float("nan"), float("nan")]
    n = len(values)
    draws = np.empty(10_000, dtype=float)
    for b in range(len(draws)):
        draws[b] = np.median(values[rng.integers(0, n, size=n)])
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def pooled_cluster_interval(frame: pd.DataFrame, metric: str,
                            rng: np.random.Generator) -> list[float]:
    """Resample whole locations and recalculate the pooled location-year median."""
    grouped = [group[metric].to_numpy(dtype=float)
               for _, group in frame.groupby("site_id", sort=True)]
    if not grouped or any(not np.isfinite(group).all() for group in grouped):
        raise ValueError("Cluster bootstrap requires finite complete paired values")
    draws = np.empty(10_000)
    for b in range(len(draws)):
        selected = rng.integers(0, len(grouped), size=len(grouped))
        draws[b] = np.median(np.concatenate([grouped[j] for j in selected]))
    return np.quantile(draws, [0.025, 0.975]).tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--flex-case", default="flex_ml0p6_b24")
    parser.add_argument("--output-suffix", default=None)
    args = parser.parse_args()

    manifest = json.loads((args.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    all_rows = pd.read_csv(args.run_dir / "site_weather_frontier.csv")
    cases = (RIGID_CASE, args.flex_case)
    frame = all_rows.loc[all_rows["process_case"].astype(str).isin(cases)].copy()
    sites = pd.read_csv(SAMPLE)
    expected_n = len(sites) * len(YEARS) * len(cases)
    keys = ["site_id", "weather_year", "process_case"]
    required_keys = {
        (site, year, case)
        for site in sites["site_id"].astype(str)
        for year in YEARS
        for case in cases
    }
    observed_keys = set(
        zip(frame["site_id"].astype(str), frame["weather_year"].astype(int), frame["process_case"].astype(str))
    )
    if manifest.get("status") != "complete" or len(frame) != expected_n:
        raise RuntimeError(f"Primary run is incomplete: manifest={manifest.get('status')}, rows={len(frame)}/{expected_n}")
    if frame.duplicated(keys).any() or observed_keys != required_keys:
        raise RuntimeError("Primary scenario keys are duplicated or missing")
    if set(frame["status"].astype(str)) - {"optimal", "not_feasible_within_reference_module_limit"}:
        raise RuntimeError("Unexpected solver status in primary run")
    if not frame["free_storage_status"].astype(str).eq("optimal").all():
        raise RuntimeError("At least one unconstrained-storage benchmark is not optimal")

    residual_cols = [
        "max_h2_balance_residual_kgph",
        "max_h2_storage_balance_residual_kg",
        "max_nh3_balance_residual_tph",
        "max_annual_nh3_service_residual_t_per_normalized_tph",
        "max_hb_ramp_violation_tph_per_hour",
        "max_site_power_violation_mw",
        "max_hourly_matching_violation_mw",
    ]
    optimal = frame[frame["status"].eq("optimal")]
    residual_max = {c: float(optimal[c].abs().max()) for c in residual_cols}
    if optimal[residual_cols].isna().any().any() or not np.isfinite(optimal[residual_cols].to_numpy()).all():
        raise RuntimeError("Missing or nonfinite structural residuals")
    if any(v > 1e-5 for v in residual_max.values()):
        raise RuntimeError(f"Structural residual exceeds 1e-5: {residual_max}")

    if args.flex_case == RIGID_CASE or args.flex_case not in set(frame["process_case"].astype(str)):
        raise RuntimeError(f"Flexible comparison case is missing or invalid: {args.flex_case}")
    rigid = frame[frame["process_case"].eq(RIGID_CASE)].set_index(["site_id", "weather_year"])
    flex = frame[frame["process_case"].eq(args.flex_case)].set_index(["site_id", "weather_year"])
    paired = rigid.add_suffix("_rigid").join(flex.add_suffix("_flex"), how="inner").reset_index()
    paired["free_h2_reduction_pct"] = 100 * (
        paired["free_h2_storage_service_hours_rigid"]
        - paired["free_h2_storage_service_hours_flex"]
    ) / paired["free_h2_storage_service_hours_rigid"]
    paired["reference_caverns_avoided"] = (
        paired["minimum_feasible_modules_rigid"] - paired["minimum_feasible_modules_flex"]
    )
    paired["rigid_free_inventory_within_one_module_volume"] = (
        paired["free_h2_storage_service_hours_rigid"]
        <= paired["module_reference_service_hours_at_site_scale_rigid"] + 1e-6
    )
    paired["flex_free_inventory_within_one_module_volume"] = (
        paired["free_h2_storage_service_hours_flex"]
        <= paired["module_reference_service_hours_at_site_scale_flex"] + 1e-6
    )
    paired["module_pair_feasible"] = (
        paired["status_rigid"].eq("optimal") & paired["status_flex"].eq("optimal")
    )
    for prefix in ("rigid", "flex"):
        ratio = (paired[f"free_h2_storage_service_hours_{prefix}"]
                 / paired[f"module_reference_service_hours_at_site_scale_{prefix}"])
        paired[f"volume_only_module_lower_bound_{prefix}"] = np.ceil(ratio - 1e-9).astype("Int64")
        paired[f"flow_or_schedule_extra_modules_{prefix}"] = (
            paired[f"minimum_feasible_modules_{prefix}"]
            - paired[f"volume_only_module_lower_bound_{prefix}"]
        ).astype("Int64")
        if (paired[f"flow_or_schedule_extra_modules_{prefix}"].dropna() < 0).any():
            raise RuntimeError("Feasible module count falls below the free-inventory volume lower bound")
    paired.loc[~paired["module_pair_feasible"], "reference_caverns_avoided"] = np.nan
    finite_reductions = paired["free_h2_reduction_pct"].to_numpy(dtype=float)
    if np.nanmin(finite_reductions) < -1e-4:
        raise RuntimeError("Flexible case has higher unconstrained minimum H2 inventory than rigid case")
    feasible_pairs = paired.loc[paired["module_pair_feasible"]]
    if (feasible_pairs["reference_caverns_avoided"] < 0).any():
        raise RuntimeError(
            "Flexible feasible set contains the rigid schedule, so it cannot require more reference modules"
        )

    site_summary = paired.groupby("site_id", as_index=False).agg(
        country=("country_rigid", "first"),
        weather_years=("weather_year", "nunique"),
        median_free_h2_reduction_pct=("free_h2_reduction_pct", "median"),
        median_reference_caverns_avoided=("reference_caverns_avoided", "median"),
        weather_years_with_lower_count=("reference_caverns_avoided", lambda x: int((x > 0).sum())),
        feasible_module_pairs=("module_pair_feasible", "sum"),
        all_replay_module_count_rigid=("minimum_feasible_modules_rigid", "max"),
        all_replay_module_count_flex=("minimum_feasible_modules_flex", "max"),
        maximum_annual_free_inventory_hours_rigid=("free_h2_storage_service_hours_rigid", "max"),
        maximum_annual_free_inventory_hours_flex=("free_h2_storage_service_hours_flex", "max"),
    )
    site_summary['all_replay_module_count_reduction']=(
        site_summary.all_replay_module_count_rigid-site_summary.all_replay_module_count_flex)
    site_summary['maximum_annual_free_inventory_reduction_pct']=100*(
        site_summary.maximum_annual_free_inventory_hours_rigid
        -site_summary.maximum_annual_free_inventory_hours_flex)/site_summary.maximum_annual_free_inventory_hours_rigid
    incomplete_sites=site_summary.feasible_module_pairs!=len(YEARS)
    site_summary.loc[incomplete_sites,['all_replay_module_count_rigid','all_replay_module_count_flex',
                                     'all_replay_module_count_reduction']]=np.nan
    year_summary = paired.groupby("weather_year", as_index=False).agg(
        matched_locations=("site_id", "nunique"),
        median_free_h2_reduction_pct=("free_h2_reduction_pct", "median"),
        median_reference_caverns_avoided=("reference_caverns_avoided", "median"),
        location_years_with_lower_count=("reference_caverns_avoided", lambda x: int((x > 0).sum())),
    )

    module = paired.loc[paired["module_pair_feasible"], "reference_caverns_avoided"].to_numpy(dtype=float)
    rng = np.random.default_rng(20260930)
    cluster_medians = site_summary["median_free_h2_reduction_pct"].to_numpy(dtype=float)
    summary = {
        "status": "validated_complete",
        "run_label": manifest.get("run_label"),
        "flexible_case": args.flex_case,
        "rows_expected": expected_n,
        "rows_observed": int(len(frame)),
        "rows_in_combined_run": int(len(all_rows)),
        "locations": int(sites["site_id"].nunique()),
        "countries": sorted(sites["country"].unique().tolist()),
        "evaluation_years": list(YEARS),
        "optimal_rows": int(frame["status"].eq("optimal").sum()),
        "not_feasible_within_16_reference_caverns_rows": int(frame["status"].eq("not_feasible_within_reference_module_limit").sum()),
        "maximum_absolute_structural_residuals": residual_max,
        "free_h2_reduction_pct_location_year_median": float(paired["free_h2_reduction_pct"].median()),
        "free_h2_reduction_pct_location_year_iqr": [
            float(paired["free_h2_reduction_pct"].quantile(0.25)),
            float(paired["free_h2_reduction_pct"].quantile(0.75)),
        ],
        "free_h2_reduction_pct_location_year_range": [float(paired["free_h2_reduction_pct"].min()), float(paired["free_h2_reduction_pct"].max())],
        "location_year_pairs_with_lower_reference_cavern_count": int((module > 0).sum()),
        "location_year_pairs_with_equal_reference_cavern_count": int((module == 0).sum()),
        "location_year_pairs_with_higher_reference_cavern_count": int((module < 0).sum()),
        "feasible_pairs_crossing_to_fewer_modules_under_flexibility": int((module > 0).sum()),
        "feasible_pairs_crossing_from_multiple_modules_to_one": int(
            ((paired["module_pair_feasible"])
             & (paired["minimum_feasible_modules_rigid"] > 1)
             & (paired["minimum_feasible_modules_flex"] == 1)).sum()
        ),
        "rigid_cases_where_one_module_volume_covers_free_inventory_but_multiple_modules_are_required": int(
            ((paired["minimum_feasible_modules_rigid"] > 1)
             & paired["rigid_free_inventory_within_one_module_volume"]).sum()
        ),
        "flexible_cases_where_one_module_volume_covers_free_inventory_but_multiple_modules_are_required": int(
            ((paired["minimum_feasible_modules_flex"] > 1)
             & paired["flex_free_inventory_within_one_module_volume"]).sum()
        ),
        "feasible_paired_cavern_solutions": int(len(module)),
        "median_reference_caverns_avoided_among_feasible_pairs": float(np.median(module)) if len(module) else None,
        'sites_with_complete_all_replay_module_adequacy':int((~incomplete_sites).sum()),
        'sites_with_lower_all_replay_module_count':int((site_summary.all_replay_module_count_reduction>0).sum()),
        'sites_with_equal_all_replay_module_count':int((site_summary.all_replay_module_count_reduction==0).sum()),
        'median_all_replay_module_count_reduction':float(site_summary.all_replay_module_count_reduction.median()),
        'median_maximum_annual_free_inventory_reduction_pct':float(site_summary.maximum_annual_free_inventory_reduction_pct.median()),
        'all_replay_design_interpretation':'Maximum annual requirement covers all nine independently cyclic perfect-foresight replays; not a contiguous multiyear trajectory or a common initial inventory.',
        "median_volume_only_module_lower_bound_rigid": float(paired["volume_only_module_lower_bound_rigid"].median()),
        "median_volume_only_module_lower_bound_flexible": float(paired["volume_only_module_lower_bound_flex"].median()),
        "pairs_where_rigid_flow_or_schedule_requires_extra_module_over_volume_bound": int((paired.loc[paired["module_pair_feasible"], "flow_or_schedule_extra_modules_rigid"] > 0).sum()),
        "pairs_where_flexible_flow_or_schedule_requires_extra_module_over_volume_bound": int((paired.loc[paired["module_pair_feasible"], "flow_or_schedule_extra_modules_flex"] > 0).sum()),
        "median_extra_modules_over_volume_only_bound_rigid": float(paired.loc[paired["module_pair_feasible"], "flow_or_schedule_extra_modules_rigid"].median()),
        "median_extra_modules_over_volume_only_bound_flexible": float(paired.loc[paired["module_pair_feasible"], "flow_or_schedule_extra_modules_flex"].median()),
        "cluster_bootstrap_95pct_interval_for_median_site_median_free_inventory_reduction_pct": percentile_interval(cluster_medians, rng),
        "cluster_bootstrap_95pct_interval_for_pooled_location_year_median_free_inventory_reduction_pct": pooled_cluster_interval(paired, "free_h2_reduction_pct", rng),
        "bootstrap_interpretation": "Descriptive cluster-resampling interval over the selected 15 EHO city clusters; not an EU-27 population confidence interval.",
        "boundary": "Perfect-foresight independent annual historical replay; cyclic inventories; no plant or cavern availability data.",
    }
    suffix = args.output_suffix or args.flex_case
    paired.to_csv(args.run_dir / f"paired_site_weather_summary_{suffix}.csv", index=False)
    site_summary.to_csv(args.run_dir / f"summary_by_site_{suffix}.csv", index=False)
    year_summary.to_csv(args.run_dir / f"summary_by_weather_year_{suffix}.csv", index=False)
    (args.run_dir / f"validated_summary_{suffix}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

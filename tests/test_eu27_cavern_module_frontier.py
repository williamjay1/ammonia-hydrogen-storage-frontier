import numpy as np
import pandas as pd

from src.run_eu27_cavern_module_frontier import (
    CAVERN_CASES,
    H2_DENSITY_KG_PER_SM3,
    H2_PER_NH3_KG_PER_T,
    STORAGE_COMPRESSION_KWH_PER_KG,
    CavernCase,
    build_lp,
    process_ramp_edges,
    solve_shared_horizon,
    storage_capex_screen,
)
from src import run_eu27_rfnbo_frontier as weather_model


def _solve(generation, *, matching="hourly", boundary="cyclic", flexible=False,
           buffer_hours=0.0, cavern=None, cavern_count=None):
    generation = np.asarray(generation, dtype=float)
    months = np.array(["2015-01"] * len(generation), dtype=str)
    return solve_shared_horizon(
        generation, months, [len(generation)],
        electrolyser_mw=27.7,
        grid_connection_mw=30.0,
        min_hb_load_fraction=0.40,
        flexible=flexible,
        nh3_buffer_limit_hours=buffer_hours,
        matching_rule=matching,
        boundary=boundary,
        cavern=cavern,
        cavern_count=cavern_count,
        allow_short_periods_for_test=True,
    )


def test_hystories_reference_conversion_uses_15c_standard_volume():
    assert abs(H2_DENSITY_KG_PER_SM3 - 0.0852567) < 1e-6
    low, mid, high = CAVERN_CASES
    assert abs(low.reference_service_hours - 525.8) < 1.0
    assert abs(mid.reference_service_hours - 263.3) < 1.0
    assert abs(high.reference_service_hours - 131.2) < 1.0
    assert abs(high.reference_peak_rate_ratio - 0.4907) < 0.002
    assert abs(STORAGE_COMPRESSION_KWH_PER_KG - 0.5869) < 0.001


def test_cavern_module_normalization_can_match_public_site_service_scale():
    base = CAVERN_CASES[-1]
    site_tph = 21.511358797449617
    site = CavernCase(
        base.case_id,
        base.working_gas_million_sm3,
        base.peak_withdrawal_million_sm3_day,
        base.reference_site_cavern_count,
        service_reference_nh3_tph=site_tph,
    )
    assert abs(
        site.working_gas_kg_per_normalized_plant
        - site.working_gas_kg / site_tph
    ) < 1e-8
    assert abs(
        site.reference_service_hours
        - site.working_gas_kg / (H2_PER_NH3_KG_PER_T * site_tph)
    ) < 1e-8
    assert site.reference_service_hours > base.reference_service_hours


def test_constant_renewable_supply_needs_no_inventory_under_zero_boundaries():
    out = _solve(np.full(48, 20.0), boundary="zero")
    assert out["status"] == "optimal", out
    assert np.isfinite(out["grid_import_mwh"])
    assert abs(out["grid_import_mwh"] - out["grid_import_mwh_per_1_tph_nh3"]) < 1e-9
    assert out["h2_storage_capacity_kg_per_1_tph_nh3"] < 1e-6
    assert out["nh3_buffer_capacity_t_per_1_tph_nh3"] < 1e-6
    assert out["max_h2_balance_residual_kgph"] < 1e-6
    assert out["max_h2_storage_balance_residual_kg"] < 1e-6
    assert out["max_annual_nh3_service_residual_t"] < 1e-6


def test_monthly_versus_hourly_matching_changes_storage_need_in_synthetic_case():
    generation = np.r_[np.zeros(24), np.full(24, 30.0)]
    monthly = _solve(generation, matching="monthly")
    hourly = _solve(generation, matching="hourly")
    assert monthly["status"] == hourly["status"] == "optimal"
    assert monthly["h2_storage_capacity_kg_per_1_tph_nh3"] < 1e-5
    assert hourly["h2_storage_capacity_kg_per_1_tph_nh3"] > 1000.0
    assert monthly["max_matching_violation_native"] < 1e-6
    assert hourly["max_matching_violation_native"] < 1e-6


def test_published_small_cavern_peak_rate_can_bind_even_if_capacity_is_large():
    generation = np.r_[np.zeros(24), np.full(24, 30.0)]
    out = _solve(
        generation,
        matching="hourly",
        flexible=False,
        cavern=CAVERN_CASES[-1],
        cavern_count=1,
    )
    assert out["status"] == "infeasible"


def test_flexible_process_and_product_buffer_can_relax_peak_withdrawal_requirement():
    generation = np.r_[np.zeros(24), np.full(24, 30.0)]
    out = _solve(
        generation,
        matching="hourly",
        flexible=True,
        buffer_hours=24.0,
        cavern=CAVERN_CASES[0],
        cavern_count=1,
    )
    assert out["status"] == "optimal", out
    assert out["max_h2_discharge_kgph"] <= out["cavern_peak_withdrawal_limit_kgph_per_1_tph_nh3"] + 1e-5
    assert out["nh3_buffer_capacity_t_per_1_tph_nh3"] <= 24.0 + 1e-5


def test_hystories_wtir_applies_distinct_injection_and_withdrawal_limits():
    cavern = CAVERN_CASES[1]
    months = np.array(["2015-01"] * 48, dtype=str)
    args = dict(
        electrolyser_mw=27.7,
        grid_connection_mw=30.0,
        min_hb_load_fraction=0.4,
        flexible=True,
        nh3_buffer_limit_hours=24.0,
        matching_rule="hourly",
        cavern=cavern,
        cavern_count=1,
        withdrawal_to_injection_ratio=2.0,
        allow_short_periods_for_test=True,
    )
    *_, bounds, idx = build_lp(
        np.r_[np.zeros(24), np.full(24, 30.0)], months, [48], **args
    )
    charge_limit = bounds[idx.pos("h2_charge_kgph", 0)][1]
    withdrawal_limit = bounds[idx.pos("h2_discharge_kgph", 0)][1]
    expected_withdrawal = cavern.peak_withdrawal_kg_per_hour_per_normalized_plant
    assert abs(withdrawal_limit - expected_withdrawal) < 1e-8
    assert abs(charge_limit - expected_withdrawal / 2.0) < 1e-8


def test_sensitivity_parameters_are_applied_and_reported():
    generation = np.full(48, 30.0)
    months = np.array(["2015-01"] * len(generation), dtype=str)
    out = solve_shared_horizon(
        generation, months, [len(generation)],
        electrolyser_mw=27.7,
        grid_connection_mw=30.0,
        min_hb_load_fraction=0.4,
        flexible=False,
        nh3_buffer_limit_hours=0.0,
        matching_rule="hourly",
        boundary="zero",
        auxiliary_mwh_per_t_nh3=0.50,
        storage_loss_per_hour=0.001,
        compression_kwh_per_kg=0.8,
        withdrawal_to_injection_ratio=3.0,
        allow_short_periods_for_test=True,
    )
    assert out["status"] == "optimal", out
    assert out["auxiliary_mwh_per_t_nh3"] == 0.50
    assert out["storage_loss_per_hour"] == 0.001
    assert out["compression_kwh_per_kg_h2"] == 0.8
    assert out["withdrawal_to_injection_ratio"] == 3.0
    assert out["max_h2_storage_balance_residual_kg"] < 1e-6


def test_process_nameplate_and_ramp_sensitivity_are_applied_and_audited():
    generation = np.full(48, 30.0)
    months = np.array(["2015-01"] * len(generation), dtype=str)
    out = solve_shared_horizon(
        generation, months, [len(generation)],
        electrolyser_mw=27.7,
        grid_connection_mw=30.0,
        min_hb_load_fraction=0.60,
        flexible=True,
        nh3_buffer_limit_hours=24.0,
        matching_rule="hourly",
        boundary="zero",
        hb_nameplate_tph=1.25,
        ramp_fraction_of_nameplate_per_hour=0.30,
        allow_short_periods_for_test=True,
    )
    assert out["status"] == "optimal", out
    assert out["hb_nameplate_tph"] == 1.25
    assert out["ramp_fraction_of_nameplate_per_hour"] == 0.30
    assert out["max_hb_ramp_violation_tph_per_hour"] < 1e-6


def test_cyclic_single_year_closes_process_ramp_but_multiyear_horizon_does_not_wrap():
    single_year_edges = process_ramp_edges([48], "cyclic")
    zero_boundary_edges = process_ramp_edges([48], "zero")
    continuous_multiyear_edges = process_ramp_edges([24, 24], "cyclic")
    assert (47, 0) in single_year_edges
    assert (47, 0) not in zero_boundary_edges
    assert (23, 24) in continuous_multiyear_edges
    assert (47, 0) not in continuous_multiyear_edges

    common = dict(
        electrolyser_mw=27.7,
        grid_connection_mw=30.0,
        min_hb_load_fraction=0.40,
        flexible=True,
        nh3_buffer_limit_hours=24.0,
        matching_rule="hourly",
        allow_short_periods_for_test=True,
    )
    generation = np.full(48, 30.0)
    months = np.array(["2015-01"] * len(generation), dtype=str)
    cyclic_ub = build_lp(generation, months, [48], boundary="cyclic", **common)[1]
    zero_ub = build_lp(generation, months, [48], boundary="zero", **common)[1]
    assert cyclic_ub.shape[0] - zero_ub.shape[0] == 2


def test_storage_capex_screen_is_dimensionally_scaled_and_linear_in_modules():
    cavern = CAVERN_CASES[1]
    one = storage_capex_screen(cavern, 1, 24.0)
    two = storage_capex_screen(cavern, 2, 24.0)
    assert abs(two["installed_cavern_working_gas_t_h2_500kt_y"]
               - 2 * one["installed_cavern_working_gas_t_h2_500kt_y"]) < 1e-9
    assert abs(two["h2_subsurface_capex_rate_equiv_eur_mid"]
               - 2 * one["h2_subsurface_capex_rate_equiv_eur_mid"]) < 1e-6
    assert abs(two["h2_surface_capex_rate_equiv_eur_205_per_kw"]
               - 2 * one["h2_surface_capex_rate_equiv_eur_205_per_kw"]) < 1e-6
    assert one["nh3_tank_reference_capex_eur_888_per_t"] == two["nh3_tank_reference_capex_eur_888_per_t"]
    assert abs(one["storage_capex_rate_equiv_mid_plus_nh3_tank_reference_eur"]
               - (one["h2_subsurface_capex_rate_equiv_eur_mid"]
                  + one["h2_surface_capex_rate_equiv_eur_205_per_kw"]
                  + one["nh3_tank_reference_capex_eur_888_per_t"])) < 1e-6


def test_renewable_design_sensitivity_scales_capacity_without_changing_resource_profile():
    index = pd.date_range("2006-01-01", "2014-12-31 23:00", freq="h", tz="UTC")
    pv = pd.Series(np.full(len(index), 0.25), index=index)
    wind = pd.Series(np.full(len(index), 12.0), index=index)
    base = weather_model.design_capacities("UTC", pv, wind)
    higher_overbuild = weather_model.design_capacities(
        "UTC", pv, wind, renewable_overbuild=2.0, wind_energy_share=0.5
    )
    wind_heavy = weather_model.design_capacities(
        "UTC", pv, wind, renewable_overbuild=1.5, wind_energy_share=0.7
    )
    assert abs(higher_overbuild["pv_capacity_mw"] / base["pv_capacity_mw"] - 2.0 / 1.5) < 1e-9
    assert abs(higher_overbuild["wind_capacity_mw"] / base["wind_capacity_mw"] - 2.0 / 1.5) < 1e-9
    assert wind_heavy["wind_capacity_mw"] > base["wind_capacity_mw"]
    assert wind_heavy["pv_capacity_mw"] < base["pv_capacity_mw"]

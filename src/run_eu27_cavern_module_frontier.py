"""Shared-horizon P2A inventory screen against public cavern capacity/rate bands.

This is a conditional engineering screen, not plant reliability or a site
selection model. The standardized process is normalized to 1 t NH3/h; cavern
limits are scaled consistently from a 500 kt/y reference plant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_eu27_rfnbo_frontier as weather  # noqa: E402
from run_eu27_rfnbo_frontier_aux import (  # noqa: E402
    AUX_MWH_PER_T_DEFAULT,
    design_capacities_with_aux,
)
from rfnbo_inventory_frontier import H2_PER_NH3_KG_PER_T  # noqa: E402

OUT_ROOT = ROOT / "results" / "eu27_cavern_module_frontier"
R = 8.31446261815324
P_STANDARD_PA = 101325.0
T_STANDARD_K = 273.15 + 15.0
H2_MOLAR_MASS_KG_PER_MOL = 0.00201588
H2_DENSITY_KG_PER_SM3 = H2_MOLAR_MASS_KG_PER_MOL * P_STANDARD_PA / (R * T_STANDARD_K)
H2_LHV_KWH_PER_KG = 33.33
NH3_REFERENCE_TONNES_PER_YEAR = 500_000.0
REFERENCE_NH3_TPH = NH3_REFERENCE_TONNES_PER_YEAR / 8760.0
PLANT_SCALE_FROM_1_TPH = REFERENCE_NH3_TPH / weather.DEMAND_TPH
ELECTROLYSER_SEC_KWH_PER_KG = weather.SEC_KWH_PER_KG
H2_KG_PER_MWH_ELECTROLYSER = 1000.0 / ELECTROLYSER_SEC_KWH_PER_KG
# Hystories D7.1, Table 14: installed brake power divided by peak cavern
# injection flow gives 0.585-0.587 kWh/kg at the report's 15 C standard state.
# Use a linearized 0.587 kWh/kg screening value; no site compressor map exists.
STORAGE_COMPRESSION_KWH_PER_KG = 0.587
STORAGE_LOSS_PER_HOUR = 0.0
SOLVER_METHOD = "highs-ds"
MAX_MODULES_DEFAULT = 16
HYSTORIES_SUBSURFACE_CAPEX_EUR_PER_KWH = {"low": 0.44, "mid": 0.51, "high": 0.69}
HYSTORIES_SURFACE_CAPEX_EUR_PER_KW_WITHDRAWAL = 205.0
NH3_TANK_CAPEX_EUR_PER_T = 888.0


@dataclass(frozen=True)
class CavernCase:
    case_id: str
    working_gas_million_sm3: float
    peak_withdrawal_million_sm3_day: float
    reference_site_cavern_count: int
    service_reference_nh3_tph: float = REFERENCE_NH3_TPH

    def __post_init__(self) -> None:
        if not math.isfinite(self.service_reference_nh3_tph) or self.service_reference_nh3_tph <= 0:
            raise ValueError("service reference throughput must be finite and positive")

    @property
    def working_gas_kg(self) -> float:
        return self.working_gas_million_sm3 * 1e6 * H2_DENSITY_KG_PER_SM3

    @property
    def peak_withdrawal_kg_per_hour(self) -> float:
        return self.peak_withdrawal_million_sm3_day * 1e6 * H2_DENSITY_KG_PER_SM3 / 24.0

    @property
    def working_gas_kg_per_normalized_plant(self) -> float:
        return self.working_gas_kg / self.service_reference_nh3_tph

    @property
    def peak_withdrawal_kg_per_hour_per_normalized_plant(self) -> float:
        return self.peak_withdrawal_kg_per_hour / self.service_reference_nh3_tph

    @property
    def reference_service_hours(self) -> float:
        return self.working_gas_kg / (
            H2_PER_NH3_KG_PER_T * self.service_reference_nh3_tph
        )

    @property
    def reference_peak_rate_ratio(self) -> float:
        reference_kg_per_day = (
            H2_PER_NH3_KG_PER_T * self.service_reference_nh3_tph * 24.0
        )
        return self.peak_withdrawal_kg_per_hour * 24.0 / reference_kg_per_day


def storage_capex_screen(cavern: CavernCase, module_count: int,
                         nh3_buffer_t_per_1_tph: float) -> dict[str, float]:
    """Return rate-equivalent CAPEX sensitivities at the 500 kt/y scale.

    Hystories D7.2 rates describe a conceptual 250-million-Sm3 reference site,
    not a project quote. Applying those rates to conceptual module equivalents
    is a linear screening extrapolation that omits shared-site costs. The NH3
    tank rate is terminal-scale and is not a fitted small-tank cost curve.
    """
    if module_count < 0 or nh3_buffer_t_per_1_tph < 0:
        raise ValueError("module count and NH3 buffer must be nonnegative")
    installed_working_gas_kg = module_count * cavern.working_gas_kg
    installed_withdrawal_kgph = module_count * cavern.peak_withdrawal_kg_per_hour
    h2_capacity_kwh = installed_working_gas_kg * H2_LHV_KWH_PER_KG
    h2_withdrawal_kw = installed_withdrawal_kgph * H2_LHV_KWH_PER_KG
    subsurface = {
        level: rate * h2_capacity_kwh
        for level, rate in HYSTORIES_SUBSURFACE_CAPEX_EUR_PER_KWH.items()
    }
    surface = HYSTORIES_SURFACE_CAPEX_EUR_PER_KW_WITHDRAWAL * h2_withdrawal_kw
    nh3_buffer_t = nh3_buffer_t_per_1_tph * cavern.service_reference_nh3_tph
    nh3_tank = NH3_TANK_CAPEX_EUR_PER_T * nh3_buffer_t
    return {
        "installed_cavern_working_gas_t_h2_500kt_y": installed_working_gas_kg / 1000.0,
        "installed_cavern_max_withdrawal_tph_h2_500kt_y": installed_withdrawal_kgph / 1000.0,
        "h2_subsurface_capex_rate_equiv_eur_low": subsurface["low"],
        "h2_subsurface_capex_rate_equiv_eur_mid": subsurface["mid"],
        "h2_subsurface_capex_rate_equiv_eur_high": subsurface["high"],
        "h2_surface_capex_rate_equiv_eur_205_per_kw": surface,
        "nh3_buffer_capacity_t_500kt_y": nh3_buffer_t,
        "nh3_tank_reference_capex_eur_888_per_t": nh3_tank,
        "storage_capex_rate_equiv_mid_plus_nh3_tank_reference_eur": (
            subsurface["mid"] + surface + nh3_tank
        ),
    }


CAVERN_CASES = (
    CavernCase("low_investment_large_cavern", 62.5, 5.9, 4),
    CavernCase("mid_investment_cavern", 31.3, 2.8, 8),
    CavernCase("high_investment_small_cavern", 15.6, 1.4, 16),
)


class Index:
    names = ("electrolyser_mw", "nh3_tph", "h2_stock_kg", "nh3_stock_t",
             "h2_charge_kgph", "h2_discharge_kgph", "grid_import_mw")

    def __init__(self, hours: int) -> None:
        self.hours = hours
        self.starts: dict[str, int] = {}
        cursor = 0
        for name in self.names:
            self.starts[name] = cursor
            cursor += hours
        self.h2_capacity = cursor
        self.nh3_capacity = cursor + 1
        self.size = cursor + 2

    def pos(self, name: str, hour: int) -> int:
        return self.starts[name] + hour


def process_ramp_edges(period_lengths: list[int], boundary: str) -> list[tuple[int, int]]:
    """Return (previous, current) hourly ramp edges for the chosen horizon.

    A single cyclic calendar-year replay closes the process ramp at its year
    boundary. A multi-year shared horizon remains chronological and is not
    wrapped from its final hour to its first.
    """
    if not period_lengths or any(length < 1 for length in period_lengths):
        raise ValueError("period lengths must be positive")
    total = sum(period_lengths)
    edges = [(hour - 1, hour) for hour in range(1, total)]
    if boundary == "cyclic" and len(period_lengths) == 1 and total > 1:
        edges.append((total - 1, 0))
    return edges


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_lp(
    generation_mw: np.ndarray,
    local_months: np.ndarray,
    period_lengths: list[int],
    *,
    electrolyser_mw: float,
    grid_connection_mw: float,
    min_hb_load_fraction: float,
    flexible: bool,
    nh3_buffer_limit_hours: float,
    matching_rule: str,
    boundary: str = "cyclic",
    auxiliary_mwh_per_t_nh3: float = AUX_MWH_PER_T_DEFAULT,
    storage_loss_per_hour: float = STORAGE_LOSS_PER_HOUR,
    compression_kwh_per_kg: float = STORAGE_COMPRESSION_KWH_PER_KG,
    withdrawal_to_injection_ratio: float = 2.0,
    cavern: CavernCase | None = None,
    cavern_count: int | None = None,
    demand_tph: float = weather.DEMAND_TPH,
    hb_nameplate_tph: float = weather.HB_NAMEPLATE_TPH,
    ramp_fraction_of_nameplate_per_hour: float = weather.HB_RAMP_FRACTION_NAMEPLATE_PER_HOUR,
    allow_short_periods_for_test: bool = False,
) -> tuple[np.ndarray, sparse.csr_matrix, np.ndarray, sparse.csr_matrix, np.ndarray,
           list[tuple[float | None, float | None]], Index]:
    """Build a physically dimensioned hourly LP over one continuous horizon."""
    g = np.asarray(generation_mw, dtype=float)
    months = np.asarray(local_months, dtype=str)
    if g.ndim != 1 or g.size == 0 or months.shape != g.shape:
        raise ValueError("generation and local-month arrays must be nonempty and aligned")
    if not np.isfinite(g).all() or np.any(g < 0):
        raise ValueError("renewable generation must be finite and nonnegative")
    if matching_rule not in {"monthly", "hourly"}:
        raise ValueError("matching_rule must be monthly or hourly")
    if boundary not in {"cyclic", "zero"}:
        raise ValueError("boundary must be cyclic or zero")
    if not period_lengths or sum(period_lengths) != g.size:
        raise ValueError("period lengths must cover the horizon exactly")
    if not allow_short_periods_for_test and any(n not in (8760, 8784) for n in period_lengths):
        raise ValueError("each calendar-year period must contain 8,760 or 8,784 hours")
    if cavern_count is not None and (cavern is None or cavern_count < 0):
        raise ValueError("a nonnegative cavern count requires a cavern archetype")
    if not 0 <= min_hb_load_fraction <= 1 or nh3_buffer_limit_hours < 0:
        raise ValueError("invalid process or product-buffer scenario")
    if (not 0 <= storage_loss_per_hour < 1 or compression_kwh_per_kg < 0
            or not 1 <= withdrawal_to_injection_ratio <= 5):
        raise ValueError("invalid storage loss, compression, or WTIR parameter")

    n = int(g.size)
    idx = Index(n)
    q_min = (
        hb_nameplate_tph * min_hb_load_fraction
        if flexible else demand_tph
    )
    q_max = hb_nameplate_tph if flexible else demand_tph
    if q_min > demand_tph or q_max < demand_tph:
        raise ValueError("process envelope cannot deliver the declared average service")
    buffer_cap_max_t = nh3_buffer_limit_hours * demand_tph

    bounds: list[tuple[float | None, float | None]] = [(0.0, None)] * idx.size
    for t in range(n):
        bounds[idx.pos("electrolyser_mw", t)] = (0.0, electrolyser_mw)
        bounds[idx.pos("nh3_tph", t)] = (q_min, q_max)
        bounds[idx.pos("h2_stock_kg", t)] = (0.0, None)
        bounds[idx.pos("nh3_stock_t", t)] = (0.0, None)
        bounds[idx.pos("h2_charge_kgph", t)] = (0.0, None)
        bounds[idx.pos("h2_discharge_kgph", t)] = (0.0, None)
        bounds[idx.pos("grid_import_mw", t)] = (0.0, grid_connection_mw)
    bounds[idx.h2_capacity] = (0.0, None)
    bounds[idx.nh3_capacity] = (0.0, buffer_cap_max_t)

    if cavern is not None and cavern_count is not None:
        cap = cavern_count * cavern.working_gas_kg_per_normalized_plant
        withdrawal_flow = cavern_count * cavern.peak_withdrawal_kg_per_hour_per_normalized_plant
        injection_flow = withdrawal_flow / withdrawal_to_injection_ratio
        bounds[idx.h2_capacity] = (0.0, cap)
        for t in range(n):
            bounds[idx.pos("h2_charge_kgph", t)] = (0.0, injection_flow)
            bounds[idx.pos("h2_discharge_kgph", t)] = (0.0, withdrawal_flow)
        if cavern_count == 0:
            bounds[idx.h2_capacity] = (0.0, 0.0)
    if boundary == "zero":
        # Stocks are zero immediately before the first hour and after the last.
        bounds[idx.pos("h2_stock_kg", n - 1)] = (0.0, 0.0)
        bounds[idx.pos("nh3_stock_t", n - 1)] = (0.0, 0.0)

    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_vals: list[float] = []
    for t in range(n):
        prev = (n - 1 if boundary == "cyclic" else None) if t == 0 else t - 1
        # H2 stock: h_t = (1-loss)h_(t-1) + charge - discharge.
        row = t
        eq_rows.extend((row, row, row))
        eq_cols.extend((idx.pos("h2_stock_kg", t), idx.pos("h2_charge_kgph", t),
                        idx.pos("h2_discharge_kgph", t)))
        eq_vals.extend((1.0, -1.0, 1.0))
        if prev is not None:
            eq_rows.append(row)
            eq_cols.append(idx.pos("h2_stock_kg", prev))
            eq_vals.append(-(1.0 - storage_loss_per_hour))

        # NH3 product inventory carries continuously and closes only at horizon end.
        row = n + t
        eq_rows.extend((row, row))
        eq_cols.extend((idx.pos("nh3_stock_t", t), idx.pos("nh3_tph", t)))
        eq_vals.extend((1.0, -1.0))
        if prev is not None:
            eq_rows.append(row)
            eq_cols.append(idx.pos("nh3_stock_t", prev))
            eq_vals.append(-1.0)

        # Electrolytic H2 plus withdrawal minus injection must feed the HB loop.
        row = 2 * n + t
        eq_rows.extend((row, row, row, row))
        eq_cols.extend((idx.pos("electrolyser_mw", t), idx.pos("h2_discharge_kgph", t),
                        idx.pos("h2_charge_kgph", t), idx.pos("nh3_tph", t)))
        eq_vals.extend((H2_KG_PER_MWH_ELECTROLYSER, 1.0, -1.0,
                        -H2_PER_NH3_KG_PER_T))

    annual_start = 3 * n
    cursor = 0
    annual_targets: list[float] = []
    for year_no, length in enumerate(period_lengths):
        row = annual_start + year_no
        for t in range(cursor, cursor + length):
            eq_rows.append(row)
            eq_cols.append(idx.pos("nh3_tph", t))
            eq_vals.append(1.0)
        annual_targets.append(demand_tph * length)
        cursor += length
    a_eq = sparse.coo_matrix(
        (eq_vals, (eq_rows, eq_cols)), shape=(3 * n + len(period_lengths), idx.size)
    ).tocsr()
    b_eq = np.concatenate((
        np.zeros(n),
        np.full(n, -demand_tph, dtype=float),
        np.zeros(n),
        np.asarray(annual_targets, dtype=float),
    ))

    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_vals: list[float] = []
    b_ub: list[float] = []
    row = 0
    for t in range(n):
        # Shared working-gas and product-buffer capacities.
        ub_rows.extend((row, row))
        ub_cols.extend((idx.pos("h2_stock_kg", t), idx.h2_capacity))
        ub_vals.extend((1.0, -1.0))
        b_ub.append(0.0)
        row += 1
        ub_rows.extend((row, row))
        ub_cols.extend((idx.pos("nh3_stock_t", t), idx.nh3_capacity))
        ub_vals.extend((1.0, -1.0))
        b_ub.append(0.0)
        row += 1

    # Hourly process ramp is continuous across chronological year joins. For
    # an independently replayed cyclic calendar year, also close the annual
    # schedule from its final hour back to its first.
    if flexible:
        ramp = ramp_fraction_of_nameplate_per_hour * hb_nameplate_tph
        for prev, t in process_ramp_edges(period_lengths, boundary):
            ub_rows.extend((row, row))
            ub_cols.extend((idx.pos("nh3_tph", t), idx.pos("nh3_tph", prev)))
            ub_vals.extend((1.0, -1.0))
            b_ub.append(ramp)
            row += 1
            ub_rows.extend((row, row))
            ub_cols.extend((idx.pos("nh3_tph", prev), idx.pos("nh3_tph", t)))
            ub_vals.extend((1.0, -1.0))
            b_ub.append(ramp)
            row += 1

    # Full physical site-power balance. Grid imports may serve site load, but
    # are not counted as renewable eligibility for electrolyser input.
    compression_mwh_per_kg = compression_kwh_per_kg / 1000.0
    for t in range(n):
        ub_rows.extend((row, row, row, row))
        ub_cols.extend((idx.pos("electrolyser_mw", t), idx.pos("nh3_tph", t),
                        idx.pos("h2_charge_kgph", t), idx.pos("grid_import_mw", t)))
        ub_vals.extend((1.0, auxiliary_mwh_per_t_nh3, compression_mwh_per_kg, -1.0))
        b_ub.append(float(g[t]))
        row += 1

    if matching_rule == "hourly":
        for t in range(n):
            ub_rows.append(row)
            ub_cols.append(idx.pos("electrolyser_mw", t))
            ub_vals.append(1.0)
            b_ub.append(float(g[t]))
            row += 1
    else:
        for month in dict.fromkeys(months.tolist()):
            ixm = np.flatnonzero(months == month)
            for t in ixm:
                ub_rows.append(row)
                ub_cols.append(idx.pos("electrolyser_mw", int(t)))
                ub_vals.append(1.0)
            b_ub.append(float(g[ixm].sum()))
            row += 1

    a_ub = sparse.coo_matrix((ub_vals, (ub_rows, ub_cols)), shape=(row, idx.size)).tocsr()
    c = np.zeros(idx.size, dtype=float)
    c[idx.h2_capacity] = 1.0
    return c, a_ub, np.asarray(b_ub), a_eq, b_eq, bounds, idx


def solve_shared_horizon(
    generation_mw: np.ndarray,
    local_months: np.ndarray,
    period_lengths: list[int],
    *,
    electrolyser_mw: float,
    grid_connection_mw: float,
    min_hb_load_fraction: float,
    flexible: bool,
    nh3_buffer_limit_hours: float,
    matching_rule: str,
    boundary: str = "cyclic",
    auxiliary_mwh_per_t_nh3: float = AUX_MWH_PER_T_DEFAULT,
    storage_loss_per_hour: float = STORAGE_LOSS_PER_HOUR,
    compression_kwh_per_kg: float = STORAGE_COMPRESSION_KWH_PER_KG,
    withdrawal_to_injection_ratio: float = 2.0,
    hb_nameplate_tph: float = weather.HB_NAMEPLATE_TPH,
    ramp_fraction_of_nameplate_per_hour: float = weather.HB_RAMP_FRACTION_NAMEPLATE_PER_HOUR,
    cavern: CavernCase | None = None,
    cavern_count: int | None = None,
    lexicographic: bool = True,
    allow_short_periods_for_test: bool = False,
) -> dict[str, Any]:
    c1, a_ub, b_ub, a_eq, b_eq, bounds, idx = build_lp(
        generation_mw, local_months, period_lengths,
        electrolyser_mw=electrolyser_mw,
        grid_connection_mw=grid_connection_mw,
        min_hb_load_fraction=min_hb_load_fraction,
        flexible=flexible,
        nh3_buffer_limit_hours=nh3_buffer_limit_hours,
        matching_rule=matching_rule,
        boundary=boundary,
        auxiliary_mwh_per_t_nh3=auxiliary_mwh_per_t_nh3,
        storage_loss_per_hour=storage_loss_per_hour,
        compression_kwh_per_kg=compression_kwh_per_kg,
        withdrawal_to_injection_ratio=withdrawal_to_injection_ratio,
        hb_nameplate_tph=hb_nameplate_tph,
        ramp_fraction_of_nameplate_per_hour=ramp_fraction_of_nameplate_per_hour,
        cavern=cavern,
        cavern_count=cavern_count,
        allow_short_periods_for_test=allow_short_periods_for_test,
    )
    result1 = linprog(c1, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                      bounds=bounds, method=SOLVER_METHOD)
    rec: dict[str, Any] = {
        "status": "optimal" if result1.success else ("infeasible" if result1.status == 2 else "solver_failed"),
        "solver_status": int(result1.status),
        "solver_message": str(result1.message),
        "solver_method": SOLVER_METHOD,
        "matching_rule": matching_rule,
        "boundary_condition": boundary,
        "flexible_process": flexible,
        "min_hb_load_fraction_of_nameplate": min_hb_load_fraction if flexible else 1.0,
        "hb_nameplate_tph": hb_nameplate_tph,
        "ramp_fraction_of_nameplate_per_hour": ramp_fraction_of_nameplate_per_hour,
        "nh3_buffer_limit_hours": nh3_buffer_limit_hours,
        "auxiliary_mwh_per_t_nh3": auxiliary_mwh_per_t_nh3,
        "storage_loss_per_hour": storage_loss_per_hour,
        "compression_kwh_per_kg_h2": compression_kwh_per_kg,
        "withdrawal_to_injection_ratio": withdrawal_to_injection_ratio,
        "cavern_archetype": cavern.case_id if cavern else "unconstrained",
        "cavern_count_limit": cavern_count,
        "h2_storage_capacity_kg_per_1_tph_nh3": math.nan,
        "h2_storage_capacity_reference_service_hours": math.nan,
        "nh3_buffer_capacity_t_per_1_tph_nh3": math.nan,
        "grid_import_mwh": math.nan,
        "max_h2_balance_residual_kgph": math.nan,
        "max_h2_storage_balance_residual_kg": math.nan,
        "max_nh3_balance_residual_tph": math.nan,
        "max_annual_nh3_service_residual_t": math.nan,
        "max_hb_ramp_violation_tph_per_hour": math.nan,
        "max_site_power_violation_mw": math.nan,
        "max_matching_violation_native": math.nan,
        "max_h2_charge_kgph": math.nan,
        "max_h2_discharge_kgph": math.nan,
    }
    if not result1.success:
        return rec

    solutions = result1.x
    h_value = float(solutions[idx.h2_capacity])
    b_value = float(solutions[idx.nh3_capacity])
    if lexicographic:
        bounds2 = list(bounds)
        bounds2[idx.h2_capacity] = (h_value, h_value)
        c2 = np.zeros(idx.size, dtype=float)
        c2[idx.nh3_capacity] = 1.0
        result2 = linprog(c2, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                          bounds=bounds2, method=SOLVER_METHOD)
        if not result2.success:
            rec.update(status="solver_failed", solver_status=int(result2.status),
                       solver_message=f"second lexicographic stage failed: {result2.message}")
            return rec
        b_value = float(result2.x[idx.nh3_capacity])
        bounds3 = list(bounds2)
        bounds3[idx.nh3_capacity] = (b_value, b_value)
        c3 = np.zeros(idx.size, dtype=float)
        c3[idx.starts["grid_import_mw"]:idx.starts["grid_import_mw"] + idx.hours] = 1.0
        result3 = linprog(c3, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                          bounds=bounds3, method=SOLVER_METHOD)
        if not result3.success:
            rec.update(status="solver_failed", solver_status=int(result3.status),
                       solver_message=f"third lexicographic stage failed: {result3.message}")
            return rec
        solutions = result3.x

    u = solutions[idx.starts["electrolyser_mw"]:idx.starts["electrolyser_mw"] + idx.hours]
    q = solutions[idx.starts["nh3_tph"]:idx.starts["nh3_tph"] + idx.hours]
    h = solutions[idx.starts["h2_stock_kg"]:idx.starts["h2_stock_kg"] + idx.hours]
    a = solutions[idx.starts["nh3_stock_t"]:idx.starts["nh3_stock_t"] + idx.hours]
    charge = solutions[idx.starts["h2_charge_kgph"]:idx.starts["h2_charge_kgph"] + idx.hours]
    discharge = solutions[idx.starts["h2_discharge_kgph"]:idx.starts["h2_discharge_kgph"] + idx.hours]
    imports = solutions[idx.starts["grid_import_mw"]:idx.starts["grid_import_mw"] + idx.hours]
    if boundary == "cyclic":
        prev_h, prev_a = np.roll(h, 1), np.roll(a, 1)
    else:
        prev_h = np.concatenate(([0.0], h[:-1]))
        prev_a = np.concatenate(([0.0], a[:-1]))
    h2_balance = H2_KG_PER_MWH_ELECTROLYSER * u + discharge - charge - H2_PER_NH3_KG_PER_T * q
    h2_stock_balance = h - (1.0 - storage_loss_per_hour) * prev_h - charge + discharge
    nh3_balance = a - prev_a - q + weather.DEMAND_TPH
    annual_residuals: list[float] = []
    cursor = 0
    for length in period_lengths:
        annual_residuals.append(float(q[cursor:cursor + length].sum() - weather.DEMAND_TPH * length))
        cursor += length
    power_violation = (
        u + auxiliary_mwh_per_t_nh3 * q
        + (compression_kwh_per_kg / 1000.0) * charge
        - generation_mw - imports
    )
    if matching_rule == "hourly":
        matching_violation = u - generation_mw
    else:
        month_violation = [
            max(0.0, float(u[local_months == month].sum() - generation_mw[local_months == month].sum()))
            for month in dict.fromkeys(local_months.tolist())
        ]
        matching_violation = np.asarray(month_violation, dtype=float)
    ramp_violation = 0.0
    if flexible and q.size > 1:
        edge_changes = np.asarray([q[t] - q[prev]
                                   for prev, t in process_ramp_edges(period_lengths, boundary)])
        ramp_violation = max(0.0, float(np.max(
            np.abs(edge_changes) - ramp_fraction_of_nameplate_per_hour * hb_nameplate_tph
        )))
    rec.update({
        "status": "optimal",
        "h2_storage_capacity_kg_per_1_tph_nh3": float(solutions[idx.h2_capacity]),
        "h2_storage_capacity_reference_service_hours": float(
            solutions[idx.h2_capacity] / (H2_PER_NH3_KG_PER_T * weather.DEMAND_TPH)
        ),
        "h2_storage_capacity_t_for_500kt_y_reference": float(
            solutions[idx.h2_capacity] * PLANT_SCALE_FROM_1_TPH / 1000.0
        ),
        "nh3_buffer_capacity_t_per_1_tph_nh3": float(solutions[idx.nh3_capacity]),
        "nh3_buffer_capacity_t_for_500kt_y_reference": float(
            solutions[idx.nh3_capacity] * PLANT_SCALE_FROM_1_TPH
        ),
        "grid_import_mwh": float(imports.sum()),
        "grid_import_mwh_per_1_tph_nh3": float(imports.sum()),
        "grid_import_gwh_for_500kt_y_reference": float(
            imports.sum() * PLANT_SCALE_FROM_1_TPH / 1000.0
        ),
        "max_h2_balance_residual_kgph": float(np.max(np.abs(h2_balance))),
        "max_h2_storage_balance_residual_kg": float(np.max(np.abs(h2_stock_balance))),
        "max_nh3_balance_residual_tph": float(np.max(np.abs(nh3_balance))),
        "max_annual_nh3_service_residual_t": max(abs(x) for x in annual_residuals),
        "max_hb_ramp_violation_tph_per_hour": ramp_violation,
        "max_site_power_violation_mw": float(max(0.0, np.max(power_violation))),
        "max_matching_violation_native": float(max(0.0, np.max(matching_violation))) if matching_violation.size else 0.0,
        "max_h2_charge_kgph": float(np.max(charge)),
        "max_h2_discharge_kgph": float(np.max(discharge)),
        "peak_nh3_buffer_use_t_per_1_tph_nh3": float(np.max(a)),
        "n_hours": idx.hours,
    })
    if cavern is not None and cavern_count is not None:
        rec["h2_capacity_module_limit_kg_per_1_tph_nh3"] = (
            cavern_count * cavern.working_gas_kg_per_normalized_plant
        )
        rec["cavern_peak_withdrawal_limit_kgph_per_1_tph_nh3"] = (
            cavern_count * cavern.peak_withdrawal_kg_per_hour_per_normalized_plant
        )
        rec["cavern_peak_injection_limit_kgph_per_1_tph_nh3"] = (
            cavern_count * cavern.peak_withdrawal_kg_per_hour_per_normalized_plant
            / withdrawal_to_injection_ratio
        )
    return rec


def build_country_horizon(country: str, years: list[int]) -> tuple[
    dict[str, Any], dict[str, float], np.ndarray, np.ndarray, list[int], dict[str, Path]
]:
    if country not in weather.COUNTRY_CODES:
        raise ValueError(f"not an EU-27 code: {country}")
    if not years or any(b != a + 1 for a, b in zip(years, years[1:])):
        raise ValueError("years must be nonempty and consecutive")
    if not set(years).issubset(weather.EVALUATION_YEARS):
        raise ValueError(f"years must be within {weather.EVALUATION_YEARS}")
    point = weather.read_country_points()[country]
    pv_path = weather.RAW / "pv" / f"{country.lower()}_pvgis_era5_2005_2023.json"
    wind_path = weather.RAW / "wind" / f"{country.lower()}_openmeteo_era5_wind100m_2005_2023.json"
    pv = weather.pvgis_series(country)
    wind_speed = weather.wind_speed_series(country)
    caps = design_capacities_with_aux(
        str(point["timezone"]), pv, wind_speed, AUX_MWH_PER_T_DEFAULT
    )
    generation_parts: list[np.ndarray] = []
    month_parts: list[np.ndarray] = []
    lengths: list[int] = []
    for year in years:
        _, generation, month_labels, _, _ = weather.profile_for_year(
            year, str(point["timezone"]), pv, wind_speed,
            caps["pv_capacity_mw"], caps["wind_capacity_mw"],
        )
        generation_parts.append(generation)
        month_parts.append(month_labels)
        lengths.append(len(generation))
    return (
        point,
        caps,
        np.concatenate(generation_parts),
        np.concatenate(month_parts),
        lengths,
        {"pv": pv_path, "wind": wind_path, "points": weather.POINTS},
    )


def minimum_feasible_modules(
    generation: np.ndarray,
    months: np.ndarray,
    lengths: list[int],
    *,
    cavern: CavernCase,
    max_modules: int,
    electrolyser_mw: float,
    grid_connection_mw: float,
    min_load_fraction: float,
    flexible: bool,
    buffer_hours: float,
    matching_rule: str,
    country: str,
    boundary: str,
    auxiliary_mwh_per_t_nh3: float = AUX_MWH_PER_T_DEFAULT,
    storage_loss_per_hour: float = STORAGE_LOSS_PER_HOUR,
    compression_kwh_per_kg: float = STORAGE_COMPRESSION_KWH_PER_KG,
    withdrawal_to_injection_ratio: float = 2.0,
    hb_nameplate_tph: float = weather.HB_NAMEPLATE_TPH,
    ramp_fraction_of_nameplate_per_hour: float = weather.HB_RAMP_FRACTION_NAMEPLATE_PER_HOUR,
) -> tuple[int | None, list[dict[str, Any]], dict[str, Any] | None]:
    """Find the smallest feasible conceptual cavern count by monotone search."""
    cache: dict[int, dict[str, Any]] = {}

    def run_count(count: int, lex: bool = False) -> dict[str, Any]:
        if count not in cache or (lex and not cache[count].get("lexicographic")):
            value = solve_shared_horizon(
                generation, months, lengths,
                electrolyser_mw=electrolyser_mw,
                grid_connection_mw=grid_connection_mw,
                min_hb_load_fraction=min_load_fraction,
                flexible=flexible,
                nh3_buffer_limit_hours=buffer_hours,
                matching_rule=matching_rule,
                boundary=boundary,
                auxiliary_mwh_per_t_nh3=auxiliary_mwh_per_t_nh3,
                storage_loss_per_hour=storage_loss_per_hour,
                compression_kwh_per_kg=compression_kwh_per_kg,
                withdrawal_to_injection_ratio=withdrawal_to_injection_ratio,
                hb_nameplate_tph=hb_nameplate_tph,
                ramp_fraction_of_nameplate_per_hour=ramp_fraction_of_nameplate_per_hour,
                cavern=cavern,
                cavern_count=count,
                lexicographic=lex,
            )
            value["lexicographic"] = lex
            cache[count] = value
            print(f"{country} {matching_rule} flex={flexible} B={buffer_hours:g} "
                  f"{cavern.case_id} n={count}: {value['status']}", flush=True)
        return cache[count]

    zero = run_count(0)
    if zero["status"] == "optimal":
        final = run_count(0, lex=True)
        return 0, [dict(n_modules=n, **v) for n, v in sorted(cache.items())], final
    if zero["status"] != "infeasible":
        return None, [dict(n_modules=n, **v) for n, v in sorted(cache.items())], None

    low, high = 0, 1
    while high <= max_modules:
        result = run_count(high)
        if result["status"] == "optimal":
            break
        if result["status"] != "infeasible":
            return None, [dict(n_modules=n, **v) for n, v in sorted(cache.items())], None
        low = high
        high *= 2
    if high > max_modules:
        high = max_modules
        result = run_count(high)
        if result["status"] != "optimal":
            return None, [dict(n_modules=n, **v) for n, v in sorted(cache.items())], None

    while high - low > 1:
        mid = (low + high) // 2
        result = run_count(mid)
        if result["status"] == "optimal":
            high = mid
        elif result["status"] == "infeasible":
            low = mid
        else:
            return None, [dict(n_modules=n, **v) for n, v in sorted(cache.items())], None
    final = run_count(high, lex=True)
    return high, [dict(n_modules=n, **v) for n, v in sorted(cache.items())], final


def run(args: argparse.Namespace) -> Path:
    years = list(args.years)
    point, caps, generation, months, period_lengths, inputs = build_country_horizon(args.country, years)
    rows: list[dict[str, Any]] = []
    feasibility_rows: list[dict[str, Any]] = []
    process_specs = {
        "rigid": {"flexible": False, "min_load_fraction": 1.0},
        "flex": {"flexible": True, "min_load_fraction": args.min_load_fraction},
    }
    for boundary in args.boundaries:
        for rule in args.rules:
            for process_name in args.process_cases:
                process = process_specs[process_name]
                for buffer_hours in args.buffers:
                    base_result = solve_shared_horizon(
                        generation, months, period_lengths,
                        electrolyser_mw=caps["electrolyser_mw"],
                        grid_connection_mw=caps["grid_connection_mw"],
                        min_hb_load_fraction=process["min_load_fraction"],
                        flexible=process["flexible"],
                        nh3_buffer_limit_hours=buffer_hours,
                        matching_rule=rule,
                        boundary=boundary,
                        auxiliary_mwh_per_t_nh3=args.auxiliary_mwh_per_t_nh3,
                        storage_loss_per_hour=args.storage_loss_per_hour,
                        compression_kwh_per_kg=args.compression_kwh_per_kg,
                        withdrawal_to_injection_ratio=args.withdrawal_to_injection_ratio,
                        hb_nameplate_tph=args.hb_nameplate_multiple * weather.DEMAND_TPH,
                        ramp_fraction_of_nameplate_per_hour=args.ramp_fraction_nameplate_per_hour,
                    )
                    rows.append({
                        "country_code": args.country,
                        "country": point["country"],
                        "years": ";".join(map(str, years)),
                        "process_case": process_name,
                        "matching_rule": rule,
                        "buffer_limit_hours": buffer_hours,
                        "result_type": "unconstrained_h2_frontier",
                        "boundary_condition": boundary,
                        **base_result,
                    })
                    print(f"{args.country} {rule} {process_name} B={buffer_hours:g} "
                          f"unconstrained: {base_result['status']}", flush=True)
                    if base_result["status"] != "optimal" or args.max_modules <= 0:
                        continue
                    for cavern in CAVERN_CASES:
                        min_n, diagnostics, final = minimum_feasible_modules(
                            generation, months, period_lengths,
                            cavern=cavern,
                            max_modules=args.max_modules,
                            electrolyser_mw=caps["electrolyser_mw"],
                            grid_connection_mw=caps["grid_connection_mw"],
                            min_load_fraction=process["min_load_fraction"],
                            flexible=process["flexible"],
                            buffer_hours=buffer_hours,
                            matching_rule=rule,
                            country=args.country,
                            boundary=boundary,
                            auxiliary_mwh_per_t_nh3=args.auxiliary_mwh_per_t_nh3,
                            storage_loss_per_hour=args.storage_loss_per_hour,
                            compression_kwh_per_kg=args.compression_kwh_per_kg,
                            withdrawal_to_injection_ratio=args.withdrawal_to_injection_ratio,
                            hb_nameplate_tph=args.hb_nameplate_multiple * weather.DEMAND_TPH,
                            ramp_fraction_of_nameplate_per_hour=args.ramp_fraction_nameplate_per_hour,
                        )
                        feasibility_rows.extend({
                            "country_code": args.country,
                            "country": point["country"],
                            "years": ";".join(map(str, years)),
                            "process_case": process_name,
                            "matching_rule": rule,
                            "buffer_limit_hours": buffer_hours,
                            "cavern_archetype": cavern.case_id,
                            **item,
                        } for item in diagnostics)
                        rows.append({
                            "country_code": args.country,
                            "country": point["country"],
                            "years": ";".join(map(str, years)),
                            "process_case": process_name,
                            "matching_rule": rule,
                            "buffer_limit_hours": buffer_hours,
                            "result_type": "minimum_conceptual_cavern_count",
                            "boundary_condition": boundary,
                            "cavern_archetype": cavern.case_id,
                            "minimum_cavern_count": min_n,
                            "reference_site_cavern_count": cavern.reference_site_cavern_count,
                            "reference_module_service_hours": cavern.reference_service_hours,
                            "reference_module_peak_rate_ratio": cavern.reference_peak_rate_ratio,
                            **(final or {"status": "infeasible_or_solver_failure"}),
                            **(storage_capex_screen(
                                cavern, min_n,
                                float(final.get("nh3_buffer_capacity_t_per_1_tph_nh3", 0.0)),
                            ) if min_n is not None and final is not None else {}),
                        })

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / f"{stamp}_{args.run_label}"
    out_dir.mkdir(parents=True, exist_ok=False)
    result_path = out_dir / "frontier_results.csv"
    pd.DataFrame(rows).to_csv(result_path, index=False)
    if feasibility_rows:
        pd.DataFrame(feasibility_rows).to_csv(out_dir / "module_feasibility_trace.csv", index=False)
    script_paths = [
        Path(__file__).resolve(),
        Path(weather.__file__).resolve(),
        Path(__import__("run_eu27_rfnbo_frontier_aux").__file__).resolve(),
        Path(__import__("rfnbo_inventory_frontier").__file__).resolve(),
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_label": args.run_label,
        "estimand": "minimum shared internal H2 working inventory and generic cavern-module count required to serve exact annual NH3 output over a continuous historical weather sequence, under perfect foresight",
        "claim_boundary": "one public country reference point; not an NH3 facility, national mean, cavern site, operating reliability, or RFNBO certification",
        "countries": [args.country],
        "years": years,
        "training_years_for_fixed_design": list(weather.TRAIN_YEARS),
        "period_lengths_hours": period_lengths,
        "process_scale": {
            "normalized_model_demand_tph": weather.DEMAND_TPH,
            "hb_nameplate_multiple_of_nominal_service": args.hb_nameplate_multiple,
            "hb_nameplate_tph": args.hb_nameplate_multiple * weather.DEMAND_TPH,
            "hb_minimum_load_fraction_of_nameplate": args.min_load_fraction,
            "hb_ramp_fraction_of_nameplate_per_hour": args.ramp_fraction_nameplate_per_hour,
            "standardized_reference_plant_tpy_nh3": NH3_REFERENCE_TONNES_PER_YEAR,
            "plant_scale_from_model": PLANT_SCALE_FROM_1_TPH,
            "hydrogen_kg_per_t_nh3": H2_PER_NH3_KG_PER_T,
            "electrolyser_sec_kwh_per_kg_h2": ELECTROLYSER_SEC_KWH_PER_KG,
            "evaluation_auxiliary_mwh_per_t_nh3": args.auxiliary_mwh_per_t_nh3,
            "fixed_design_auxiliary_mwh_per_t_nh3": AUX_MWH_PER_T_DEFAULT,
            "storage_compression_kwh_per_kg_h2": args.compression_kwh_per_kg,
            "storage_loss_per_hour": args.storage_loss_per_hour,
        },
        "cavern_reference": {
            "source": "Hystories D7.1-1, public conceptual design",
            "standard_pressure_pa": P_STANDARD_PA,
            "standard_temperature_c": T_STANDARD_K - 273.15,
            "h2_density_ideal_gas_kg_per_sm3": H2_DENSITY_KG_PER_SM3,
            "cases": [
                {
                    "case_id": c.case_id,
                    "working_gas_million_sm3": c.working_gas_million_sm3,
                    "peak_withdrawal_million_sm3_day": c.peak_withdrawal_million_sm3_day,
                    "reference_site_cavern_count": c.reference_site_cavern_count,
                    "working_gas_kg": c.working_gas_kg,
                    "peak_withdrawal_kgph": c.peak_withdrawal_kg_per_hour,
                    "service_hours_500kt_y": c.reference_service_hours,
                } for c in CAVERN_CASES
            ],
        },
        "storage_flow_assumptions": {
            "withdrawal_to_injection_capacity_ratio": args.withdrawal_to_injection_ratio,
            "injection_rate_definition": "Hystories D7.2 site-level Q_i = Q_w/WTIR; applied proportionally to identical conceptual modules for screening",
            "documented_wtir_range": [1.0, 5.0],
            "documented_typical_wtir": 2.0,
        },
        "storage_compression_assumption": {
            "linearized_kwh_per_kg_h2": args.compression_kwh_per_kg,
            "public_anchor": "Hystories D7.1-1 Table 14; total installed brake power / total peak salt-cavern injection flow",
            "source_derived_cases_kwh_per_kg": {
                "low": 0.5868618278239139,
                "mid": 0.5869902018429064,
                "high": 0.584957711354134,
            },
            "limitations": "Conceptual site-level peak ratio used as constant marginal electricity; no site-specific compressor map, motor efficiency, or pressure-dependent part-load curve",
        },
        "storage_capex_screen": {
            "hystories_d7_2_subsurface_eur_per_kwh_h2_lhv": HYSTORIES_SUBSURFACE_CAPEX_EUR_PER_KWH,
            "hystories_d7_2_surface_eur_per_kw_h2_lhv_max_withdrawal": HYSTORIES_SURFACE_CAPEX_EUR_PER_KW_WITHDRAWAL,
            "hystories_cost_basis": "conceptual 250 million Sm3 reference site with 4-16 caverns; applying marginal rates to module equivalents is a linear rate-equivalent sensitivity, not a site or project quote",
            "hystories_reported_cost_estimate_accuracy": "typically 30-50%; conceptual European-scale rates, not project quotes",
            "ammonia_tank_reference_eur_per_t": NH3_TANK_CAPEX_EUR_PER_T,
            "ammonia_tank_source_scale_t": [60060, 74700],
            "ammonia_tank_reference_scope": "terminal-scale 22,000 m3-class tank cases; no smaller-buffer scale curve assumed",
            "reference_h2_lhv_kwh_per_kg": H2_LHV_KWH_PER_KG,
            "boundary": "rate-equivalent upfront storage CAPEX sensitivity only; conceptual site rates are linearly mapped to module equivalents and omit fixed/shared-site costs; ammonia tank comparator is terminal-scale; not a project quote, LCOA, NPV or commercial feasibility estimate; excludes process retrofit, financing, site-specific geology/connection, replacements and full-system costs",
        },
        "matching_definition": "strict hourly electrolyser eligibility or local-calendar-month energy-budget proxy; no price exception and no full RED/RFNBO certification",
        "boundary_conditions": args.boundaries,
        "solver": {"method": SOLVER_METHOD, "threads": 1},
        "input_hashes": {str(path): sha256_file(path) for path in inputs.values()},
        "code_hashes": {str(path): sha256_file(path) for path in script_paths},
        "capacity_design": caps,
        "results": [result_path.name] + (["module_feasibility_trace.csv"] if feasibility_rows else []),
        "row_count": len(rows),
        "feasibility_trace_rows": len(feasibility_rows),
        "all_module_counts_capped_at": args.max_modules,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="DE", choices=weather.COUNTRY_CODES)
    parser.add_argument("--years", nargs="+", type=int, default=[2015, 2016])
    parser.add_argument("--rules", nargs="+", choices=("monthly", "hourly"), default=["hourly"])
    parser.add_argument("--boundaries", nargs="+", choices=("cyclic", "zero"), default=["cyclic", "zero"])
    parser.add_argument("--process-cases", nargs="+", choices=("rigid", "flex"), default=["rigid", "flex"])
    parser.add_argument("--buffers", nargs="+", type=float, default=[0.0, 72.0, 168.0])
    parser.add_argument("--min-load-fraction", type=float, default=0.60,
                        help="HB minimum fraction of nameplate; 0.60 is the conservative primary case")
    parser.add_argument("--hb-nameplate-multiple", type=float,
                        default=weather.HB_NAMEPLATE_TPH / weather.DEMAND_TPH,
                        help="HB nameplate divided by fixed nominal ammonia service")
    parser.add_argument("--ramp-fraction-nameplate-per-hour", type=float,
                        default=weather.HB_RAMP_FRACTION_NAMEPLATE_PER_HOUR,
                        help="Absolute hourly HB ramp limit as a fraction of nameplate")
    parser.add_argument("--max-modules", type=int, default=MAX_MODULES_DEFAULT,
                        help="0 disables module-count search; otherwise integer search up to this limit")
    parser.add_argument("--auxiliary-mwh-per-t-nh3", type=float, default=AUX_MWH_PER_T_DEFAULT)
    parser.add_argument("--storage-loss-per-hour", type=float, default=STORAGE_LOSS_PER_HOUR)
    parser.add_argument("--compression-kwh-per-kg", type=float, default=STORAGE_COMPRESSION_KWH_PER_KG)
    parser.add_argument("--withdrawal-to-injection-ratio", type=float, default=2.0)
    parser.add_argument("--run-label", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_label):
        parser.error("--run-label must use only letters, digits, hyphens, and underscores")
    if (not 0 <= args.min_load_fraction <= 1 or args.hb_nameplate_multiple < 1
            or args.min_load_fraction * args.hb_nameplate_multiple > 1
            or args.ramp_fraction_nameplate_per_hour < 0 or args.max_modules < 0
            or args.auxiliary_mwh_per_t_nh3 < 0 or args.compression_kwh_per_kg < 0
            or not 0 <= args.storage_loss_per_hour < 1
            or not 1 <= args.withdrawal_to_injection_ratio <= 5):
        parser.error("invalid process load or module count")
    if not args.years or any(b != a + 1 for a, b in zip(args.years, args.years[1:])):
        parser.error("years must form a nonempty consecutive sequence")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

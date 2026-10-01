"""Auxiliary-electricity variant of the public H2/NH3 inventory LP.

Unlike the legacy v1 screen, this model adds a literature-bounded, throughput-
proportional HB/ASU electricity load to the site power balance. The temporal-
matching proxy is applied only to electrolyser input. It still excludes H2
compression beyond the stated electrolyser SEC, start-up/shutdown, reliability,
and full certification details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

SOLVER_METHOD = "highs-ds"

try:
    from .rfnbo_inventory_frontier import (
        H2_LHV_MWH_PER_KG,
        H2_PER_NH3_KG_PER_T,
        LPSettings,
    )
except ImportError:  # Support direct execution from the src directory.
    from rfnbo_inventory_frontier import (
        H2_LHV_MWH_PER_KG,
        H2_PER_NH3_KG_PER_T,
        LPSettings,
    )


@dataclass
class AuxFrontierPoint:
    rule: str
    flexible: bool
    buffer_limit_t: float
    h2_capacity_kg: float
    h2_capacity_mwh_lhv: float
    h2_inventory_max_kg: float
    nh3_buffer_capacity_t: float
    nh3_buffer_used_t: float
    nh3_buffer_used_hours: float
    annual_nh3_delivered_t: float
    annual_electrolysis_mwh: float
    annual_auxiliary_mwh: float
    annual_total_site_electricity_mwh: float
    renewable_matching_share: float
    price_exception_share: float
    renewable_direct_use_mwh: float
    grid_import_mwh: float
    price_exception_import_mwh: float
    curtailed_renewable_mwh: float
    max_electricity_balance_residual_mw: float
    max_h2_balance_residual_kg: float
    max_nh3_balance_residual_t: float
    max_hourly_matching_violation_mwh: float
    max_monthly_matching_violation_mwh: float
    max_ramp_tph: float
    ramp_limit_tph: float
    ramp_binding_hours: int
    objective_mode: str
    market_energy_cost_eur: float
    annualized_h2_storage_cost_eur: float
    annualized_nh3_storage_cost_eur: float
    proxy_annual_cost_eur: float
    status: str


def _add(rows: list[int], cols: list[int], vals: list[float],
         row: int, col: int, value: float) -> None:
    if value:
        rows.append(row)
        cols.append(col)
        vals.append(float(value))


def solve_frontier_point_aux(
    renewable_mw: Sequence[float],
    local_months: Sequence[str],
    *,
    rule: str,
    flexible: bool,
    ammonia_buffer_limit_t: float,
    settings: LPSettings,
    auxiliary_mwh_per_t_nh3: float,
    day_ahead_eur_mwh: Sequence[float] | None = None,
    objective_mode: str = "min_h2",
    h2_capacity_limit_kg: float | None = None,
    h2_storage_cost_eur_kg_year: float = 0.0,
    nh3_storage_cost_eur_t_year: float = 0.0,
) -> tuple[AuxFrontierPoint, dict[str, np.ndarray]]:
    """Solve a two-inventory frontier with q-proportional HB/ASU power.

    `auxiliary_mwh_per_t_nh3` is an explicit scenario parameter. The hourly
    site load is electrolyser electricity plus auxiliary electricity. The
    temporal-matching proxy is applied to electrolyser input only; auxiliary
    loads remain in the physical site power balance. This is not a full-plant
    energy or RFNBO-certification model.
    """
    g = np.asarray(renewable_mw, dtype=float)
    months = np.asarray(local_months, dtype=str)
    if rule not in {"monthly", "hourly"}:
        raise ValueError("rule must be 'monthly' or 'hourly'")
    if objective_mode not in {"min_h2", "min_nh3", "market_cost"}:
        raise ValueError("invalid objective_mode")
    if g.ndim != 1 or months.shape != g.shape or g.size == 0:
        raise ValueError("renewable profile and month labels must be non-empty and aligned")
    if not np.isfinite(g).all() or np.any(g < 0):
        raise ValueError("renewable generation must be finite and non-negative")
    alpha = float(auxiliary_mwh_per_t_nh3)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("auxiliary electricity must be finite and non-negative")

    n = int(g.size)
    k_e = float(settings.electrolyser_mw)
    kg_per_mwh = 1000.0 / float(settings.electrolyser_kwh_per_kg)
    qd = float(settings.ammonia_demand_tph)
    q_hi = float(settings.hb_nameplate_tph)
    q_lo = q_hi * float(settings.hb_min_load_fraction) if flexible else qd
    ramp = q_hi * float(settings.hb_ramp_fraction_nameplate_per_hour)
    if not (0 < qd <= q_hi and 0 <= q_lo <= qd and k_e > 0 and ramp >= 0):
        raise ValueError("inconsistent process or electrolyser bounds")
    if ammonia_buffer_limit_t < 0:
        raise ValueError("ammonia buffer limit must be non-negative")
    if not flexible and ammonia_buffer_limit_t != 0:
        raise ValueError("a rigid synthesis case must use a zero ammonia-buffer limit")
    if h2_capacity_limit_kg is not None and h2_capacity_limit_kg < 0:
        raise ValueError("hydrogen-capacity limit must be non-negative")
    grid_limit = (k_e if settings.grid_connection_mw is None
                  else float(settings.grid_connection_mw))
    if grid_limit < 0:
        raise ValueError("grid connection limit must be non-negative")

    # Variable blocks: electrolyser u, NH3 output q, H2 stock h, NH3 stock a,
    # price exception x, direct renewable r, grid imports m, capacities H and B.
    iu = np.arange(0, n)
    iq = np.arange(n, 2 * n)
    ih = np.arange(2 * n, 3 * n)
    ia = np.arange(3 * n, 4 * n)
    ix = np.arange(4 * n, 5 * n)
    ir = np.arange(5 * n, 6 * n)
    im = np.arange(6 * n, 7 * n)
    i_hcap, i_bcap, nvar = 7 * n, 7 * n + 1, 7 * n + 2

    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_vals: list[float] = []
    b_eq = np.empty(3 * n, dtype=float)
    for t in range(n):
        prev = (t - 1) % n
        rh, ra, re = 3 * t, 3 * t + 1, 3 * t + 2
        _add(eq_rows, eq_cols, eq_vals, rh, int(ih[t]), 1)
        _add(eq_rows, eq_cols, eq_vals, rh, int(ih[prev]), -1)
        _add(eq_rows, eq_cols, eq_vals, rh, int(iu[t]), -kg_per_mwh)
        _add(eq_rows, eq_cols, eq_vals, rh, int(iq[t]), H2_PER_NH3_KG_PER_T)
        _add(eq_rows, eq_cols, eq_vals, ra, int(ia[t]), 1)
        _add(eq_rows, eq_cols, eq_vals, ra, int(ia[prev]), -1)
        _add(eq_rows, eq_cols, eq_vals, ra, int(iq[t]), -1)
        _add(eq_rows, eq_cols, eq_vals, re, int(iu[t]), 1)
        _add(eq_rows, eq_cols, eq_vals, re, int(iq[t]), alpha)
        _add(eq_rows, eq_cols, eq_vals, re, int(ir[t]), -1)
        _add(eq_rows, eq_cols, eq_vals, re, int(im[t]), -1)
        b_eq[rh], b_eq[ra], b_eq[re] = 0.0, -qd, 0.0
    a_eq = coo_matrix((eq_vals, (eq_rows, eq_cols)),
                      shape=(3 * n, nvar)).tocsr()

    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_vals: list[float] = []
    b_ub: list[float] = []
    row = 0
    for t in range(n):
        _add(ub_rows, ub_cols, ub_vals, row, int(ih[t]), 1)
        _add(ub_rows, ub_cols, ub_vals, row, i_hcap, -1)
        b_ub.append(0.0)
        row += 1
        _add(ub_rows, ub_cols, ub_vals, row, int(ia[t]), 1)
        _add(ub_rows, ub_cols, ub_vals, row, i_bcap, -1)
        b_ub.append(0.0)
        row += 1
    for t in range(n):
        _add(ub_rows, ub_cols, ub_vals, row, int(ix[t]), 1)
        _add(ub_rows, ub_cols, ub_vals, row, int(im[t]), -1)
        b_ub.append(0.0)
        row += 1

    # The temporal-eligibility proxy is applied to electrolyser electricity.
    # HB/ASU auxiliary electricity remains in the physical site balance, but
    # is not treated as electrolyser input by this proxy. This is not a full
    # RFNBO certification model.
    if rule == "hourly":
        for t in range(n):
            _add(ub_rows, ub_cols, ub_vals, row, int(iu[t]), 1)
            _add(ub_rows, ub_cols, ub_vals, row, int(ix[t]), -1)
            b_ub.append(float(g[t]))
            row += 1
    else:
        for month in dict.fromkeys(months.tolist()):
            ixm = np.flatnonzero(months == month)
            for t in ixm:
                _add(ub_rows, ub_cols, ub_vals, row, int(iu[t]), 1)
                _add(ub_rows, ub_cols, ub_vals, row, int(ix[t]), -1)
            b_ub.append(float(g[ixm].sum()))
            row += 1

    if flexible:
        for t in range(n):
            prev = (t - 1) % n
            _add(ub_rows, ub_cols, ub_vals, row, int(iq[t]), 1)
            _add(ub_rows, ub_cols, ub_vals, row, int(iq[prev]), -1)
            b_ub.append(ramp)
            row += 1
            _add(ub_rows, ub_cols, ub_vals, row, int(iq[t]), -1)
            _add(ub_rows, ub_cols, ub_vals, row, int(iq[prev]), 1)
            b_ub.append(ramp)
            row += 1
    a_ub = coo_matrix((ub_vals, (ub_rows, ub_cols)),
                      shape=(row, nvar)).tocsr()

    price = None
    if day_ahead_eur_mwh is not None:
        price = np.asarray(day_ahead_eur_mwh, dtype=float)
        if price.shape != g.shape or not np.isfinite(price).all():
            raise ValueError("day-ahead prices must be finite and aligned hourly values")
    cheap = np.zeros(n, dtype=bool)
    if settings.price_exception_eur_mwh is not None:
        if price is None:
            raise ValueError("day-ahead prices are required for the price-exception case")
        cheap = price <= float(settings.price_exception_eur_mwh)

    bounds: list[tuple[float, float | None]] = []
    bounds.extend([(0.0, k_e)] * n)
    bounds.extend([(q_lo, q_hi)] * n)
    bounds.extend([(0.0, None)] * n)
    if ammonia_buffer_limit_t == 0 or not flexible:
        bounds.extend([(0.0, 0.0)] * n)
        b_cap_bound: tuple[float, float] = (0.0, 0.0)
    else:
        bounds.extend([(0.0, None)] * n)
        b_cap_bound = (0.0, float(ammonia_buffer_limit_t))
    bounds.extend([(0.0, grid_limit if cheap[t] else 0.0) for t in range(n)])
    bounds.extend([(0.0, float(g[t])) for t in range(n)])
    bounds.extend([(0.0, grid_limit)] * n)
    bounds.append((0.0, None if h2_capacity_limit_kg is None
                   else float(h2_capacity_limit_kg)))
    bounds.append(b_cap_bound)

    c = np.zeros(nvar, dtype=float)
    if objective_mode == "min_h2":
        c[i_hcap] = 1.0
    elif objective_mode == "min_nh3":
        c[i_bcap] = 1.0
    else:
        if price is None:
            raise ValueError("market_cost requires aligned public day-ahead prices")
        if h2_storage_cost_eur_kg_year < 0 or nh3_storage_cost_eur_t_year < 0:
            raise ValueError("storage cost coefficients must be non-negative")
        c[im] = price
        c[i_hcap] = h2_storage_cost_eur_kg_year
        c[i_bcap] = nh3_storage_cost_eur_t_year
    first = linprog(c, A_ub=a_ub, b_ub=np.asarray(b_ub), A_eq=a_eq,
                    b_eq=b_eq, bounds=bounds, method=SOLVER_METHOD)
    if not first.success:
        raise RuntimeError(f"LP failed ({rule}, flexible={flexible}, B={ammonia_buffer_limit_t}): {first.message}")

    if objective_mode in {"min_h2", "min_nh3"}:
        first_index, second_index = ((i_hcap, i_bcap) if objective_mode == "min_h2"
                                     else (i_bcap, i_hcap))
        first_value = float(first.x[first_index])
        bounds_2 = list(bounds)
        bounds_2[first_index] = (first_value, first_value)
        c2 = np.zeros(nvar, dtype=float)
        c2[second_index] = 1.0
        second = linprog(c2, A_ub=a_ub, b_ub=np.asarray(b_ub), A_eq=a_eq,
                         b_eq=b_eq, bounds=bounds_2, method=SOLVER_METHOD)
        if not second.success:
            raise RuntimeError(f"Lexicographic inventory LP failed: {second.message}")
        second_value = float(second.x[second_index])
        bounds_3 = list(bounds_2)
        bounds_3[second_index] = (second_value, second_value)
        c3 = np.zeros(nvar, dtype=float)
        c3[im] = 1.0
        third = linprog(c3, A_ub=a_ub, b_ub=np.asarray(b_ub), A_eq=a_eq,
                        b_eq=b_eq, bounds=bounds_3, method=SOLVER_METHOD)
        if not third.success:
            raise RuntimeError(f"Renewable-priority LP failed: {third.message}")
        x = third.x
    else:
        x = first.x

    u, q, h, a, exception, renewable_use, grid_import = (
        x[iu], x[iq], x[ih], x[ia], x[ix], x[ir], x[im]
    )
    aux = alpha * q
    site_load = u + aux
    h_res = np.roll(h, 1) + kg_per_mwh * u - H2_PER_NH3_KG_PER_T * q - h
    a_res = np.roll(a, 1) + q - qd - a
    total_energy = float(np.sum(site_load))
    market_cost = float(np.dot(price, grid_import)) if price is not None else 0.0
    h2_cost = float(x[i_hcap] * h2_storage_cost_eur_kg_year)
    nh3_cost = float(x[i_bcap] * nh3_storage_cost_eur_t_year)
    hourly_violation = np.maximum(u - exception - g, 0.0)
    monthly_violation = []
    for month in dict.fromkeys(months.tolist()):
        ixm = np.flatnonzero(months == month)
        monthly_violation.append(max(
            0.0, float(np.sum(u[ixm] - exception[ixm]) - np.sum(g[ixm]))
        ))
    q_change = np.abs(q - np.roll(q, 1))
    point = AuxFrontierPoint(
        rule=rule, flexible=bool(flexible), buffer_limit_t=float(ammonia_buffer_limit_t),
        h2_capacity_kg=float(x[i_hcap]),
        h2_capacity_mwh_lhv=float(x[i_hcap] * H2_LHV_MWH_PER_KG),
        h2_inventory_max_kg=float(np.max(h)),
        nh3_buffer_capacity_t=float(x[i_bcap]),
        nh3_buffer_used_t=float(np.max(a)),
        nh3_buffer_used_hours=float(np.max(a) / qd),
        annual_nh3_delivered_t=float(n * qd),
        annual_electrolysis_mwh=float(np.sum(u)),
        annual_auxiliary_mwh=float(np.sum(aux)),
        annual_total_site_electricity_mwh=total_energy,
        renewable_matching_share=(float(np.sum(u - exception) / np.sum(u))
                                  if np.sum(u) else float("nan")),
        price_exception_share=(float(np.sum(exception) / np.sum(u))
                               if np.sum(u) else 0.0),
        renewable_direct_use_mwh=float(np.sum(renewable_use)),
        grid_import_mwh=float(np.sum(grid_import)),
        price_exception_import_mwh=float(np.sum(exception)),
        curtailed_renewable_mwh=float(np.sum(g - renewable_use)),
        max_electricity_balance_residual_mw=float(np.max(np.abs(site_load - renewable_use - grid_import))),
        max_h2_balance_residual_kg=float(np.max(np.abs(h_res))),
        max_nh3_balance_residual_t=float(np.max(np.abs(a_res))),
        max_hourly_matching_violation_mwh=float(np.max(hourly_violation)),
        max_monthly_matching_violation_mwh=float(max(monthly_violation, default=0.0)),
        max_ramp_tph=float(np.max(q_change)), ramp_limit_tph=float(ramp if flexible else 0.0),
        ramp_binding_hours=int(np.sum(q_change >= ramp - 1e-7)) if flexible and ramp > 0 else 0,
        objective_mode=objective_mode, market_energy_cost_eur=market_cost,
        annualized_h2_storage_cost_eur=h2_cost,
        annualized_nh3_storage_cost_eur=nh3_cost,
        proxy_annual_cost_eur=market_cost + h2_cost + nh3_cost, status="optimal",
    )
    series = {
        "electrolysis_mwh": u, "auxiliary_load_mw": aux,
        "total_site_load_mw": site_load, "nh3_production_tph": q,
        "h2_inventory_kg": h, "nh3_inventory_t": a,
        "price_exception_mwh": exception,
        "matching_electricity_mwh": u - exception,
        "renewable_direct_mwh": renewable_use, "grid_import_mwh": grid_import,
    }
    return point, series

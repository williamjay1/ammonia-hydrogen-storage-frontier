"""Hourly LP for a public-data RFNBO/Haber-Bosch two-inventory frontier.

The LP is a conditional design-screening model, not plant telemetry or a
certification engine. It keeps the hydrogen eligibility ledger separate from
the H2 and NH3 material inventories. All time series are one-hour interval
values indexed in UTC; month labels are supplied in the bidding-zone timezone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


H2_PER_NH3_KG_PER_T = 1000.0 * 3.0 * 2.01588 / (2.0 * 17.03052)
H2_LHV_MWH_PER_KG = 0.03333
# HiGHS defaults to automatic thread selection. Bound each full-year LP to one
# solver thread so repeated lexicographic solves do not multiply memory use on
# high-core-count hosts. SciPy forwards these HiGHS-specific options.
HIGHS_LP_OPTIONS = {"threads": 1, "parallel": False}


@dataclass(frozen=True)
class LPSettings:
    electrolyser_mw: float
    grid_connection_mw: float | None = None
    electrolyser_kwh_per_kg: float = 52.0
    ammonia_demand_tph: float = 1.0
    hb_nameplate_tph: float = 1.5
    hb_min_load_fraction: float = 0.40
    # Core hourly case: 0.60 permits the full 40%-to-100% operating swing
    # in one interval and is therefore non-binding. Stricter rates are
    # reported only as explicit ramp-stress sensitivities.
    hb_ramp_fraction_nameplate_per_hour: float = 0.60
    price_exception_eur_mwh: float | None = None


@dataclass
class FrontierPoint:
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


def _add(
    rows: list[int], cols: list[int], vals: list[float],
    row: int, col: int, value: float,
) -> None:
    if value:
        rows.append(row)
        cols.append(col)
        vals.append(float(value))


def solve_frontier_point(
    renewable_mw: Sequence[float],
    local_months: Sequence[str],
    *,
    rule: str,
    flexible: bool,
    ammonia_buffer_limit_t: float,
    settings: LPSettings,
    day_ahead_eur_mwh: Sequence[float] | None = None,
    objective_mode: str = "min_h2",
    h2_capacity_limit_kg: float | None = None,
    h2_storage_cost_eur_kg_year: float = 0.0,
    nh3_storage_cost_eur_t_year: float = 0.0,
) -> tuple[FrontierPoint, dict[str, np.ndarray]]:
    """Solve one point on a conditional two-inventory operating frontier.

    `min_h2` minimizes H2 capacity at a fixed NH3-buffer ceiling and then
    minimizes NH3 capacity; `min_nh3` does the reverse subject to an optional
    H2-capacity ceiling. A final tie-break minimizes grid imports. `monthly`
    enforces one energy budget per local calendar month; `hourly` enforces it
    in each hour. Optional cheap-electricity eligibility implements only the
    EUR 20/MWh limb of Article 6.
    `min_nh3` finds the minimum product-buffer capacity subject to an optional
    H2-capacity ceiling. `market_cost` minimizes hourly day-ahead-priced
    electricity plus annualized storage-capacity proxies; it is not a full
    plant LCOA or PPA settlement.
    """
    g = np.asarray(renewable_mw, dtype=float)
    months = np.asarray(local_months, dtype=str)
    if rule not in {"monthly", "hourly"}:
        raise ValueError("rule must be 'monthly' or 'hourly'")
    if objective_mode not in {"min_h2", "min_nh3", "market_cost"}:
        raise ValueError("objective_mode must be 'min_h2', 'min_nh3', or 'market_cost'")
    if g.ndim != 1 or months.shape != g.shape or g.size == 0:
        raise ValueError("renewable profile and month labels must be non-empty and aligned")
    if not np.isfinite(g).all() or np.any(g < 0):
        raise ValueError("renewable generation must be finite and non-negative")
    n = int(g.size)
    settings = settings
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
    grid_limit = k_e if settings.grid_connection_mw is None else float(settings.grid_connection_mw)
    if grid_limit < 0:
        raise ValueError("grid connection limit must be non-negative")

    # Variable blocks: electrolyser electricity u, NH3 production q, H2 stock h,
    # NH3 stock a, price-exception electricity x, direct renewable use r, grid
    # imports m, followed by the two inventory capacities H and B.
    iu = np.arange(0, n)
    iq = np.arange(n, 2 * n)
    ih = np.arange(2 * n, 3 * n)
    ia = np.arange(3 * n, 4 * n)
    ix = np.arange(4 * n, 5 * n)
    ir = np.arange(5 * n, 6 * n)
    im = np.arange(6 * n, 7 * n)
    i_hcap, i_bcap, nvar = 7 * n, 7 * n + 1, 7 * n + 2

    # Two cyclic material balances plus one hourly electricity balance.
    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_vals: list[float] = []
    b_eq = np.empty(3 * n, dtype=float)
    for t in range(n):
        prev = (t - 1) % n
        rh = 3 * t
        ra = rh + 1
        re = rh + 2
        _add(eq_rows, eq_cols, eq_vals, rh, int(ih[t]), 1)
        _add(eq_rows, eq_cols, eq_vals, rh, int(ih[prev]), -1)
        _add(eq_rows, eq_cols, eq_vals, rh, int(iu[t]), -kg_per_mwh)
        _add(eq_rows, eq_cols, eq_vals, rh, int(iq[t]), H2_PER_NH3_KG_PER_T)
        _add(eq_rows, eq_cols, eq_vals, ra, int(ia[t]), 1)
        _add(eq_rows, eq_cols, eq_vals, ra, int(ia[prev]), -1)
        _add(eq_rows, eq_cols, eq_vals, ra, int(iq[t]), -1)
        _add(eq_rows, eq_cols, eq_vals, re, int(iu[t]), 1)
        _add(eq_rows, eq_cols, eq_vals, re, int(ir[t]), -1)
        _add(eq_rows, eq_cols, eq_vals, re, int(im[t]), -1)
        b_eq[rh] = 0.0
        b_eq[ra] = -qd
        b_eq[re] = 0.0
    a_eq = coo_matrix((eq_vals, (eq_rows, eq_cols)), shape=(3 * n, nvar)).tocsr()

    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_vals: list[float] = []
    b_ub: list[float] = []
    row = 0
    # Capacity ceilings on each state of charge.
    for t in range(n):
        _add(ub_rows, ub_cols, ub_vals, row, int(ih[t]), 1)
        _add(ub_rows, ub_cols, ub_vals, row, i_hcap, -1)
        b_ub.append(0.0)
        row += 1

    # Price-exception electricity must be a subset of actual grid imports.
    # Without this constraint the eligibility ledger could label onsite
    # renewable consumption as cheap-grid electricity.
    for t in range(n):
        _add(ub_rows, ub_cols, ub_vals, row, int(ix[t]), 1)
        _add(ub_rows, ub_cols, ub_vals, row, int(im[t]), -1)
        b_ub.append(0.0)
        row += 1
        _add(ub_rows, ub_cols, ub_vals, row, int(ia[t]), 1)
        _add(ub_rows, ub_cols, ub_vals, row, i_bcap, -1)
        b_ub.append(0.0)
        row += 1

    # Renewable correlation budget. Cheap-price electricity is accounted for
    # once in x and cannot be counted again against renewable generation.
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

    # Symmetric hourly ramp limit, including the cyclic year boundary.
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

    a_ub = coo_matrix((ub_vals, (ub_rows, ub_cols)), shape=(row, nvar)).tocsr()

    cheap = np.zeros(n, dtype=bool)
    price = None
    if day_ahead_eur_mwh is not None:
        price = np.asarray(day_ahead_eur_mwh, dtype=float)
        if price.shape != g.shape or not np.isfinite(price).all():
            raise ValueError("day-ahead prices must be finite and aligned hourly values")
    if settings.price_exception_eur_mwh is not None:
        if price is None:
            raise ValueError("day-ahead prices are required for the price-exception case")
        cheap = price <= float(settings.price_exception_eur_mwh)

    bounds: list[tuple[float, float | None]] = []
    bounds.extend([(0.0, k_e)] * n)  # u
    bounds.extend([(q_lo, q_hi)] * n)  # q
    bounds.extend([(0.0, None)] * n)  # H2 inventory
    if ammonia_buffer_limit_t == 0 or not flexible:
        bounds.extend([(0.0, 0.0)] * n)
        b_cap_bound: tuple[float, float] = (0.0, 0.0)
    else:
        bounds.extend([(0.0, None)] * n)
        b_cap_bound = (0.0, float(ammonia_buffer_limit_t))
    bounds.extend([(0.0, k_e if cheap[t] else 0.0) for t in range(n)])
    bounds.extend([(0.0, float(g[t])) for t in range(n)])  # direct renewable use
    bounds.extend([(0.0, grid_limit)] * n)  # grid import
    h2_cap_upper = (None if h2_capacity_limit_kg is None
                    else float(h2_capacity_limit_kg))
    bounds.append((0.0, h2_cap_upper))  # H2 working capacity
    bounds.append(b_cap_bound)  # NH3 product buffer capacity

    c = np.zeros(nvar, dtype=float)
    if objective_mode == "min_h2":
        c[i_hcap] = 1.0
    elif objective_mode == "min_nh3":
        c[i_bcap] = 1.0
    else:
        if price is None:
            raise ValueError("market_cost requires aligned public day-ahead prices")
        if h2_storage_cost_eur_kg_year < 0 or nh3_storage_cost_eur_t_year < 0:
            raise ValueError("annualized storage cost coefficients must be non-negative")
        c[im] = price
        c[i_hcap] = float(h2_storage_cost_eur_kg_year)
        c[i_bcap] = float(nh3_storage_cost_eur_t_year)
    first = linprog(c, A_ub=a_ub, b_ub=np.asarray(b_ub), A_eq=a_eq,
                    b_eq=b_eq, bounds=bounds, method="highs",
                    options=HIGHS_LP_OPTIONS)
    if not first.success:
        raise RuntimeError(f"LP infeasible/failed ({rule}, flexible={flexible}, B={ammonia_buffer_limit_t}): {first.message}")
    if objective_mode in {"min_h2", "min_nh3"}:
        # Lexicographic frontier: first minimize one inventory capacity,
        # then fix it exactly and minimize the other, then minimize grid
        # imports without changing either capacity. This removes solver
        # degeneracy and prioritizes physically available renewable power.
        if objective_mode == "min_h2":
            first_index, second_index = i_hcap, i_bcap
        else:
            first_index, second_index = i_bcap, i_hcap
        first_value = float(first.x[first_index])
        bounds_2 = list(bounds)
        bounds_2[first_index] = (first_value, first_value)
        c2 = np.zeros(nvar, dtype=float)
        c2[second_index] = 1.0
        second = linprog(c2, A_ub=a_ub, b_ub=np.asarray(b_ub), A_eq=a_eq,
                         b_eq=b_eq, bounds=bounds_2, method="highs",
                         options=HIGHS_LP_OPTIONS)
        if not second.success:
            raise RuntimeError(f"Lexicographic inventory LP failed: {second.message}")
        bounds_3 = list(bounds_2)
        second_value = float(second.x[second_index])
        bounds_3[second_index] = (second_value, second_value)
        c3 = np.zeros(nvar, dtype=float)
        c3[im] = 1.0
        third = linprog(c3, A_ub=a_ub, b_ub=np.asarray(b_ub), A_eq=a_eq,
                        b_eq=b_eq, bounds=bounds_3, method="highs",
                        options=HIGHS_LP_OPTIONS)
        if not third.success:
            raise RuntimeError(f"Renewable-priority LP failed: {third.message}")
        x = third.x
    else:
        x = first.x
    u, q, h, a, exception, renewable_use, grid_import = (
        x[iu], x[iq], x[ih], x[ia], x[ix], x[ir], x[im]
    )
    h_res = np.roll(h, 1) + kg_per_mwh * u - H2_PER_NH3_KG_PER_T * q - h
    a_res = np.roll(a, 1) + q - qd - a
    total_energy = float(np.sum(u))
    market_cost = float(np.dot(price, grid_import)) if price is not None else 0.0
    h2_cost = float(x[i_hcap] * h2_storage_cost_eur_kg_year)
    nh3_cost = float(x[i_bcap] * nh3_storage_cost_eur_t_year)
    hourly_match_violation = np.maximum(u - exception - g, 0.0)
    monthly_match_violation = []
    for month in dict.fromkeys(months.tolist()):
        ixm = np.flatnonzero(months == month)
        monthly_match_violation.append(
            max(0.0, float(np.sum(u[ixm] - exception[ixm]) - np.sum(g[ixm])))
        )
    q_change = np.abs(q - np.roll(q, 1))
    output = FrontierPoint(
        rule=rule,
        flexible=bool(flexible),
        buffer_limit_t=float(ammonia_buffer_limit_t),
        h2_capacity_kg=float(x[i_hcap]),
        h2_capacity_mwh_lhv=float(x[i_hcap] * H2_LHV_MWH_PER_KG),
        h2_inventory_max_kg=float(np.max(h)),
        nh3_buffer_capacity_t=float(x[i_bcap]),
        nh3_buffer_used_t=float(np.max(a)),
        nh3_buffer_used_hours=float(np.max(a) / qd),
        annual_nh3_delivered_t=float(n * qd),
        annual_electrolysis_mwh=total_energy,
        renewable_matching_share=(float(np.sum(u - exception) / total_energy)
                                  if total_energy else float("nan")),
        price_exception_share=(float(np.sum(exception) / total_energy)
                               if total_energy else 0.0),
        renewable_direct_use_mwh=float(np.sum(renewable_use)),
        grid_import_mwh=float(np.sum(grid_import)),
        price_exception_import_mwh=float(np.sum(exception)),
        curtailed_renewable_mwh=float(np.sum(g - renewable_use)),
        max_electricity_balance_residual_mw=float(
            np.max(np.abs(u - renewable_use - grid_import))
        ),
        max_h2_balance_residual_kg=float(np.max(np.abs(h_res))),
        max_nh3_balance_residual_t=float(np.max(np.abs(a_res))),
        max_hourly_matching_violation_mwh=float(np.max(hourly_match_violation)),
        max_monthly_matching_violation_mwh=float(max(monthly_match_violation, default=0.0)),
        max_ramp_tph=float(np.max(q_change)),
        ramp_limit_tph=float(ramp if flexible else 0.0),
        ramp_binding_hours=int(np.sum(q_change >= ramp - 1e-7))
        if flexible and ramp > 0 else 0,
        objective_mode=objective_mode,
        market_energy_cost_eur=market_cost,
        annualized_h2_storage_cost_eur=h2_cost,
        annualized_nh3_storage_cost_eur=nh3_cost,
        proxy_annual_cost_eur=market_cost + h2_cost + nh3_cost,
        status="optimal",
    )
    series = {"electrolysis_mwh": u, "nh3_production_tph": q,
              "h2_inventory_kg": h, "nh3_inventory_t": a,
              "price_exception_mwh": exception,
              "matching_electricity_mwh": u - exception,
              "renewable_direct_mwh": renewable_use,
              "grid_import_mwh": grid_import}
    return output, series


def stoichiometry_check() -> dict[str, float]:
    """Return formula checks used by unit tests and run manifests."""
    h2_mol = 3.0
    nh3_mol = 2.0
    kg_h2 = h2_mol * 2.01588
    kg_nh3 = nh3_mol * 17.03052
    exact = kg_h2 / kg_nh3 * 1000.0
    return {"exact_kg_h2_per_t_nh3": exact,
            "model_kg_h2_per_t_nh3": H2_PER_NH3_KG_PER_T,
            "relative_error": abs(exact - H2_PER_NH3_KG_PER_T) / exact}

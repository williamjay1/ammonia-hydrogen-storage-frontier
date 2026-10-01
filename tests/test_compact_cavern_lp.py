from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_eu27_cavern_module_frontier as model
from compact_cavern_lp import solve_compact


@pytest.mark.parametrize("boundary", ["cyclic", "zero"])
@pytest.mark.parametrize("flexible", [False, True])
@pytest.mark.parametrize("matching", ["hourly", "monthly"])
def test_eliminated_flow_formulation_preserves_objectives_and_feasibility(
        boundary, flexible, matching):
    generation = np.array([0., 35., 25., 0., 10., 40., 25., 10.])
    months = np.repeat("2020-01", len(generation))
    args = dict(electrolyser_mw=30., grid_connection_mw=30.,
                min_hb_load_fraction=.4, flexible=flexible,
                nh3_buffer_limit_hours=2., matching_rule=matching,
                boundary=boundary, hb_nameplate_tph=1.5,
                cavern=model.CAVERN_CASES[-1], cavern_count=1,
                allow_short_periods_for_test=True)
    c, ub, b, eq, beq, bounds, idx = model.build_lp(
        generation, months, [len(generation)], **args)
    for stage in ("h2", "nh3", "grid"):
        objective = np.zeros_like(c)
        if stage == "h2":
            objective[idx.h2_capacity] = 1
        elif stage == "nh3":
            objective[idx.nh3_capacity] = 1
        else:
            objective[idx.starts["grid_import_mw"]:idx.starts["grid_import_mw"]+idx.hours] = 1
        original = linprog(objective, A_ub=ub, b_ub=b, A_eq=eq,
                           b_eq=beq, bounds=bounds, method="highs-ds",
                           options={"threads": 1})
        compact = solve_compact(objective, A_ub=ub, b_ub=b, A_eq=eq,
                                b_eq=beq, bounds=bounds, method="highs-ipm")
        assert original.status == compact.status
        if compact.success:
            assert compact.fun == pytest.approx(original.fun, abs=1e-6)
            assert np.max(np.abs(eq @ compact.x - beq)) < 1e-6
            assert np.max(ub @ compact.x - b) < 1e-6
            if stage == "h2":
                bounds[idx.h2_capacity] = (original.fun, original.fun)
            elif stage == "nh3":
                bounds[idx.nh3_capacity] = (original.fun, original.fun)

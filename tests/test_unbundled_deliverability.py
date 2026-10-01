from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_eho_unbundled_deliverability import minimum_deliverability
from run_eu27_cavern_module_frontier import H2_PER_NH3_KG_PER_T, H2_KG_PER_MWH_ELECTROLYSER


def test_rate_relief_at_same_volume_preserves_hourly_product_service():
    h = H2_PER_NH3_KG_PER_T
    generation = np.array([0., 2*h, 0., 2*h])/H2_KG_PER_MWH_ELECTROLYSER
    capacity = {"electrolyser_mw": 30., "grid_connection_mw": 30.}
    outputs = []
    for flexible, floor, buffer in ((False, 1., 0.), (True, .4, 1.)):
        outputs.append(minimum_deliverability(
            generation, np.repeat("2020-01", 4), capacities=capacity,
            q_floor=floor, flexible=flexible, buffer_hours=buffer,
            fixed_volume=2*h, wtir=1., test_short=True))
    assert outputs[0]["minimum_withdrawal_kgph_per_normalized_tph"] == pytest.approx(h, abs=1e-6)
    assert outputs[1]["minimum_withdrawal_kgph_per_normalized_tph"] == pytest.approx(.6*h, abs=1e-6)

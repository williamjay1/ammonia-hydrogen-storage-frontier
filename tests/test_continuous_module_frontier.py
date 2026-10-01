import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_eho_continuous_module_frontier import continuous_frontier


@pytest.mark.parametrize('flex,floor,buffer,expected',[(False,1.,0.,1.),(True,.4,1.,1./.6)])
def test_frontier_recovers_volume_rate_and_service_scale(flex,floor,buffer,expected):
    h=177.55300484072126
    result=continuous_frontier(np.array([0.,2*h,0.,2*h])*52/1000,
            np.ones(4,dtype=int),capacities={'electrolyser_mw':30.,'grid_connection_mw':30.},
            q_floor=floor,flexible=flex,buffer_hours=buffer,reference_volume=2*h,
            reference_withdrawal=h,wtir=1.,test_short=True)
    assert result['maximum_service_tph_per_reference_module']==pytest.approx(expected,abs=1e-6)
    assert result['max_primal_residual']<1e-5

import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import resume_eho_threshold_expanded as runner


def test_full_year_constructive_witness_respects_original_matrix(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('A checked witness does not need an LP solve')
    monkeypatch.setattr(runner,'linprog',forbidden)
    h=runner.site.model.H2_PER_NH3_KG_PER_T
    eta=runner.site.model.H2_KG_PER_MWH_ELECTROLYSER
    generation=np.tile(np.array([0.,2.02*h/eta]),4380)
    archetype=runner.site.model.CAVERN_CASES[-1]
    reference=runner.site.model.CavernCase(archetype.case_id,
        archetype.working_gas_million_sm3,archetype.peak_withdrawal_million_sm3_day,
        archetype.reference_site_cavern_count,service_reference_nh3_tph=1.)
    ok,status=runner.feasible(generation,np.ones(8760),
        {'electrolyser_mw':30.,'grid_connection_mw':30.},reference,
        2/3,5.,0.,ramp=.03)
    assert ok and status=='feasible_constructive_rigid_witness'
    assert runner.RESIDUALS[-1]<1e-5

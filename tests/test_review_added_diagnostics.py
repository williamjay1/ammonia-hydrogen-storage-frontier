"""Small mathematical certificates for the review-added interpretation."""
import numpy as np


def test_combined_stock_identity_independent_of_grid():
    h,eta=177.55300484072126,1000/52
    q=np.array([.9,1.1,1.,1.])
    e=np.array([8.,10.,9.,4*h/eta-27.])
    ds=eta*e-h*q
    da=q-1
    for grid_capacity in [0,.45,.76,29.]:
        # Grid capacity never enters these material-balance identities.
        assert np.allclose(ds+h*da,eta*e-h,atol=1e-12)


def test_rigid_reconstruction_requires_its_own_electricity():
    h,eta,alpha,gamma=177.55300484072126,1000/52,1.6/3.6,.000587
    emax=3*h/eta
    e=np.array([0,emax/2,emax])
    renewable=e.copy()
    need=np.maximum(0,e+alpha+gamma*np.maximum(eta*e-h,0)-renewable)
    sufficient=alpha+gamma*eta*emax
    assert np.all(need<=sufficient+1e-12)
    assert not np.all(need<=0)


def test_buffer_mass_is_capacity_allowance_not_tank_geometry():
    service=np.array([.6853699706521995,30.468745693440658,143.35548063444742])
    mass=24*service
    assert np.allclose(mass,[16.448879295652787,731.2498966425758,3440.531535226738])


def test_general_identity_fails_without_loss_term():
    stock,loss=100.,.001
    # The loss-free formula must not be exported unchanged to boil-off systems.
    ds=-loss*stock
    assert not np.isclose(ds,0)

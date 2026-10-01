from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_eho_capacity_deliverability import (
    H2_PER_T, cyclic_minimum_sum, integer_ceiling, service_rate_bounds,
)


def test_cyclic_minimum_sum_includes_end_to_start_interval():
    assert cyclic_minimum_sum(np.array([1., 5., 5., 1.]), 2) == 2.
    assert cyclic_minimum_sum(np.array([1., 5., 5., 1.]), 4) == 12.


def test_roundoff_does_not_add_a_module():
    assert integer_ceiling(2.0 + 1e-11) == 2
    assert integer_ceiling(2.001) == 3


def test_zero_buffer_enforces_service_cut_even_with_low_process_floor():
    supply = np.array([0., 2 * H2_PER_T, 0., 2 * H2_PER_T])
    low_floor = service_rate_bounds(
        supply, minimum_nh3_tph=.6, product_buffer_hours=0,
        service_scale_tph=1, working_gas_kg=H2_PER_T,
        withdrawal_kgph=.75 * H2_PER_T, wtir=1,
        free_inventory_kg_per_tph=H2_PER_T, lengths=(1, 2),
    )
    assert low_floor["exact_volume_module_lower_bound"] == 1
    assert low_floor["withdrawal_cut_module_lower_bound"] == 2
    assert low_floor["certified_deliverability_above_volume_bound"] == 1


def test_product_buffer_can_relax_the_rate_bound():
    supply = np.array([0., 2 * H2_PER_T, 0., 2 * H2_PER_T])
    result = service_rate_bounds(
        supply, minimum_nh3_tph=.6, product_buffer_hours=1,
        service_scale_tph=1, working_gas_kg=H2_PER_T,
        withdrawal_kgph=.75 * H2_PER_T, wtir=1,
        free_inventory_kg_per_tph=.6 * H2_PER_T, lengths=(1, 2),
    )
    assert result["withdrawal_cut_module_lower_bound"] == 1


def test_constant_supply_cannot_create_a_rate_requirement():
    result = service_rate_bounds(
        np.full(10, H2_PER_T), minimum_nh3_tph=.9,
        product_buffer_hours=24, service_scale_tph=50,
        working_gas_kg=1e6, withdrawal_kgph=1000, wtir=5,
        free_inventory_kg_per_tph=0, lengths=(1, 6),
    )
    assert result["combined_necessary_module_lower_bound"] == 0


def test_invalid_storage_rate_is_rejected():
    with pytest.raises(ValueError):
        service_rate_bounds(
            np.ones(10), minimum_nh3_tph=.9, product_buffer_hours=24,
            service_scale_tph=1, working_gas_kg=1e6, withdrawal_kgph=0,
            wtir=2, free_inventory_kg_per_tph=100, lengths=(1,),
        )

import numpy as np

from src.run_eu27_rfnbo_frontier import (
    BUFFER_HOURS,
    buffer_frontier_cases,
    local_year_index,
)


def test_buffer_grid_includes_explicit_flexible_zero_buffer_control():
    cases = buffer_frontier_cases()

    assert BUFFER_HOURS == (0.0, 24.0, 72.0, 168.0, 336.0)
    assert cases[0] == (0.0, "flexible_no_product_buffer")
    assert len(cases) == 5
    assert 2 * (1 + len(cases) + 3) == 18  # two rules, one rigid, five buffers, three targets


def test_local_year_index_and_matching_labels_follow_civil_calendar():
    timezone_name = "Europe/Berlin"
    index = local_year_index(2015, timezone_name)
    local = index.tz_convert(timezone_name)
    month_labels = np.asarray(local.strftime("%Y-%m"), dtype=str)

    assert len(index) == 8760
    assert index.tz is not None and str(index.tz) == "UTC"
    assert local[0].strftime("%Y-%m-%d %H:%M") == "2015-01-01 00:00"
    assert local[-1].strftime("%Y-%m-%d %H:%M") == "2015-12-31 23:00"
    assert len(np.unique(month_labels)) == 12
    assert month_labels[0] == "2015-01"
    assert month_labels[-1] == "2015-12"
    assert np.count_nonzero(month_labels == "2015-01") == 744
    assert np.count_nonzero(month_labels == "2015-03") == 743
    assert np.count_nonzero(month_labels == "2015-10") == 745
    assert np.count_nonzero(month_labels == "2015-12") == 744


def test_local_leap_year_index_contains_all_february_hours():
    timezone_name = "Europe/Berlin"
    index = local_year_index(2016, timezone_name)
    local = index.tz_convert(timezone_name)
    month_labels = np.asarray(local.strftime("%Y-%m"), dtype=str)

    assert len(index) == 8784
    assert len(np.unique(month_labels)) == 12
    assert np.count_nonzero(month_labels == "2016-02") == 696
    assert local[0].strftime("%Y-%m-%d %H:%M") == "2016-01-01 00:00"
    assert local[-1].strftime("%Y-%m-%d %H:%M") == "2016-12-31 23:00"

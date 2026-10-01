import pandas as pd

from src.run_eho_site_module_frontier import append_row


def test_append_row_expands_schema_without_ragged_csv_records(tmp_path):
    output = tmp_path / "diagnostic.csv"
    append_row(output, {"status": "infeasible", "message": "no feasible point"})
    append_row(output, {
        "status": "optimal",
        "message": "solved, then audited",
        "capacity": 12.5,
    })
    append_row(output, {"capacity": 8.0, "status": "optimal", "message": "ok"})

    frame = pd.read_csv(output)
    assert frame.columns.tolist() == ["status", "message", "capacity"]
    assert len(frame) == 3
    assert frame.loc[0, "status"] == "infeasible"
    assert pd.isna(frame.loc[0, "capacity"])
    assert frame.loc[1, "message"] == "solved, then audited"
    assert frame.loc[1, "capacity"] == 12.5
    assert frame.loc[2, "capacity"] == 8.0

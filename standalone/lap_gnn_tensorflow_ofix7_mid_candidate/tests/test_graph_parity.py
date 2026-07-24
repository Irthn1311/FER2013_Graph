import csv

from _helpers import ROOT


def test_graph_parity():
    report = ROOT.parents[1] / "outputs" / "d16_analysis" / "lap_gnn_tensorflow_port" / "08_graph_parity.csv"
    if report.is_file():
        rows = list(csv.DictReader(report.open(encoding="utf-8")))
        assert len(rows) == 32
        assert max(float(row["max_difference"]) for row in rows) == 0.0


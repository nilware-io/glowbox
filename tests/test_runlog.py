import numpy as np
import pyarrow.parquet as pq

from glowbox.analysis.runlog import RunLog, load_run, write_report
from glowbox.control.pulse import PulseSequence, pulsed_rabi
from glowbox.sim.interface import FluorescenceTrace


def test_runlog_round_trip(tmp_path):
    seq = pulsed_rabi(40, 2.87e9)
    counts = np.arange(2 * seq.n_bins, dtype=np.int64).reshape(2, -1)
    trace = FluorescenceTrace(counts, seq.grid_ns, counts.astype(float) * 1e8, {"source": "test"})
    with RunLog("unit test", config={"a": 1, "b": [1, 2]}, seed=7, output_root=tmp_path) as run:
        assert run.rng.integers(1 << 30) == np.random.default_rng(7).integers(1 << 30)
        run.log_trace(trace, seq, rabi_ns=40)
        run.log_trace(FluorescenceTrace(counts[:1], seq.grid_ns, None, {"source": "hw"}))
        run.log_metrics(0, loss=1.0)
        run.log_metrics(1, loss=0.5)
    info = load_run(run.dir)
    assert info["seed"] == 7 and info["config"] == {"a": 1, "b": [1, 2]}
    assert info["finished_utc"] is not None
    assert "device" in info["environment"] and "torch" in info["environment"]["versions"]
    tr = info["traces"]
    assert len(tr) == 3 and list(tr["trace_id"]) == [0, 0, 1]
    np.testing.assert_array_equal(np.stack(tr["counts"][:2]), counts)
    assert tr["rate_cps"][2] is None
    assert PulseSequence.from_dict(__import__("json").loads(tr["sequence"][0])) == seq
    assert b"glowbox_run" in pq.read_schema(run.dir / "traces.parquet").metadata
    assert list(info["metrics"]["loss"]) == [1.0, 0.5]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1])
    png = run.save_figure(fig, "line")
    report = write_report(run.dir, "Unit test", "It works.", figures=[png], report_dir=tmp_path / "results")
    text = report.read_text()
    assert "# Unit test" in text and "seed: 7" in text and "| loss | 0.5 |" in text
    assert (tmp_path / "results" / "figures" / f"{run.run_id}-line.png").exists()

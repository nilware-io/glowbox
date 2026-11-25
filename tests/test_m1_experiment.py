"""Smoke test of the milestone-1 script on a tiny configuration (both simulators, logging, plots, report)."""

from __future__ import annotations

import math

from glowbox.analysis.runlog import load_run
from glowbox.experiments.m1_odmr_rabi import DEFAULT_CONFIG, _merge, load_config, main


def test_m1_runs_end_to_end(tmp_path):
    cfg = _merge(
        load_config(DEFAULT_CONFIG, quick=True),
        {
            "n_detuning": 4,
            "odmr": {"step_hz": 8e6, "shots": 20},
            "rabi": {"max_ns": 200, "step_ns": 40, "shots": 50},
            "throughput": {"devices": ["cpu"], "n_sequences": 4, "shots": 10, "repeat": 1},
        },
    )
    report, summary = main(cfg, output_root=tmp_path / "out", report_dir=tmp_path / "results")
    text = report.read_text(encoding="utf-8")
    assert report.name == "m1-odmr-rabi.md" and "## Comparison" in text and "shots/s" in text
    assert (tmp_path / "results" / "figures" / "m1-odmr-rabi-odmr.png").exists()
    assert (tmp_path / "results" / "figures" / "m1-odmr-rabi-rabi.png").exists()
    assert summary["odmr_max_rel_rate_diff"] < 5e-4 and summary["rabi_max_rel_rate_diff"] < 5e-4
    assert math.isfinite(summary["shots_per_s_cpu"])
    (run_dir,) = (tmp_path / "out").iterdir()
    info = load_run(run_dir)
    assert info["seed"] == cfg["seed"] and info["config"]["n_detuning"] == 4
    tr = info["traces"]
    # every ODMR point + MW-off baseline and every Rabi duration, once per simulator
    n_odmr = summary["odmr_points"] + 1
    assert len(tr) == 2 * (n_odmr + 6)

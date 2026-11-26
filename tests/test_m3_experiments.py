"""Smoke tests of the milestone-3 scripts on tiny configurations (codebooks / series, readouts, quantum test, timing
probes, QuTiP reference check, plots, report). Marked slow (a few minutes on a CPU): run with ``pytest -m slow``."""

from __future__ import annotations

import math

import pytest

from glowbox.analysis.runlog import load_run
from glowbox.experiments import m3_bit_memory, m3_forecast
from glowbox.experiments.m1_odmr_rabi import _merge, load_config


@pytest.mark.slow
def test_m3_bit_memory_runs_end_to_end(tmp_path):
    cfg = _merge(
        load_config(m3_bit_memory.DEFAULT_CONFIG, quick=True),
        {
            "device": {"n_detuning": 4},
            "grid": {"bits": [1, 2], "delays_ns": [100, 1000], "shots": [10000]},
            "headline": {"bits": 2, "delay_ns": 1000, "shots": 10000, "read": "none"},
            "ridge": {"n_train": 256, "n_test": 256},
            "cnn": {"pairs": [[2, 1000]], "steps": 20, "eval_every": 10, "batch": 32, "n_val": 256},
            "reference": {"n_patterns": 2, "workers": 2},
        },
    )
    report, summary = m3_bit_memory.main(cfg, output_root=tmp_path / "out", report_dir=tmp_path / "results")
    text = report.read_text(encoding="utf-8")
    assert report.name == "m3-bit-memory.md"
    for section in ("## Quantum test", "## Recall vs delay", "## Reference check", "strategy_learns_timing"):
        assert section in text
    for fig in ("inputs", "recall-vs-bits", "recall-vs-delay", "quantum-test", "timing"):
        assert (tmp_path / "results" / "figures" / f"m3-bit-memory-{fig}.png").exists()
    assert summary["oracle_bit_acc_detuned"] < 0.6 and summary["no_line_margin_mhz"] > 100
    assert summary["ref_none_max_rel_template_diff"] < 1e-3 and summary["strategy_learns_timing"] == 0.0
    (run_dir,) = (tmp_path / "out").iterdir()
    info = load_run(run_dir)
    assert info["seed"] == cfg["seed"] and len(info["traces"]) == 2 * 2 * 10  # condition x pattern x shots
    assert (run_dir / "results.parquet").exists() and (run_dir / "quantum_test.json").exists()


@pytest.mark.slow
def test_m3_forecast_runs_end_to_end(tmp_path):
    cfg = _merge(
        load_config(m3_forecast.DEFAULT_CONFIG, quick=True),
        {
            "device": {"n_detuning": 4},
            "series": {"n_cycles": 12, "burn_in": 2},
            "data": {"n_train_seq": 16, "n_val_seq": 4, "n_test_seq": 4, "window": 4, "ar_draws": 1},
            "cnn": {"steps": 20, "eval_every": 10, "batch": 32},
            "reservoir": {"n_train": 64},
            "reference": {"n_series": 1, "n_reservoir": 2, "workers": 2},
        },
    )
    report, summary = m3_forecast.main(cfg, output_root=tmp_path / "out", report_dir=tmp_path / "results")
    text = report.read_text(encoding="utf-8")
    assert report.name == "m3-forecast.md"
    for section in ("## Quantum test and results", "## Reference check", "strategy_learns_timing"):
        assert section in text
    for fig in ("inputs", "quantum-test", "predictions", "timing"):
        assert (tmp_path / "results" / "figures" / f"m3-forecast-{fig}.png").exists()
    assert math.isfinite(summary["ar_skill_gap"]) and summary["no_line_margin_mhz"] > 100
    assert summary["ref_series_max_rel_cell_diff"] < 1e-3 and summary["strategy_learns_timing"] == 0.0
    (run_dir,) = (tmp_path / "out").iterdir()
    assert load_run(run_dir)["seed"] == cfg["seed"]
    assert (run_dir / "results.parquet").exists() and (run_dir / "timing.parquet").exists()

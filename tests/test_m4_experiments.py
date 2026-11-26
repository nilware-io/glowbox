"""Smoke tests of the milestone-4 scripts on tiny configurations (end-to-end training, banks, quantum test, timing
probes, QuTiP reference check, plots, report). Marked slow (minutes on a CPU): run with ``pytest -m slow``."""

from __future__ import annotations

import pytest

from glowbox.analysis.runlog import load_run
from glowbox.experiments import m4_autoencoder, m4_learned_sensor
from glowbox.experiments.m1_odmr_rabi import _merge, load_config


@pytest.mark.slow
def test_m4_autoencoder_runs_end_to_end(tmp_path):
    cfg = _merge(
        load_config(m4_autoencoder.DEFAULT_CONFIG, quick=True),
        {
            "data": {"n_train": 256, "n_val": 64, "n_test": 128},
            "train": {"steps": 6, "batch": 16, "val_every": 3},
            "bank_decoder": {"steps": 40, "batch": 32},
            "ideal": {"steps": 40},
            "grid": {"blocks": [2], "shots": [10000]},
            "timing": {"duration_shifts_ns": [-20, 0, 20], "retrain_shifts_ns": [20]},
            "reference": {"n_patterns": 1, "workers": 1},
        },
    )
    report, summary = m4_autoencoder.main(cfg, output_root=tmp_path / "out", report_dir=tmp_path / "results")
    text = report.read_text(encoding="utf-8")
    assert report.name == "m4-autoencoder.md"
    for section in ("## Quantum test", "## Learned strategies and timing", "## Checks", "strategy_learns_timing"):
        assert section in text
    for fig in ("reconstructions", "nmse", "quantum-test", "sequences", "timing"):
        assert (tmp_path / "results" / "figures" / f"m4-autoencoder-{fig}.png").exists()
    assert summary["learned_nmse_detuned"] > 0.95 and summary["no_line_margin_mhz"] > 100
    assert summary["ref_max_rel_diff"] < 1e-3
    (run_dir,) = (tmp_path / "out").iterdir()
    info = load_run(run_dir)
    assert info["seed"] == cfg["seed"] and len(info["traces"]) == 2 * 2 * 10  # condition x patch x shots
    for f in ("results.parquet", "timing.parquet", "areas.parquet", "quantum_test.json", "reference_check.json"):
        assert (run_dir / f).exists()


@pytest.mark.slow
def test_m4_learned_sensor_runs_end_to_end(tmp_path):
    cfg = _merge(
        load_config(m4_learned_sensor.DEFAULT_CONFIG, quick=True),
        {
            "bank": {"n_train": 64, "n_val": 16, "n_test": 16, "workers": 2},
            "protocol_train": {"steps": 2, "fields_per_step": 2},
            "train": {"steps": 20, "eval_every": 10},
            "eval": {"shots": [1000, 5000]},
            "timing": {"n_fields": 8, "duration_shifts_ns": [-50, 0, 50]},
            "reference": {"n_fields": 1, "workers": 2},
        },
    )
    report, summary = m4_learned_sensor.main(cfg, output_root=tmp_path / "out", report_dir=tmp_path / "results")
    text = report.read_text(encoding="utf-8")
    assert report.name == "m4-learned-sensor.md"
    for section in ("## Quantum test", "## What the protocol learned, and timing", "## Reference check"):
        assert section in text
    for fig in ("protocol", "error-vs-shots", "quantum-test", "training", "timing"):
        assert (tmp_path / "results" / "figures" / f"m4-learned-sensor-{fig}.png").exists()
    assert summary["no_line_margin_mhz"] > 100 and summary["ref_max_rel_feature_diff"] < 1e-3
    (run_dir,) = (tmp_path / "out").iterdir()
    info = load_run(run_dir)
    assert len(info["traces"]) == 2 * 2 * 10  # condition x point x shots
    for f in ("results.parquet", "timing.parquet", "protocols.parquet", "quantum_test.json"):
        assert (run_dir / f).exists()

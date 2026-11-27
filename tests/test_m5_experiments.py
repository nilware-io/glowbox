"""Smoke tests of the milestone-5 scripts on tiny configurations (posterior episodes, PPO, CNN, quantum test,
timing probes, QuTiP reference through run(), raw traces, plots, report). Marked slow (minutes on a CPU): run with
``pytest -m slow``."""

from __future__ import annotations

import pytest

from glowbox.analysis.runlog import load_run
from glowbox.experiments import m5_adaptive_field, m5_rl_bit_write
from glowbox.experiments.m1_odmr_rabi import _merge, load_config


@pytest.mark.slow
def test_m5_adaptive_field_runs_end_to_end(tmp_path):
    cfg = _merge(
        load_config(m5_adaptive_field.DEFAULT_CONFIG, quick=True),
        {"budget": {"n_steps": 6, "curve_steps": [3, 6]}, "data": {"n_test": 6, "n_pool": 24, "n_val": 8}},
    )
    report, summary = m5_adaptive_field.main(cfg, output_root=tmp_path / "out", report_dir=tmp_path / "results")
    text = report.read_text(encoding="utf-8")
    assert report.name == "m5-adaptive-field.md"
    for section in ("## Quantum test", "## Shots to accuracy", "## Timing", "## Emulator and reference check"):
        assert section in text
    for fig in ("shots-to-accuracy", "choices", "quantum-test", "rl-training", "timing"):
        assert (tmp_path / "results" / "figures" / f"m5-adaptive-field-{fig}.png").exists()
    assert summary["no_line_margin_mhz"] > 100
    (run_dir,) = [d for d in (tmp_path / "out").iterdir() if (d / "run.yaml").exists()]  # plus the table cache
    assert any((tmp_path / "out" / "cache").iterdir())
    info = load_run(run_dir)
    assert info["seed"] == cfg["seed"] and len(info["traces"]) == 2 * 2 * 3 * 10  # condition x policy x step x shots
    for f in ("curves.parquet", "timing.parquet", "rl_training.parquet", "quantum_test.json", "reference_check.json"):
        assert (run_dir / f).exists()


@pytest.mark.slow
def test_m5_rl_bit_write_runs_end_to_end(tmp_path):
    cfg = load_config(m5_rl_bit_write.DEFAULT_CONFIG, quick=True)
    report, summary = m5_rl_bit_write.main(cfg, output_root=tmp_path / "out", report_dir=tmp_path / "results")
    text = report.read_text(encoding="utf-8")
    assert report.name == "m5-rl-bit-write.md"
    for section in ("## Quantum test", "## Recall vs shots", "## Timing", "## Reference check"):
        assert section in text
    for fig in ("recall", "training", "codebooks", "timing"):
        assert (tmp_path / "results" / "figures" / f"m5-rl-bit-write-{fig}.png").exists()
    assert summary["gradient_bits_detuned"] < 0.1
    (run_dir,) = (tmp_path / "out").iterdir()
    info = load_run(run_dir)
    assert len(info["traces"]) == 2 * 2 * 2 * 10  # condition x write x pattern x shots
    for f in ("recall.parquet", "timing.parquet", "areas.parquet", "quantum_test.json", "reference_check.json"):
        assert (run_dir / f).exists()

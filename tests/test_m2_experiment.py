"""Smoke test of the milestone-2 script on a tiny configuration (banks, training, quantum test, fit, timing probe,
reference check, plots, report). Marked slow (~1-2 min CPU): run with ``pytest -m slow``."""

from __future__ import annotations

import math

import pytest

from glowbox.analysis.runlog import load_run
from glowbox.experiments.m2_field_estimation import DEFAULT_CONFIG, _merge, load_config, main


@pytest.mark.slow
def test_m2_runs_end_to_end(tmp_path):
    cfg = _merge(
        load_config(DEFAULT_CONFIG, quick=True),
        {
            "bank": {"n_train": 32, "n_val": 16, "n_test": 16, "n_detuning": 4, "fields_per_env": 8, "workers": 2},
            "sweep": {"step_hz": 4e6, "pi_ns": 310},
            "train": {"steps": 20, "eval_every": 10, "batch": 16},
            "linear_train": {"steps": 20},
            "eval": {"shots": [100, 1000]},
            "fit": {"n_fields": 4, "workers": 2},
            "timing": {"n_fields": 8, "pi_shifts_ns": [-100, 0, 100], "noise_repeats": 1},
            "reference": {"n_fields": 1, "cw_ns": 3000, "workers": 2},
        },
    )
    report, summary = main(cfg, output_root=tmp_path / "out", report_dir=tmp_path / "results")
    text = report.read_text(encoding="utf-8")
    assert report.name == "m2-field-estimation.md"
    for section in ("## Quantum test", "## Calibration", "## Reference check", "uses_us_timing"):
        assert section in text
    for fig in ("inputs", "error-vs-shots", "calibration", "quantum-test", "timing"):
        assert (tmp_path / "results" / "figures" / f"m2-field-estimation-{fig}.png").exists()
    assert math.isfinite(summary["cw_cnn_gap_ut"]) and summary["no_line_margin_mhz"] > 100
    assert summary["pulsed_ref_max_rel_feature_diff"] < 1e-2
    (run_dir,) = (tmp_path / "out").iterdir()
    info = load_run(run_dir)
    assert info["seed"] == cfg["seed"] and len(info["traces"]) == 2 * 2 * 2 * 10  # cond x protocol x point x shots
    assert (run_dir / "results.parquet").exists() and (run_dir / "quantum_test.json").exists()

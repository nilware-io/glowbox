"""Quantum-test harness: conditions, detuning helpers, t and bootstrap CIs, paired gaps, timing probes. CPU, < 5 s."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from glowbox.analysis.quantum_test import (
    RESONANT,
    Condition,
    bootstrap_ci,
    check_no_line_driven,
    detune_sequence,
    detuned_condition,
    mw_timing_structure,
    rabi_band_fraction,
    run_quantum_test,
    t_ci,
    time_saliency,
    timing_sensitivity,
)
from glowbox.control.pulse import cw_odmr_point, hahn_echo, pulsed_rabi
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate import ModelOptions, SurrogateSimulator


def test_detune_sequence_shifts_only_frequencies():
    seq = hahn_echo(500, 2.80e9, 100)
    d = detune_sequence(seq, 4e8)
    assert [m.frequency_hz for m in d.microwave] == [m.frequency_hz + 4e8 for m in seq.microwave]
    assert [(m.start_ns, m.duration_ns, m.phase_rad) for m in d.microwave] == [
        (m.start_ns, m.duration_ns, m.phase_rad) for m in seq.microwave
    ]
    assert d.laser == seq.laser and d.label["mw_offset_hz"] == 4e8


def test_detuned_sequence_is_dark_to_the_spins():
    """Physics check of the condition: a detuned pi pulse gives the MW-off trace (docs/physics.md section 9.5)."""
    env = EnvParams(b_field_t=(0.0, 0.0, 3e-3))
    f_line = 2.87e9 - 2.803e10 * 3e-3 / np.sqrt(3.0)
    # float64: in float32 a frame 400 MHz from the lines costs ~3e-4 relative round-off over 9 us
    sim = SurrogateSimulator("cpu", ModelOptions(n_detuning=4, dtype=torch.float64))
    seq = pulsed_rabi(60, f_line, 1.0)
    on, det, off = (
        sim.expected_rates([seq], env)[0],
        sim.expected_rates([detune_sequence(seq, 4e8)], env)[0],
        sim.expected_rates([dataclasses.replace(seq, microwave=())], env)[0],
    )
    assert np.max(np.abs(on - off)) / off.max() > 0.01
    assert np.max(np.abs(det - off)) / off.max() < 1e-4
    ok, margin = check_no_line_driven(np.array([f_line + 4e8]), np.array(env.b_field_t), env)
    assert ok and margin > 2.9e8  # nearest is the f+ line, 97 MHz above f-
    assert not check_no_line_driven(np.array([f_line + 1e6]), np.array(env.b_field_t), env)[0]


def test_bootstrap_ci_covers_and_orders():
    rng = np.random.default_rng(0)
    v, lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], rng=rng)
    assert v == 3.0 and lo < v < hi and lo >= 1.0 and hi <= 5.0
    assert bootstrap_ci([2.0, 2.0, 2.0, 2.0, 2.0]) == (2.0, 2.0, 2.0)
    med = bootstrap_ci([1.0, 2.0, 100.0], stat=np.median)
    assert med[0] == 2.0


def test_t_ci_covers_at_five_seeds_and_bootstrap_does_not():
    # 2000 draws of 5 N(0, 1) values: the t interval of the mean covers the true 0 in ~95 % of draws, the percentile
    # bootstrap only in ~83 % (why the quantum test reports t intervals over seeds).
    rng = np.random.default_rng(42)
    draws = rng.normal(size=(2000, 5))
    t_cov = np.mean([lo < 0.0 < hi for _, lo, hi in (t_ci(d) for d in draws)])
    b_cov = np.mean([lo < 0.0 < hi for _, lo, hi in (bootstrap_ci(d, n_boot=500, rng=rng) for d in draws[:500])])
    assert 0.935 < t_cov < 0.965
    assert b_cov < 0.88
    m, lo, hi = t_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert m == 3.0 and m - lo == pytest.approx(hi - m)
    assert hi - m == pytest.approx(2.776445 * np.std([1, 2, 3, 4, 5], ddof=1) / np.sqrt(5), rel=1e-5)


def test_run_quantum_test_pairs_seeds_and_signs_gaps():
    calls = []

    def task(cond: Condition, seed: int) -> dict[str, float]:
        calls.append((cond.name, seed))
        err = 1.0 + 0.01 * seed if not cond.detuned else 10.0 + 0.01 * seed
        return {"err": err, "skill": 1.0 - err / 10.0}

    res = run_quantum_test(task, range(5), higher_is_better={"skill": True})
    assert calls[:2] == [("resonant", 0), ("detuned", 0)]  # seed-major: a pair runs together
    np.testing.assert_allclose(res.gap_per_seed("err"), 9.0)
    np.testing.assert_allclose(res.gap_per_seed("skill"), 0.9)
    assert res.gap_significant("err") and res.gap_significant("skill")
    s = res.summary("err")
    assert s["resonant"][0] == pytest.approx(1.02) and s["gap"][0] == pytest.approx(9.0)
    md = res.to_markdown()
    assert "| err |" in md and "gap" in md
    with pytest.raises(ValueError):
        run_quantum_test(task, range(4))


def test_no_gap_is_not_significant():
    rng = np.random.default_rng(1)
    noise = {(c, s): rng.normal() for c in ("resonant", "detuned") for s in range(8)}
    res = run_quantum_test(lambda cond, s: {"err": 5.0 + noise[(cond.name, s)]}, range(8))
    _, lo, hi = res.summary("err")["gap"]
    assert lo < 0 < hi and not res.gap_significant("err")


def test_conditions():
    assert not RESONANT.detuned and detuned_condition().detuned
    assert detuned_condition(1e9).mw_offset_hz == 1e9


def test_timing_structure_and_spectrum():
    env = EnvParams()
    f_r = env.rabi_full_scale_hz * np.sqrt(2 / 3)
    pi_ns = round(1e9 / (2 * f_r) / 10) * 10
    seq = hahn_echo(800, 2.87e9, 2 * (pi_ns // 2))
    st = mw_timing_structure(seq, env)
    areas = [s["area_pi"] for s in st["segments"]]
    assert areas[1] == pytest.approx(1.0, abs=0.1) and areas[0] == pytest.approx(0.5, abs=0.1)
    assert st["gaps_ns"] == [800, 800] and not st["cw_like"]
    assert mw_timing_structure(cw_odmr_point(2.87e9, 2000), env)["cw_like"]
    # a drive switched on and off at the Rabi period has its envelope power there; CW has none
    assert rabi_band_fraction(cw_odmr_point(2.87e9, 2000), f_r) == 0.0
    assert rabi_band_fraction(seq, 1e6 / 1.6) > rabi_band_fraction(seq, 50e6)


def test_timing_sensitivity_and_saliency():
    ts = timing_sensitivity(lambda s: 1.0 + abs(s) / 100, [-100, 0, 100])
    np.testing.assert_allclose(ts.relative, [1.0, 0.0, 1.0])

    class OnlyFirstBins(torch.nn.Module):
        def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            m = x[:, :3].sum(dim=(1, 2))[:, None]
            return m, torch.zeros_like(m)

    sal = time_saliency(OnlyFirstBins(), torch.randn(4, 10, 5), torch.zeros(4), lambda m: m[:, 0])
    np.testing.assert_allclose(sal, [5, 5, 5, 0, 0, 0, 0, 0, 0, 0])

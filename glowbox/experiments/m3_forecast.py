"""Milestone 3, Task 3: predict the next fluorescence reading of the driven NV ensemble, with the quantum test.

    uv run python -m glowbox.experiments.m3_forecast [--config PATH] [--quick] [--report-dir DIR]

The system (glowbox.learn.forecast.DrivenSeries) is driven by a hidden, correlated random MW amplitude, one pulse
per cycle, each followed by a short readout; the readings form a time series. Predictors of the next reading from
the last ``window`` readings: the mean, persistence, linear AR (ridge), a CNN decoder on the time-resolved
readouts, and the NV reservoir (the past readings encoded by a fixed random pulse encoder into a second sequence on
the same diamond, ridge on its readout; also with the raw readings appended). Upper references: the noise ceiling
and the hidden-state oracle (Bayes-optimal given the true drive history and spin state).

The quantum test runs everything per seed twice: as designed, and with every MW frequency (the system's drive and
the reservoir's encoding) shifted by ``detuned_offset_hz``. In the resonant arm the reservoir is also run with only
its encoding detuned (the spins of the reservoir no longer respond while the series still carries information).
Timing: the reservoir's MW durations shifted (frozen and retrained readout), the CNN's saliency over readout bins
and past cycles. Nothing that sets timing is learned (``strategy_learns_timing`` False).

Outputs: ``outputs/<run-id>/`` (run.yaml, raw example traces, metrics, predictions, plots) and a report
(``outputs/<run-id>/report/`` unless ``--report-dir``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

from glowbox.analysis.quantum_test import (
    RESONANT,
    Condition,
    QuantumTestResult,
    check_no_line_driven,
    detuned_condition,
    run_quantum_test,
    t_ci,
)
from glowbox.analysis.runlog import RunLog, write_report
from glowbox.experiments.m1_odmr_rabi import _merge, load_config
from glowbox.experiments.m2_field_estimation import _style
from glowbox.experiments.m3_bit_memory import _band, _ci, _tgen, device_env, input_saliency, paired_diff
from glowbox.learn.bit_memory import device_lines_hz
from glowbox.learn.decoder import ConvDecoder, DecoderConfig, TrainConfig, predict, train_decoder
from glowbox.learn.forecast import DrivenSeries, gather_windows, noise_ceiling, skill, windows
from glowbox.learn.reservoir import (
    EncoderSpec,
    RandomPulseEncoder,
    RidgeReadout,
    compose_suffix,
    full_sequence,
    init_prefix,
    simulate_suffixes,
)
from glowbox.sim.interface import EnvParams
from glowbox.sim.surrogate import SurrogateSimulator
from glowbox.sim.surrogate.model import ModelOptions

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "m3_forecast.yaml"
REPORT_NAME = "m3-forecast"
PREDICTORS = ("persistence", "ar", "cnn", "reservoir", "reservoir_direct", "reservoir_offres")
COLORS = {
    "persistence": "#8a8985",
    "ar": "#eb6834",
    "cnn": "#2a78d6",
    "reservoir": "#a35fd0",
    "reservoir_direct": "#d6639b",
    "reservoir_offres": "#c3c2b7",
    "oracle": "#1baf7a",
    "ceiling": "#52514e",
}
LABELS = {
    "persistence": "persistence",
    "ar": "linear AR",
    "cnn": "CNN decoder",
    "reservoir": "NV reservoir",
    "reservoir_direct": "NV reservoir + direct input",
    "reservoir_offres": "NV reservoir, encoding detuned",
    "oracle": "hidden-state oracle",
    "ceiling": "noise ceiling",
}

__all__ = ["DEFAULT_CONFIG", "ForecastTask", "_merge", "load_config", "main"]


def condition_for(cfg: dict[str, Any], name: str) -> Condition:
    return RESONANT if name == "resonant" else detuned_condition(float(cfg["detuned_offset_hz"]))


@dataclasses.dataclass
class SeedData:
    """One seed and condition: noiseless cell means per shot (n_seq, K, n_rb) per split, drives, test noise."""

    env: EnvParams
    line_hz: float
    amps: dict[str, np.ndarray]
    means: dict[str, np.ndarray]
    test_cells: np.ndarray  # noisy test cells (summed over shots)
    y_mean: float
    y_std: float
    cell_mean: np.ndarray  # (n_rb,) mean training counts per cell (summed over shots)


class ForecastTask:
    """The quantum-test task function plus everything recorded along the way."""

    def __init__(self, cfg: dict[str, Any], run: RunLog, device: torch.device) -> None:
        self.cfg = cfg
        self.run = run
        self.device = device
        self.base_seed = int(cfg["seed"])
        self.options = ModelOptions(n_detuning=int(cfg["device"]["n_detuning"]))
        self.series = DrivenSeries(**cfg["series"])
        d = cfg["data"]
        self.window = int(d["window"])
        self.shots = float(d["shots"])
        self.rows: list[dict[str, Any]] = []
        self.timing_rows: list[dict[str, Any]] = []
        self.encoding: list[dict[str, Any]] = []
        self.saliency: dict[str, Any] = {}
        self.examples: dict[str, dict[str, Any]] = {}
        self.models: dict[str, torch.nn.Module] = {}
        self.ridges: dict[str, RidgeReadout] = {}
        self.data0: dict[str, SeedData] = {}
        self.histories: dict[str, Any] = {}
        self.no_line_margin_hz = math.inf
        self.seconds: dict[str, float] = {"series": 0.0, "reservoir": 0.0, "cnn": 0.0, "oracle": 0.0}

    # --- data -----------------------------------------------------------------------------------------------------

    def data(self, cond: Condition, seed: int) -> tuple[SeedData, torch.Tensor]:
        """Simulate the seed's series (paired across conditions: same drives and noise seeds)."""
        env = device_env(self.cfg, seed, self.base_seed)
        line = float(device_lines_hz(env)[self.series.line_index])
        d = self.cfg["data"]
        rng = np.random.default_rng([self.base_seed, seed, 2])
        sizes = {"train": int(d["n_train_seq"]), "val": int(d["n_val_seq"]), "test": int(d["n_test_seq"])}
        amps = {k: self.series.drives(n, rng) for k, n in sizes.items()}
        t0 = time.perf_counter()
        means: dict[str, np.ndarray] = {}
        for k in ("train", "val"):
            means[k] = self.series.simulate(
                amps[k], env, line, offset_hz=cond.mw_offset_hz, options=self.options, device=self.device
            )
        means["test"], states = self.series.states_after_cycles(
            amps["test"], env, line, offset_hz=cond.mw_offset_hz, options=self.options, device=self.device
        )
        self.seconds["series"] += time.perf_counter() - t0
        noise = np.random.default_rng([self.base_seed, seed, 3])
        test_cells = noise.poisson(means["test"] * self.shots).astype(np.float64)
        train_y = np.random.default_rng([self.base_seed, seed, 4]).poisson(means["train"] * self.shots).sum(-1)
        sd = SeedData(
            env=env,
            line_hz=line,
            amps=amps,
            means=means,
            test_cells=test_cells,
            y_mean=float(train_y.mean()),
            y_std=float(train_y.std()),
            cell_mean=means["train"].mean(axis=(0, 1)) * self.shots,
        )
        return sd, states

    def x_features(self, sd: SeedData, cells: np.ndarray) -> np.ndarray:
        """CNN input from cells (n, window, n_rb) summed over shots: (n, n_rb, window), relative to the cell mean."""
        out: np.ndarray = 10.0 * (cells / sd.cell_mean - 1.0)
        return np.ascontiguousarray(out.transpose(0, 2, 1))

    # --- predictors -----------------------------------------------------------------------------------------------

    def train_cnn(self, sd: SeedData, seed: int, tag: str) -> torch.nn.Module:
        c, dc = self.cfg["cnn"], self.cfg["decoder"]
        dcfg = DecoderConfig(
            mix_channels=int(dc["mix_channels"]),
            channels=tuple(int(x) for x in dc["channels"]),
            kernel=int(dc["kernel"]),
            hidden=int(dc["hidden"]),
        )
        torch.manual_seed(int(np.random.default_rng([self.base_seed, seed, 7]).integers(2**31)))
        nrb, w = self.series.n_rebinned, self.window
        model = ConvDecoder(nrb, w, 1, dcfg).to(self.device)
        dev = self.device
        mean_t = torch.as_tensor(sd.means["train"] * self.shots, dtype=torch.float32, device=dev)
        mean_v = torch.as_tensor(sd.means["val"] * self.shots, dtype=torch.float32, device=dev)
        cm = torch.as_tensor(sd.cell_mean, dtype=torch.float32, device=dev)
        seq_t, tg_t = windows(mean_t.shape[0], self.series, w)
        seq_v, tg_v = windows(mean_v.shape[0], self.series, w)
        lags = torch.arange(-w, 0, device=dev)

        def batch(
            m: torch.Tensor, seq: np.ndarray, tg: np.ndarray, n: int, gen: torch.Generator
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            pick = torch.randint(0, seq.shape[0], (n,), generator=gen, device=dev)
            s = torch.as_tensor(seq, device=dev)[pick]
            t = torch.as_tensor(tg, device=dev)[pick]
            cells = torch.poisson(m[s[:, None], t[:, None] + lags[None]], generator=gen)  # (n, w, nrb)
            target = torch.poisson(m[s, t], generator=gen).sum(-1)
            x = (10.0 * (cells / cm - 1.0)).transpose(1, 2)
            y = ((target - sd.y_mean) / sd.y_std)[:, None]
            return x, torch.zeros(n, device=dev), y

        g = _tgen(dev, self.base_seed, seed, 8)
        gv = _tgen(dev, self.base_seed, seed, 9)
        val = [batch(mean_v, seq_v, tg_v, 512, gv) for _ in range(4)]
        tc = TrainConfig(
            steps=int(c["steps"]),
            batch=int(c["batch"]),
            lr=float(c["lr"]),
            weight_decay=float(c["weight_decay"]),
            eval_every=int(c["eval_every"]),
        )
        t0 = time.perf_counter()
        hist = train_decoder(model, lambda _s: batch(mean_t, seq_t, tg_t, tc.batch, g), val, tc)
        self.seconds["cnn"] += time.perf_counter() - t0
        self.histories[tag] = dataclasses.asdict(hist)
        return model

    def reservoir_features(
        self,
        sd: SeedData,
        enc: RandomPulseEncoder,
        u: np.ndarray,
        offset_hz: float,
        seed: int,
        key: int,
        shift: int = 0,
    ) -> np.ndarray:
        """Counts per shot of the reservoir readout for inputs ``u`` (n, window), Poisson at the reservoir shots."""
        rc = self.cfg["reservoir"]
        block = enc.block(u, offset_hz=offset_hz, duration_shift_ns=shift)
        suf = compose_suffix(u.shape[0], block, int(rc["delay_ns"]), int(rc["readout_ns"]))
        pre = init_prefix(int(rc["init_ns"]), int(rc["settle_ns"]))
        t0 = time.perf_counter()
        m = simulate_suffixes(
            pre, suf, sd.env, int(rc["readout_ns"]) // 10, options=self.options, device=self.device, chunk=2048
        )
        self.seconds["reservoir"] += time.perf_counter() - t0
        s = float(rc["shots"])
        out: np.ndarray = np.random.default_rng([self.base_seed, seed, 5, key]).poisson(m * s) / s
        return out

    def encoder(self, sd: SeedData, seed: int) -> RandomPulseEncoder:
        spec = EncoderSpec(n_inputs=self.window, **self.cfg["reservoir"]["encoder"])
        return RandomPulseEncoder(spec, device_lines_hz(sd.env), 1000 * seed + 77)

    # --- the task function ----------------------------------------------------------------------------------------

    def __call__(self, cond: Condition, seed: int) -> dict[str, float]:
        print(f"[m3-forecast] {cond.name} seed {seed}", flush=True)
        sd, states = self.data(cond, seed)
        w = self.window
        enc = self.encoder(sd, seed)
        if cond.detuned:
            freqs = np.concatenate([enc.base_hz + s * enc.spec.freq_span_hz for s in (-1, 1)] + [[sd.line_hz]])
            ok, margin = check_no_line_driven(freqs + cond.mw_offset_hz, np.asarray(sd.env.b_field_t), sd.env)
            if not ok:
                raise RuntimeError(f"detuned condition drives a line (margin {margin / 1e6:.1f} MHz)")
            self.no_line_margin_hz = min(self.no_line_margin_hz, margin)
        # test windows (fixed noisy realisation shared by every predictor)
        y_test = sd.test_cells.sum(-1)
        seq_te, tg_te = windows(y_test.shape[0], self.series, w)
        target = y_test[seq_te, tg_te]
        past = gather_windows(y_test, seq_te, tg_te, w)
        mean_next = sd.means["test"].sum(-1)[seq_te, tg_te] * self.shots
        preds: dict[str, np.ndarray] = {"persistence": past[:, -1]}
        # AR on noisy training draws
        means_tr = sd.means["train"]
        seq_tr, tg_tr = windows(means_tr.shape[0], self.series, w)
        rng_ar = np.random.default_rng([self.base_seed, seed, 6])
        xs, ys = [], []
        for _ in range(int(self.cfg["data"]["ar_draws"])):
            y_tr = rng_ar.poisson(means_tr * self.shots).sum(-1).astype(np.float64)
            xs.append(gather_windows(y_tr, seq_tr, tg_tr, w))
            ys.append(y_tr[seq_tr, tg_tr])
        x_ar, y_ar = np.concatenate(xs), np.concatenate(ys)
        ar = RidgeReadout().fit(x_ar, y_ar)
        preds["ar"] = ar.predict(past)
        # CNN
        model = self.train_cnn(sd, seed, f"{cond.name}/{seed}")
        cells_te = gather_windows(sd.test_cells, seq_te, tg_te, w)
        x = torch.as_tensor(self.x_features(sd, cells_te), dtype=torch.float32, device=self.device)
        mu, sig = predict(model, x, torch.zeros(x.shape[0], device=self.device))
        preds["cnn"] = mu[:, 0].cpu().numpy() * sd.y_std + sd.y_mean
        z = (target - preds["cnn"]) / (sig[:, 0].cpu().numpy() * sd.y_std)
        # NV reservoir (one noise draw of the training windows, subsampled)
        n_res = min(int(self.cfg["reservoir"]["n_train"]), x_ar.shape[0] // len(xs))
        pick = np.random.default_rng([self.base_seed, seed, 10]).choice(x_ar.shape[0] // len(xs), n_res, replace=False)
        u_tr = (x_ar[pick] - sd.y_mean) / sd.y_std
        u_te = (past - sd.y_mean) / sd.y_std
        f_tr = self.reservoir_features(sd, enc, u_tr, cond.mw_offset_hz, seed, 0)
        f_te = self.reservoir_features(sd, enc, u_te, cond.mw_offset_hz, seed, 1)
        res = RidgeReadout().fit(f_tr, y_ar[pick])
        preds["reservoir"] = res.predict(f_te)
        resd = RidgeReadout().fit(np.concatenate([f_tr, u_tr], 1), y_ar[pick])
        preds["reservoir_direct"] = resd.predict(np.concatenate([f_te, u_te], 1))
        if cond.detuned:
            preds["reservoir_offres"] = preds["reservoir"]  # the encoding is already detuned
        else:
            off = float(self.cfg["detuned_offset_hz"])
            g_tr = self.reservoir_features(sd, enc, u_tr, off, seed, 2)
            g_te = self.reservoir_features(sd, enc, u_te, off, seed, 3)
            preds["reservoir_offres"] = RidgeReadout().fit(g_tr, y_ar[pick]).predict(g_te)
        # hidden-state oracle
        t0 = time.perf_counter()
        zlat = self.series.latent(sd.amps["test"])
        oracle = np.empty(seq_te.shape[0])
        for i in range(0, seq_te.shape[0], 4096):
            s, t = seq_te[i : i + 4096], tg_te[i : i + 4096]
            st = states[torch.as_tensor(s, device=states.device), torch.as_tensor(t - 1, device=states.device)]
            oracle[i : i + 4096] = self.series.oracle_next_mean(
                zlat[s, t - 1],
                st,
                sd.env,
                sd.line_hz,
                offset_hz=cond.mw_offset_hz,
                n_quad=int(self.cfg["oracle"]["n_quad"]),
                options=self.options,
                device=self.device,
            )
        self.seconds["oracle"] += time.perf_counter() - t0
        preds["oracle"] = oracle * self.shots
        out = {f"{k}_skill": skill(v, target) for k, v in preds.items()}
        out["ceiling_skill"] = noise_ceiling(mean_next, target)
        out["cnn_z_rms"] = float(np.sqrt(np.mean(z**2)))
        out["signal_rel_sd"] = float(np.std(mean_next) / np.mean(mean_next))
        for k, v in out.items():
            self.rows.append({"condition": cond.name, "seed": seed, "metric": k, "value": v})
        if seed == 0:
            self.data0[cond.name] = sd
            sel = seq_te == 0
            self.examples[cond.name] = {
                "cycle": tg_te[sel].tolist(),
                "target": target[sel].tolist(),
                "noiseless": mean_next[sel].tolist(),
                **{k: v[sel].tolist() for k, v in preds.items()},
            }
            if cond.name == "resonant":
                self.models["cnn"] = model
                self.ridges["ar"] = ar
                self.ridges["reservoir"] = res
                self.saliency_probe(sd, model, x)
        if cond.name == "resonant":
            self.encoding_stats(sd, enc, u_te, seed)
            self.timing_probe(sd, enc, (u_tr, f_tr), (u_te, f_te), y_ar[pick], target, res, seed)
        return {k: v for k, v in out.items() if k not in ("signal_rel_sd", "cnn_z_rms")}

    # --- timing ---------------------------------------------------------------------------------------------------

    def timing_probe(
        self,
        sd: SeedData,
        enc: RandomPulseEncoder,
        train: tuple[np.ndarray, np.ndarray],
        test: tuple[np.ndarray, np.ndarray],
        y_tr: np.ndarray,
        target: np.ndarray,
        frozen: RidgeReadout,
        seed: int,
    ) -> None:
        """Every reservoir MW duration shifted: skill of the frozen (shift-0) ridge and of a retrained one."""
        (u_tr, f_tr0), (u_te, f_te0) = train, test
        for shift in [int(s) for s in self.cfg["timing"]["duration_shifts_ns"]]:
            if shift == 0:
                f_tr, f_te = f_tr0, f_te0
            else:
                f_te = self.reservoir_features(sd, enc, u_te, 0.0, seed, 1000 + shift, shift)
                f_tr = self.reservoir_features(sd, enc, u_tr, 0.0, seed, 2000 + shift, shift)
            retr = RidgeReadout().fit(f_tr, y_tr)
            for name, pred in (("frozen", frozen.predict(f_te)), ("retrained", retr.predict(f_te))):
                self.timing_rows.append(
                    {"seed": seed, "shift_ns": shift, "readout": name, "skill": skill(pred, target)}
                )

    def encoding_stats(self, sd: SeedData, enc: RandomPulseEncoder, u: np.ndarray, seed: int) -> None:
        p = enc.params(u[:2048])
        f_r = sd.env.rabi_full_scale_hz * math.sqrt(2.0 / 3.0)
        area = 2.0 * f_r * p.amplitude * p.duration_ns * 1e-9
        drive_area = 2.0 * f_r * sd.amps["test"] * self.series.mw_ns * 1e-9
        self.encoding.append(
            {
                "seed": seed,
                "median_area_pi": float(np.median(area)),
                "frac_area_quarter_to_4pi": float(np.mean((area >= 0.25) & (area <= 4.0))),
                "drive_area_pi_p5": float(np.percentile(drive_area, 5)),
                "drive_area_pi_p95": float(np.percentile(drive_area, 95)),
                "t2_star_us": sd.env.t2_star_s * 1e6,
            }
        )

    def saliency_probe(self, sd: SeedData, model: torch.nn.Module, x: torch.Tensor) -> None:
        sal = input_saliency(model, x[:2048], torch.zeros(min(2048, x.shape[0]), device=self.device), lambda m: m[:, 0])
        per_bin, per_lag = sal.sum(axis=1), sal.sum(axis=0)
        m = sd.means["test"][:, self.series.burn_in :]
        contrast = m.std(axis=(0, 1)) / m.mean(axis=(0, 1))
        self.saliency = {
            "per_bin": per_bin.tolist(),
            "per_lag": per_lag.tolist(),  # oldest first
            "contrast": contrast.tolist(),
            "corr_bin_contrast": float(np.corrcoef(per_bin, contrast)[0, 1]),
            "last_lag_fraction": float(per_lag[-1] / per_lag.sum()),
        }


# --- QuTiP reference check ----------------------------------------------------------------------------------------

_REF: dict[str, Any] = {}


def _ref_init(n_detuning: int) -> None:
    from glowbox.sim.reference import ReferenceSimulator

    _REF["sim"] = ReferenceSimulator(n_detuning=n_detuning, detuning_rule="quantile")


def _ref_rate(item: tuple[dict[str, Any], dict[str, Any]]) -> np.ndarray:
    from glowbox.control.pulse import PulseSequence

    seq, env = PulseSequence.from_dict(item[0]), EnvParams.from_dict(item[1])
    return np.asarray(_REF["sim"].expected_rate_cps(seq, env)) * seq.grid_ns * 1e-9


def run_reference_check(task: ForecastTask) -> dict[str, Any]:
    """Seed-0 resonant: system sequences and reservoir sequences on the QuTiP reference."""
    rc = task.cfg["reference"]
    sd = task.data0["resonant"]
    ser = task.series
    n_s, n_r = int(rc["n_series"]), int(rc["n_reservoir"])
    out: dict[str, Any] = {}
    t0 = time.perf_counter()
    arr = ser.arrays(sd.amps["test"][:n_s], sd.line_hz)
    seqs = [row.to_sequence() for row in arr.unbatch()]
    # reservoir: the first test windows' inputs
    w = task.window
    y_test = sd.test_cells.sum(-1)
    seq_te, tg_te = windows(y_test.shape[0], ser, w)
    past = gather_windows(y_test, seq_te, tg_te, w)[:n_r]
    u = (past - sd.y_mean) / sd.y_std
    res_cfg = task.cfg["reservoir"]
    enc = task.encoder(sd, 0)
    suf = compose_suffix(n_r, enc.block(u), int(res_cfg["delay_ns"]), int(res_cfg["readout_ns"]))
    pre = init_prefix(int(res_cfg["init_ns"]), int(res_cfg["settle_ns"]))
    rseqs = [full_sequence(pre, row) for row in suf.unbatch()]
    ctx = mp.get_context("spawn")
    with ctx.Pool(int(rc["workers"]), initializer=_ref_init, initargs=(task.options.n_detuning,)) as pool:
        items = [(s.to_dict(), sd.env.to_dict()) for s in seqs + rseqs]
        rates = pool.map(_ref_rate, items, chunksize=1)
    idx = ser.readout_index()
    ref_cells = np.stack([r[idx].reshape(ser.total_cycles, ser.n_rebinned, ser.rebin).sum(-1) for r in rates[:n_s]])
    sur_cells = sd.means["test"][:n_s]
    out["series_max_rel_cell_diff"] = float(np.max(np.abs(ref_cells - sur_cells)) / np.max(sur_cells))
    dev = sur_cells.sum(-1) - sur_cells.sum(-1).mean()
    out["series_max_reading_diff_over_signal"] = float(
        np.max(np.abs(ref_cells.sum(-1) - sur_cells.sum(-1))) / max(np.max(np.abs(dev)), 1e-30)
    )
    # predictions on noiseless readings (scaled to the shots), ref vs surrogate
    sq, tg = windows(n_s, ser, w)
    for name, cells in (("ref", ref_cells), ("sur", sur_cells)):
        y = cells.sum(-1) * task.shots
        out[f"ar_pred_{name}"] = task.ridges["ar"].predict(gather_windows(y, sq, tg, w)).tolist()
        x = task.x_features(sd, gather_windows(cells * task.shots, sq, tg, w))
        xt = torch.as_tensor(x, dtype=torch.float32, device=task.device)
        mu, _ = predict(task.models["cnn"], xt, torch.zeros(xt.shape[0], device=task.device))
        out[f"cnn_pred_{name}"] = (mu[:, 0].cpu().numpy() * sd.y_std + sd.y_mean).tolist()
    for p in ("ar", "cnn"):
        a, b = np.asarray(out[f"{p}_pred_ref"]), np.asarray(out[f"{p}_pred_sur"])
        out[f"{p}_max_pred_diff_over_ystd"] = float(np.max(np.abs(a - b)) / sd.y_std)
    nro = int(res_cfg["readout_ns"]) // 10
    ref_f = np.stack(rates[n_s:])[:, -nro:]
    sur_f = simulate_suffixes(pre, suf, sd.env, nro, options=task.options, device=task.device)
    out["reservoir_max_rel_feature_diff"] = float(np.max(np.abs(ref_f - sur_f)) / np.max(sur_f))
    fdev = sur_f - sur_f.mean(axis=0)
    out["reservoir_max_diff_over_signal"] = float(np.max(np.abs(ref_f - sur_f)) / max(np.max(np.abs(fdev)), 1e-30))
    rr = task.ridges["reservoir"]
    out["reservoir_max_pred_diff_over_ystd"] = float(np.max(np.abs(rr.predict(ref_f) - rr.predict(sur_f))) / sd.y_std)
    out["seconds"] = time.perf_counter() - t0
    out["n_series"], out["n_reservoir"] = n_s, n_r
    return out


# --- raw example traces -------------------------------------------------------------------------------------------


def log_example_traces(task: ForecastTask) -> None:
    """Seed 0, test sequence 0 of the system, both conditions, 10 shots each."""
    for cname, sd in task.data0.items():
        cond = condition_for(task.cfg, cname)
        sim = SurrogateSimulator(task.device, task.options)
        rng = np.random.default_rng([task.base_seed, 3])
        seq = task.series.arrays(sd.amps["test"][:1], sd.line_hz, cond.mw_offset_hz).unbatch()[0].to_sequence()
        task.run.log_trace(sim.run(seq, sd.env, shots=10, rng=rng), seq, condition=cname, kind="system")


# --- plots --------------------------------------------------------------------------------------------------------


def plot_inputs(task: ForecastTask) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    ser = task.series
    sd = task.data0["resonant"]
    k = np.arange(ser.total_cycles)
    ax = axes[0]
    ax.plot(k, sd.amps["test"][0], color="#52514e", lw=1.2, label="drive amplitude a_k (hidden)")
    ax.set_xlabel("cycle k")
    ax.set_ylabel("MW amplitude")
    ax2 = ax.twinx()
    for cname, col in (("resonant", COLORS["cnn"]), ("detuned", COLORS["persistence"])):
        if cname in task.data0:
            y = task.data0[cname].means["test"][0].sum(-1)
            ax2.plot(k, y / y.mean(), color=col, lw=1.6, label=f"reading, {cname}")
    ax2.set_ylabel("noiseless reading / mean")
    ax.set_title("one test sequence (seed 0)", fontsize=9)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax = axes[1]
    b = ser.burn_in
    a = sd.amps["test"][:, b + 1 :].ravel()
    a_prev = sd.amps["test"][:, b:-1].ravel()
    y = sd.means["test"][:, b + 1 :].sum(-1).ravel()
    sc = ax.scatter(a, y / y.mean(), c=a_prev, s=3, cmap="viridis", alpha=0.6, lw=0)
    fig.colorbar(sc, ax=ax, label="previous drive a_(k-1)")
    ax.set_xlabel("drive amplitude a_k")
    ax.set_ylabel("noiseless reading y_k / mean")
    ax.set_title("response: Rabi (non-monotonic) + spin memory", fontsize=9)
    _style(ax)
    ax = axes[2]
    m = sd.means["test"][:, b:]
    t = (np.arange(ser.n_rebinned) + 0.5) * ser.rebin * ser.grid_ns
    lo, hi = np.argsort(sd.amps["test"][:, b:].ravel())[[0, -1]]
    flat = m.reshape(-1, ser.n_rebinned)
    for j, lab, col in ((lo, "lowest drive", COLORS["ar"]), (hi, "highest drive", COLORS["cnn"])):
        ax.plot(t, flat[j] * 1e3, color=col, lw=1.6, label=lab)
    ax.plot(t, flat.mean(axis=0) * 1e3, color="#8a8985", lw=1.2, label="mean")
    ax.set_xlabel("time in readout (ns)")
    ax.set_ylabel("counts per shot per cell (x1e-3)")
    ax.set_title(f"readout transient ({ser.rebin * ser.grid_ns} ns cells)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_quantum_test(task: ForecastTask, qt: QuantumTestResult) -> Figure:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ticks, labels = [], []
    for x, p in enumerate((*PREDICTORS, "oracle")):
        s = qt.summary(f"{p}_skill")
        for j, (cname, alpha) in enumerate((("resonant", 1.0), ("detuned", 0.45))):
            v, lo, hi = s[cname]
            ax.bar(x + 0.36 * j, v, 0.32, color=COLORS[p], alpha=alpha, edgecolor="none")
            ax.errorbar(x + 0.36 * j, v, yerr=[[v - lo], [hi - v]], color="#0b0b0b", lw=1, capsize=3)
        ticks.append(x + 0.18)
        labels.append(LABELS[p].replace(", ", ",\n").replace(" + ", "\n+ "))
    c = qt.summary("ceiling_skill")["resonant"][0]
    ax.axhline(c, color=COLORS["ceiling"], ls=":", lw=1.2, label=f"noise ceiling, resonant ({c:.2f})")
    ax.axhline(0, color="#0b0b0b", lw=0.8)
    ax.set_xticks(ticks, labels, fontsize=7)
    ax.set_ylabel("skill = 1 - MSE / Var(y)")
    ax.set_title("quantum test: resonant (solid) vs detuned +0.4 GHz (light); 95 % t CI over seeds", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    _style(ax)
    fig.tight_layout()
    return fig


def plot_predictions(task: ForecastTask) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ex = task.examples.get("resonant")
    if ex:
        k = np.asarray(ex["cycle"])
        norm = float(np.mean(ex["noiseless"]))
        ax.plot(k, np.asarray(ex["target"]) / norm, "o", color="#0b0b0b", ms=3, label="reading (noisy, the target)")
        ax.plot(k, np.asarray(ex["noiseless"]) / norm, color="#8a8985", lw=1, label="noiseless reading")
        for p in ("ar", "cnn", "reservoir", "oracle"):
            ax.plot(k, np.asarray(ex[p]) / norm, color=COLORS[p], lw=1.6, label=LABELS[p])
    ax.set_xlabel("cycle k (prediction of y_k from y_(k-w) .. y_(k-1))")
    ax.set_ylabel("reading / mean")
    ax.set_title("one-step predictions, seed 0, test sequence 0, resonant", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=3)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_timing(task: ForecastTask) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    df = pd.DataFrame(task.timing_rows)
    ax = axes[0]
    for name, ls in (("retrained", "-"), ("frozen", "--")):
        d = df[df.readout == name]
        if len(d):
            xs = sorted(d.shift_ns.unique())
            _band(ax, xs, [d[d.shift_ns == x].skill.to_numpy() for x in xs], COLORS["reservoir"], ls, f"ridge, {name}")
    ax.axhline(0, color="#0b0b0b", lw=0.8)
    ax.set_xlabel("shift of every reservoir MW duration (ns)")
    ax.set_ylabel("NV reservoir skill")
    ax.set_title("reservoir timing probe (resonant)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    if task.saliency:
        s = task.saliency
        ax = axes[1]
        t = (np.arange(len(s["per_bin"])) + 0.5) * task.series.rebin * task.series.grid_ns
        pb, ct = np.asarray(s["per_bin"]), np.asarray(s["contrast"])
        ax.plot(t, pb / pb.max(), color=COLORS["cnn"], lw=1.8, label="CNN saliency")
        ax.plot(t, ct / ct.max(), color=COLORS["oracle"], lw=1.8, label="drive contrast (sd / mean)")
        ax.set_xlabel("time in readout (ns)")
        ax.set_ylabel("normalised")
        ax.set_title(f"CNN saliency per readout cell, r = {s['corr_bin_contrast']:.2f}", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        _style(ax)
        ax = axes[2]
        pl = np.asarray(s["per_lag"])
        lags = np.arange(-pl.size, 0) + 1
        ax.bar(lags, pl / pl.sum(), color=COLORS["cnn"])
        ax.set_xlabel("past cycle (0 = latest reading)")
        ax.set_ylabel("share of saliency")
        ax.set_title("CNN saliency per past cycle", fontsize=9)
        _style(ax)
    fig.tight_layout()
    return fig


# --- report -------------------------------------------------------------------------------------------------------


def timing_verdicts(task: ForecastTask) -> dict[str, Any]:
    df = pd.DataFrame(task.timing_rows)
    out: dict[str, Any] = {"strategy_learns_timing": False}
    if not len(df):
        return out
    shifts = sorted(df.shift_ns.unique())
    big = [s for s in shifts if abs(s) == max(abs(x) for x in shifts)]
    for name in ("frozen", "retrained"):
        out[f"shift_drop_{name}"] = [
            paired_diff(df, "skill", {"readout": name, "shift_ns": 0}, {"readout": name, "shift_ns": s}) for s in big
        ]
    out["shifts"] = [int(s) for s in big]
    out["reservoir_readout_relies_on_pulse_timing"] = any(d[1] > 0 for d in out["shift_drop_frozen"])
    out["reservoir_information_needs_exact_timing"] = any(d[1] > 0 for d in out["shift_drop_retrained"])
    if task.saliency:
        out["cnn_weighting_follows_contrast"] = task.saliency["corr_bin_contrast"] >= 0.8
    return out


def report_body(task: ForecastTask, qt: QuantumTestResult, tv: dict[str, Any], ref: dict[str, Any]) -> str:
    cfg = task.cfg
    ser = task.series
    n_seeds = len(qt.seeds)
    d, rc = cfg["data"], cfg["reservoir"]
    lines: list[str] = []
    a = lines.append
    a(
        "Task 3 of the brief: predict the next fluorescence reading. The NV ensemble is driven by a hidden, "
        "correlated random MW amplitude, one pulse per cycle, each followed by a short readout; the readings form "
        "a time series, and every predictor sees only past readings. Trivial baselines, linear AR, a CNN decoder "
        "and the NV reservoir (fixed random pulse encoding + ridge) against two upper references, over "
        f"{n_seeds} seeds on the fast surrogate, re-checked on the QuTiP reference; and the quantum test."
    )
    a("")
    a("## The series")
    a("")
    env0 = task.data0["resonant"].env
    a(
        f"- **Device** per seed as in Task 2 (static field {tuple(cfg['device']['bias_field_t'])} T, nuisance from "
        f"the milestone-2 ranges; seed 0: T2* {env0.t2_star_s * 1e6:.2f} µs, Rabi full scale "
        f"{env0.rabi_full_scale_hz / 1e6:.2f} MHz, beta {env0.optical_beta_full_scale:.3f}). M = "
        f"{task.options.n_detuning} quantile nodes."
    )
    a(
        f"- **Sequence**: {ser.init_ns} ns init laser, then {ser.burn_in} burn-in + {ser.n_cycles} cycles of "
        f"[{ser.mw_ns} ns MW on line {ser.line_index} at amplitude a_k | {ser.readout_ns} ns readout laser | "
        f"{ser.gap_ns} ns dark] ({ser.n_bins * ser.grid_ns / 1e3:.1f} µs). The reading y_k is the photon count of "
        f"cycle k's readout summed over {int(task.shots)} shots (the whole sequence repeated)."
    )
    st = pd.DataFrame(task.encoding)
    area = ""
    if len(st):
        area = (
            f" (pulse areas {st.drive_area_pi_p5.median():.2f}-{st.drive_area_pi_p95.median():.2f} pi, 5-95 %, "
            "median over seeds)"
        )
    a(
        f"- **Drive** (hidden): z_k stationary AR(1), correlation {ser.rho}, a_k = {ser.amp_center} + "
        f"{ser.amp_scale} tanh(z_k){area}. The response is non-monotonic in a_k (Rabi), and the "
        f"{ser.readout_ns} ns readout repolarises the spins only partly, so y_k also depends on earlier cycles."
    )
    a(
        f"- **Data** per seed: {d['n_train_seq']} training / {d['n_val_seq']} validation / {d['n_test_seq']} test "
        f"sequences, independent drives; predictions of y_(k+1) from the last {task.window} readings (cycles after "
        "the burn-in). Test noise: one fixed Poisson realisation shared by every predictor."
    )
    a("")
    a("## Predictors")
    a("")
    a("- **mean** (skill 0 by definition) and **persistence** (y_hat_(k+1) = y_k).")
    a(
        f"- **linear AR**: ridge (GCV penalty) on the last {task.window} readings, fitted on {d['ar_draws']} noise "
        "draws of the training set."
    )
    c = cfg["cnn"]
    a(
        f"- **CNN decoder** (`ConvDecoder`, T = {ser.n_rebinned} readout cells of {ser.rebin * ser.grid_ns} ns, F = "
        f"{task.window} past cycles, Gaussian NLL): {c['steps']} steps of {c['batch']} windows with fresh Poisson "
        "noise, best validation NLL kept."
    )
    e = rc["encoder"]
    a(
        f"- **NV reservoir**: the past readings, standardised, are the input u ({task.window} numbers) of a fixed "
        f"random pulse encoder ({e['n_segments']} segments of {e['slot_ns']} ns, as in Task 2) that writes them "
        f"into a separate sequence on the same diamond ({rc['init_ns']} ns init, {rc['settle_ns']} ns dark, the block, "
        f"{rc['readout_ns']} ns readout, {int(float(rc['shots']))} shots); ridge on its readout counts, trained on "
        f"{rc['n_train']} windows. Variants: + direct input (ridge on [reservoir counts, u]), and in the resonant arm "
        "the reservoir with only its encoding detuned by the quantum-test offset."
    )
    a(
        f"- **Hidden-state oracle**: E[y_(k+1) | true z_k and spin state after cycle k], a {cfg['oracle']['n_quad']}-"
        "node Gauss-Hermite average over the drive innovation, simulated from the exact state "
        "(`DrivenSeries.oracle_next_mean`). Bayes-optimal for anyone who knew everything hidden; a predictor "
        "that sees only noisy past readings cannot beat it on average. **Noise ceiling**: 1 - E[Poisson variance] / "
        "Var(y), what knowing the next noiseless reading would give."
    )
    a("- Skill = 1 - MSE / Var(y) on the noisy test readings.")
    a("")
    a("## Quantum test and results")
    a("")
    a(
        f"Every predictor retrained and re-evaluated with every MW frequency (the system's drive and the reservoir's "
        f"encoding) shifted by +{cfg['detuned_offset_hz'] / 1e6:.0f} MHz (nearest line "
        f"{task.no_line_margin_hz / 1e6:.0f} MHz from any MW frequency), same devices, drives and noise seeds "
        f"(paired). 95 % Student-t CIs over the {n_seeds} seeds; gap > 0 means better on resonance."
    )
    a("")
    a(qt.to_markdown(fmt="{:.3g}"))
    rows = pd.DataFrame(task.rows)
    zr = rows[(rows.metric == "cnn_z_rms") & (rows.condition == "resonant")].value.to_numpy()
    if zr.size:
        a("")
        a(f"CNN uncertainty calibration on resonance: RMS of standardised errors {_ci(t_ci(zr))} (1 = calibrated).")
    a("")
    a(outcome(qt))
    a("")
    a(comparison(qt))
    a("")
    a("### Does the strategy use microsecond-scale timing?")
    a("")
    a(
        "No predictor chooses any timing (the system's protocol is fixed, the reservoir's encoding is a fixed random "
        "map): `strategy_learns_timing` = **False**. Logged dependence on spin timing:"
    )
    a("")
    if "shift_drop_frozen" in tv:
        for name in ("retrained", "frozen"):
            parts = "; ".join(
                f"{s:+d} ns: {_ci(x)}" for s, x in zip(tv["shifts"], tv[f"shift_drop_{name}"], strict=True)
            )
            a(f"- NV reservoir skill at shift 0 minus with every reservoir MW duration shifted, ridge {name}: {parts}.")
        a(
            f"  Verdicts: `reservoir_readout_relies_on_pulse_timing` = "
            f"**{tv['reservoir_readout_relies_on_pulse_timing']}** (frozen readout), "
            f"`reservoir_information_needs_exact_timing` = **{tv['reservoir_information_needs_exact_timing']}** "
            "(retrained)."
        )
    if len(st):
        a(
            f"- Reservoir encoder pulse areas: median {st.median_area_pi.median():.2f} pi, "
            f"{100 * st.frac_area_quarter_to_4pi.mean():.0f} % of segments between pi/4 and 4 pi (Rabi-scale)."
        )
    if task.saliency:
        s = task.saliency
        a(
            f"- CNN (seed 0): saliency per readout cell vs the drive-induced contrast per cell r = "
            f"{s['corr_bin_contrast']:.2f}; {100 * s['last_lag_fraction']:.0f} % of the saliency on the latest "
            f"reading, the rest on older cycles (uniform: {100 / task.window:.0f} %)."
        )
    a(
        "- The system's own dynamics carry microsecond structure (the Rabi response to the 200 ns pulse, the "
        "partial repolarisation within the 400 ns readout); the predictors exploit it only through the readings."
    )
    a("")
    a("## Reference check (QuTiP)")
    a("")
    a(
        f"Seed 0, resonant: {ref.get('n_series', 0)} full system test sequences and {ref.get('n_reservoir', 0)} "
        "reservoir sequences re-simulated on the QuTiP reference (same quadrature nodes; "
        f"{ref.get('seconds', 0):.0f} s on the CPU pool):"
    )
    a("")
    a("| quantity | value |")
    a("| --- | --- |")
    for k, lab in (
        ("series_max_rel_cell_diff", "system: max readout-cell diff / peak"),
        ("series_max_reading_diff_over_signal", "system: max reading diff / max drive-induced deviation"),
        ("ar_max_pred_diff_over_ystd", "AR prediction diff (noiseless inputs) / sd(y)"),
        ("cnn_max_pred_diff_over_ystd", "CNN prediction diff (noiseless inputs) / sd(y)"),
        ("reservoir_max_rel_feature_diff", "reservoir: max feature diff / peak"),
        ("reservoir_max_diff_over_signal", "reservoir: max feature diff / max input-induced deviation"),
        ("reservoir_max_pred_diff_over_ystd", "reservoir prediction diff / sd(y)"),
    ):
        if k in ref:
            a(f"| {lab} | {ref[k]:.2e} |")
    a("")
    a("## Notes and open issues")
    a("")
    a(
        "- The hidden-state oracle bounds what is predictable: most of the variance of y_(k+1) comes from the next "
        "drive innovation (1 - rho^2 of the drive variance), which no past reading carries."
    )
    a(
        "- The drive process, its range and the cycle timing are assumptions of the task, chosen so the response is "
        "non-monotonic (room for nonlinear predictors) and has spin memory; other choices change every number."
    )
    a(
        "- The reservoir sees the readings through a fixed random encoding and a single readout per input (nearly "
        "one number, see Task 2); a trained encoder (next phase-2 item) is the natural follow-up."
    )
    a("- No claim of quantum advantage: the gap shows only that the predictable structure comes from the driven spins.")
    return "\n".join(lines)


def outcome(qt: QuantumTestResult) -> str:
    parts = []
    for p in ("ar", "cnn", "reservoir", "reservoir_direct"):
        s = qt.summary(f"{p}_skill")
        sig = "significant" if qt.gap_significant(f"{p}_skill") else "not significant"
        parts.append(f"{LABELS[p]} gap {_ci(s['gap'])} ({sig}; detuned {_ci(s['detuned'])})")
    return (
        "Outcome (skill): "
        + "; ".join(parts)
        + ". Detuned, the drive moves no spin, the readings are a constant plus shot noise and nothing is "
        "predictable (skill <= 0 for every predictor, persistence and AR slightly negative from fitting noise)."
    )


def comparison(qt: QuantumTestResult) -> str:
    """Generated: resonant ranking and the reservoir's dependence on its own resonance."""

    def diff(a: str, b: str) -> tuple[float, float, float]:
        va, vb = np.asarray(qt.values["resonant"][a]), np.asarray(qt.values["resonant"][b])
        return t_ci(va - vb)

    out = []
    for a_, b_ in (
        ("cnn_skill", "ar_skill"),
        ("reservoir_skill", "ar_skill"),
        ("reservoir_direct_skill", "ar_skill"),
        ("reservoir_skill", "reservoir_offres_skill"),
        ("oracle_skill", "cnn_skill"),
    ):
        d = diff(a_, b_)
        verdict = "better" if d[1] > 0 else ("worse" if d[2] < 0 else "no significant difference")
        out.append(f"{a_.removesuffix('_skill')} - {b_.removesuffix('_skill')}: {_ci(d)} ({verdict})")
    return "Paired differences on resonance (95 % t CI over seeds): " + "; ".join(out) + "."


# --- main ---------------------------------------------------------------------------------------------------------


def main(
    cfg: dict[str, Any], output_root: Path | None = None, report_dir: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    """Run the Task 3 milestone; returns (report path, summary numbers)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kw: dict[str, Any] = {} if output_root is None else {"output_root": output_root}
    t_start = time.perf_counter()
    with RunLog(str(cfg["name"]), config=cfg, seed=int(cfg["seed"]), **kw) as run:
        task = ForecastTask(cfg, run, device)
        conds = (RESONANT, detuned_condition(float(cfg["detuned_offset_hz"])))
        hib = {f"{p}_skill": True for p in (*PREDICTORS, "oracle", "ceiling")}
        qt = run_quantum_test(task, [int(s) for s in cfg["seeds"]], conds, higher_is_better=hib)
        print("[m3-forecast] reference check", flush=True)
        ref = run_reference_check(task)
        log_example_traces(task)
        tv = timing_verdicts(task)

        pd.DataFrame(task.rows).to_parquet(run.dir / "results.parquet")
        pd.DataFrame(task.timing_rows).to_parquet(run.dir / "timing.parquet")
        pd.DataFrame(task.encoding).to_parquet(run.dir / "encoding.parquet")
        data = run.dir / "data"
        data.mkdir(exist_ok=True)
        for cname, sd in task.data0.items():
            np.savez_compressed(
                data / f"series-seed0-{cname}.npz",
                test_means=sd.means["test"],
                test_amps=sd.amps["test"],
                test_cells=sd.test_cells,
            )
        (run.dir / "examples.json").write_text(json.dumps(task.examples), encoding="utf-8")
        (run.dir / "train_history.json").write_text(json.dumps(task.histories), encoding="utf-8")
        (run.dir / "reference_check.json").write_text(json.dumps(ref), encoding="utf-8")
        (run.dir / "timing_verdicts.json").write_text(json.dumps(tv, default=str), encoding="utf-8")
        (run.dir / "saliency.json").write_text(json.dumps(task.saliency), encoding="utf-8")
        (run.dir / "quantum_test.json").write_text(
            json.dumps({"seeds": qt.seeds, "values": qt.values}), encoding="utf-8"
        )
        figs = {
            "inputs": plot_inputs(task),
            "quantum-test": plot_quantum_test(task, qt),
            "predictions": plot_predictions(task),
            "timing": plot_timing(task),
        }
        paths = [run.save_figure(f, name, dpi=90) for name, f in figs.items()]
        for f in figs.values():
            plt.close(f)
        summary: dict[str, Any] = {
            "device": f"{device.type} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}".strip()
        }
        for m in qt.metrics():
            s = qt.summary(m)
            summary[f"{m}_resonant"] = s["resonant"][0]
            summary[f"{m}_detuned"] = s["detuned"][0]
            summary[f"{m}_gap"] = s["gap"][0]
            summary[f"{m}_gap_lo"] = s["gap"][1]
            summary[f"{m}_gap_hi"] = s["gap"][2]
        for k in ("series_max_rel_cell_diff", "reservoir_max_rel_feature_diff"):
            summary[f"ref_{k}"] = ref.get(k, float("nan"))
        summary["strategy_learns_timing"] = 0.0
        if "reservoir_readout_relies_on_pulse_timing" in tv:
            summary["reservoir_readout_relies_on_pulse_timing"] = float(tv["reservoir_readout_relies_on_pulse_timing"])
            summary["reservoir_information_needs_exact_timing"] = float(tv["reservoir_information_needs_exact_timing"])
        summary["no_line_margin_mhz"] = task.no_line_margin_hz / 1e6
        for k, v in task.seconds.items():
            summary[f"{k}_seconds"] = v
        summary["wall_seconds"] = time.perf_counter() - t_start
        run.log_metrics(step=0, **{k: v for k, v in summary.items() if isinstance(v, int | float)})
        body = report_body(task, qt, tv, ref)
    rdir = report_dir if report_dir is not None else run.dir / "report"
    title = "Milestone 3, Task 3: predict the next fluorescence reading"
    path = write_report(run.dir, title, body, figures=paths, report_dir=rdir, name=REPORT_NAME)
    return path, summary


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--quick", action="store_true", help="short series and short training (smoke run)")
    p.add_argument("--report-dir", type=Path, default=None, help="where to write the report (e.g. docs/results)")
    p.add_argument("--output-root", type=Path, default=None, help="run directories (default: <repo>/outputs)")
    a = p.parse_args()
    path, summary = main(load_config(a.config, a.quick), a.output_root, a.report_dir)
    for k, v in summary.items():
        print(f"{k}: {v:.6g}" if isinstance(v, float) else f"{k}: {v}")
    print(f"report: {path}")


if __name__ == "__main__":
    cli()

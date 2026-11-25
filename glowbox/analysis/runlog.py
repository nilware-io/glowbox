"""Local run logging: every run gets ``outputs/<run-id>/`` with its config, seed, provenance and raw data.

Layout of a run directory (``outputs/`` is gitignored)::

    outputs/<run-id>/
        run.yaml          config, seed, git hash + dirty flag, environment (device, library versions), start/end
        traces.parquet    one row per (trace, shot): counts, noiseless rate, bin width, sequence, metadata, tags
        metrics.parquet   one row per log_metrics() call: step + named scalars
        plots/            figures saved with save_figure()

``write_report`` turns a run into a short Markdown report in ``docs/results/`` (committed), copying its figures
next to it. Hostnames, usernames and absolute paths are never recorded.

Usage::

    with RunLog("rabi-demo", config=cfg, seed=1234) as run:
        trace = sim.run(seq, env, shots=100, rng=run.rng)
        run.log_trace(trace, seq, rabi_ns=40)
        run.log_metrics(step=0, contrast=0.12)
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import secrets
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from glowbox.control.pulse import PulseSequence
from glowbox.sim.interface import FluorescenceTrace

if TYPE_CHECKING:
    from matplotlib.figure import Figure

#: Repository root (this file is glowbox/analysis/runlog.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs"
DEFAULT_REPORT_DIR = REPO_ROOT / "docs" / "results"

TRACE_SCHEMA = pa.schema(
    [
        ("trace_id", pa.int64()),
        ("shot", pa.int64()),
        ("bin_ns", pa.int64()),
        ("counts", pa.list_(pa.int64())),
        ("rate_cps", pa.list_(pa.float64())),  # null when the source gives no noiseless rate (hardware)
        ("sequence", pa.string()),  # JSON of PulseSequence.to_dict(), repeated per shot (compressed well)
        ("metadata", pa.string()),  # JSON of FluorescenceTrace.metadata
        ("tags", pa.string()),  # JSON of the log_trace() keyword tags
    ]
)


@dataclass(frozen=True)
class GitInfo:
    commit: str | None
    dirty: bool | None


def git_info(repo: Path = REPO_ROOT) -> GitInfo:
    """Commit hash and dirty flag (untracked files count as dirty); ``None`` fields when git is unavailable."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout
        return GitInfo(commit=commit, dirty=bool(status.strip()))
    except (OSError, subprocess.CalledProcessError):
        return GitInfo(commit=None, dirty=None)


def _version(dist: str) -> str | None:
    try:
        return importlib_metadata.version(dist)
    except importlib_metadata.PackageNotFoundError:
        return None


def environment_info() -> dict[str, Any]:
    """Python/platform, library versions and compute device. No hostname or user information."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.machine()}",
        "versions": {d: _version(d) for d in ("glowbox", "torch", "qutip", "numpy", "scipy", "pyarrow")},
    }
    try:
        import torch

        cuda = torch.cuda.is_available()
        info["device"] = "cuda" if cuda else "cpu"
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if cuda else None
    except ImportError:  # pragma: no cover - torch is a hard dependency
        info["device"] = "cpu"
    return info


def new_run_id(name: str) -> str:
    """``YYYYmmdd-HHMMSS-<name>-<4 hex>``, UTC, sortable and collision-safe."""
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name)
    return f"{stamp}-{safe}-{secrets.token_hex(2)}"


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, tuple):
        return list(x)
    raise TypeError(f"not JSON serialisable: {type(x).__name__}")


def _dumps(x: Any) -> str:
    return json.dumps(x, default=_jsonable, sort_keys=True)


class RunLog:
    """One run's output directory. Use as a context manager so Parquet files are closed and run.yaml finalised."""

    def __init__(
        self,
        name: str,
        config: dict[str, Any],
        seed: int,
        output_root: Path | str = DEFAULT_OUTPUT_ROOT,
        run_id: str | None = None,
    ) -> None:
        self.run_id = run_id or new_run_id(name)
        self.dir = Path(output_root) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=False)
        self.seed = int(seed)
        #: Seeded generator for the run; pass it to Simulator.run so the run is reproducible from run.yaml.
        self.rng = np.random.default_rng(self.seed)
        g = git_info()
        self.info: dict[str, Any] = {
            "run_id": self.run_id,
            "name": name,
            "seed": self.seed,
            "git_commit": g.commit,
            "git_dirty": g.dirty,
            "started_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "finished_utc": None,
            "environment": environment_info(),
            "config": json.loads(_dumps(config)),
        }
        self._trace_writer: pq.ParquetWriter | None = None
        self._n_traces = 0
        self._metrics: list[dict[str, Any]] = []
        self._write_yaml()

    def _write_yaml(self) -> None:
        with open(self.dir / "run.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(self.info, f, sort_keys=False)

    def log_trace(self, trace: FluorescenceTrace, sequence: PulseSequence | None = None, **tags: Any) -> int:
        """Append a trace (all shots). Returns its ``trace_id``."""
        tid = self._n_traces
        self._n_traces += 1
        n = trace.shots
        rates = [None] * n if trace.rate_cps is None else [r.tolist() for r in np.asarray(trace.rate_cps)]
        seq_json = _dumps(sequence.to_dict()) if sequence is not None else None
        table = pa.table(
            {
                "trace_id": [tid] * n,
                "shot": list(range(n)),
                "bin_ns": [trace.bin_ns] * n,
                "counts": [r.tolist() for r in np.asarray(trace.counts)],
                "rate_cps": rates,
                "sequence": [seq_json] * n,
                "metadata": [_dumps(trace.metadata)] * n,
                "tags": [_dumps(tags)] * n,
            },
            schema=TRACE_SCHEMA,
        )
        if self._trace_writer is None:
            meta = {b"glowbox_run": _dumps({k: v for k, v in self.info.items() if k != "config"}).encode()}
            self._trace_writer = pq.ParquetWriter(self.dir / "traces.parquet", TRACE_SCHEMA.with_metadata(meta))
        self._trace_writer.write_table(table)
        return tid

    def log_metrics(self, step: int, **values: float) -> None:
        """Record scalar metrics; written to metrics.parquet on close."""
        self._metrics.append({"step": int(step), **{k: float(v) for k, v in values.items()}})

    def save_figure(self, fig: Figure, name: str, dpi: int = 120) -> Path:
        """Save a matplotlib figure as ``plots/<name>.png``."""
        path = self.dir / "plots" / f"{name}.png"
        path.parent.mkdir(exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        return path

    def close(self) -> None:
        if self._trace_writer is not None:
            self._trace_writer.close()
            self._trace_writer = None
        if self._metrics:
            pq.write_table(pa.Table.from_pylist(self._metrics), self.dir / "metrics.parquet")
        self.info["finished_utc"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        self._write_yaml()

    def __enter__(self) -> RunLog:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if exc_type is not None:
            self.info["error"] = f"{exc_type.__name__}: {exc}"
        self.close()


def load_run(run_dir: Path | str) -> dict[str, Any]:
    """Read run.yaml and the Parquet tables of a run (``traces``/``metrics`` are pandas DataFrames or None)."""
    d = Path(run_dir)
    with open(d / "run.yaml", encoding="utf-8") as f:
        info: dict[str, Any] = yaml.safe_load(f)
    tp, mp = d / "traces.parquet", d / "metrics.parquet"
    info["traces"] = pq.read_table(tp).to_pandas() if tp.exists() else None
    info["metrics"] = pq.read_table(mp).to_pandas() if mp.exists() else None
    return info


def write_report(
    run_dir: Path | str,
    title: str,
    summary_md: str,
    figures: Sequence[Path | str] = (),
    report_dir: Path | str = DEFAULT_REPORT_DIR,
    name: str | None = None,
) -> Path:
    """Write ``<report_dir>/<name>.md`` (default name: the run id) for a finished run and copy ``figures`` to
    ``<report_dir>/figures/<name>-<figure>.png``.

    The report holds the title, the caller's summary (results, conclusions), provenance (git, seed, environment),
    the config and the final metrics. Figures are referenced by relative path so the report renders in the repo.
    """
    info = load_run(run_dir)
    out_dir = Path(report_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = info["run_id"]
    stem = name or run_id
    lines = [f"# {title}", "", summary_md.strip(), ""]
    for fpath in figures:
        src = Path(fpath)
        fig_dir.mkdir(exist_ok=True)
        dst = fig_dir / f"{stem}-{src.name}"
        shutil.copyfile(src, dst)
        lines += [f"![{src.stem}](figures/{dst.name})", ""]
    env = info["environment"]
    dirty = " (dirty tree)" if info.get("git_dirty") else ""
    lines += [
        "## Provenance",
        "",
        f"- run id: `{run_id}`",
        f"- git: `{info.get('git_commit')}`{dirty}",
        f"- seed: {info['seed']}",
        f"- device: {env.get('device')} {env.get('gpu') or ''}".rstrip(),
        f"- python {env.get('python')}, "
        + ", ".join(f"{k} {v}" for k, v in env.get("versions", {}).items() if v is not None),
        f"- started {info.get('started_utc')}, finished {info.get('finished_utc')}",
        "",
        "## Config",
        "",
        "```yaml",
        yaml.safe_dump(info["config"], sort_keys=False).strip(),
        "```",
        "",
    ]
    metrics = info["metrics"]
    if metrics is not None and len(metrics):
        last = metrics.iloc[-1]
        lines += ["## Final metrics", "", "| metric | value |", "| --- | --- |"]
        lines += [f"| {k} | {last[k]:.6g} |" for k in metrics.columns if k != "step"]
        lines.append("")
    path = out_dir / f"{stem}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

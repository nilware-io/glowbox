# Development setup

Glowbox is a uv-managed Python 3.12 project. One universal `uv.lock` serves every machine: on Windows and Linux,
PyTorch comes from the PyTorch CUDA 12.8 index (`cu128`; the first CUDA line with kernels for Blackwell, sm_120,
e.g. RTX 50-series), which also runs on machines without a GPU. All code is device-agnostic: pick the device with
`torch.device("cuda" if torch.cuda.is_available() else "cpu")`.

## Local (CPU is fine)

1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `pip install --user uv`).
2. `uv sync` in the repository: creates `.venv/` with Python 3.12 (`.python-version`), all runtime and dev
   dependencies (pytest, ruff, mypy). The first sync downloads the CUDA build of torch (a few GB) even on a CPU box.
3. Check:

       uv run pytest              # GPU tests are skipped without CUDA; slow tests deselected
       uv run pytest -m slow      # the slow ones (milestone smoke runs, minutes on a CPU)
       uv run ruff check . && uv run ruff format --check .
       uv run mypy

Keep local CPU runs short (minutes); long training and sweeps belong on the GPU machine.

## GPU machine (Windows or Linux, NVIDIA driver recent enough for CUDA 12.8)

1. Install uv and git. uv installs Python 3.12 itself if needed (`uv python install 3.12`).
2. Clone the repository there (see "Sync" below) and run `uv sync`.
3. Verify the CUDA build sees the card:

       uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.version.cuda)"
       uv run pytest              # includes the CUDA matmul test

On Windows, `cmd.exe` is the default remote shell over ssh: use double quotes and `&` to chain commands, e.g.
`ssh <gpu-machine> "cd /d %USERPROFILE%\path\to\glowbox && uv run pytest -q"`.

## Sync: push-to-checkout

There is no central remote yet. The GPU machine holds a non-bare clone configured to update its working tree when
pushed to:

    # once, on the GPU machine, inside its clone
    git config receive.denyCurrentBranch updateInstead

    # once, on the development machine
    git remote add gpu <gpu-machine>:path/to/glowbox

    # every time
    git commit ...
    git push gpu main            # the GPU machine's checkout now matches
    ssh <gpu-machine> "cd ... && uv sync && uv run ..."

`updateInstead` refuses the push if the remote working tree has local changes to tracked files. So never edit
tracked files on the GPU machine: only gitignored things (`.venv/`, `outputs/`, caches) may change there. Results
come back as files under `outputs/<run-id>/` (copy them with scp if needed); reports go to `docs/results/` from the
development machine.

## Run outputs

`glowbox.analysis.runlog.RunLog` creates `outputs/<run-id>/` (gitignored) with `run.yaml` (config, seed, git commit
and dirty flag, environment), `traces.parquet`, `metrics.parquet` and `plots/`. `write_report` writes a Markdown
summary to `docs/results/<run-id>.md` (committed). Configs are plain YAML.

## Conventions

- Physics: `docs/physics.md` is the spec; constants live in `glowbox/sim/constants.py` with units in names and a
  literature source each.
- Small tested steps; type hints throughout; `ruff` (lint + format, line length 120) and `mypy` clean.
- No hostnames, usernames, home paths or IP addresses in committed files.
- Confirmatory comparisons of decision arms (an outcome-contingent choice applied to the control) or observer arms
  (a process reading the results during the run) must go through `docs/preregistration/` (README there): a locked
  plan and analysis script, a calibration block of control runs, sealed outcomes and blinded arm codes. Without a
  locked plan such a comparison is exploratory and must be labelled so.

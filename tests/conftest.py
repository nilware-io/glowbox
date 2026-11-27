"""Shared test setup: cap torch's CPU threads so the suite does not oversubscribe a shared machine."""

from __future__ import annotations

import os

import torch

torch.set_num_threads(max(1, min(8, (os.cpu_count() or 1) // 2)))

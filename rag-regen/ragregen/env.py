"""Cache paths and GPU housekeeping. Import this before transformers.

HF_HOME must be set before huggingface_hub is imported anywhere, because the
library freezes its token path at import time.
"""
from __future__ import annotations

import gc
import os
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HF_CACHE = Path("/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/.cache")


def setup() -> None:
    """Point HuggingFace at the shared cache. Safe to call repeatedly."""
    os.environ["HF_HOME"] = str(HF_CACHE)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # Multi-image VLM batches have uneven activation sizes. Expandable CUDA
    # segments prevent reserved-but-unallocated fragments from stranding the
    # final few hundred MiB on a shared card.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                          "expandable_segments:True")


def _cuda_mem_get_info(device_index: int):
    """(free, total) bytes, or None when there is no usable card.

    Split out so tests can substitute it without a CUDA context.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.mem_get_info(device_index)
    except (RuntimeError, AssertionError):
        return None


def free_vram_gb(device_index: int = 0) -> float | None:
    """Free VRAM in GB, or None when it cannot be read.

    None means unknown, and every caller must treat unknown as permission to
    proceed -- blocking on it would abort every CPU-only run.
    """
    info = _cuda_mem_get_info(device_index)
    if info is None:
        return None
    return float(info[0]) / (1024 ** 3)


def exit_now(code: int = 0) -> None:
    """Exit immediately, skipping native-library teardown.

    torch, faiss and PIL each load native extensions into the same process,
    and unloading them together segfaults intermittently in this env. On
    2026-08-07 build_index took a SIGSEGV *after* writing a complete
    56,657-vector index and recording the run ok; the stage was scored as a
    failure and the campaign spent a second attempt re-encoding the corpus
    from scratch for nothing. The same crash on the pipeline stages would
    recur on every resumed attempt, because the work is already finished and
    teardown is all that is left to do.

    Once artifacts are on disk and run.json is written, nothing useful
    happens during shutdown -- so don't run it. Callers must have committed
    their work before calling this; buffered stdout is flushed here because
    os._exit skips the implicit flush too.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def reclaim_gpu() -> None:
    """Drop cached allocations. Call between stages, never mid-stage.

    A device-side failure poisons the CUDA context for the remainder of the
    process. Cleanup is best-effort in that state: surfacing a second error
    from ``empty_cache`` would mask the original, resumable stage failure.
    """
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except RuntimeError as exc:
            warnings.warn(f"CUDA cleanup skipped after device failure: {exc}",
                          RuntimeWarning, stacklevel=2)


setup()

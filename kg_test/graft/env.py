"""Cache paths and GPU housekeeping. Import this before transformers/diffusers.

Mirrors the idiom in rag-regen/ragregen/env.py: HF_HOME must be set before
huggingface_hub is imported anywhere, because the library freezes its token
path at import time.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

HF_CACHE = Path("/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/.cache")


def setup() -> None:
    """Point HuggingFace at the shared cache. Safe to call repeatedly."""
    os.environ["HF_HOME"] = str(HF_CACHE)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # Multi-image VLM batches have uneven activation sizes. Expandable CUDA
    # segments prevent reserved-but-unallocated fragments from stranding the
    # final few hundred MiB on a shared card.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # graft's phase-based model loading (kg_build/verify/generate each
    # unload a model the moment its phase ends, so peak GPU usage is one
    # model at a time -- see kg_build.py's module comment) creates and
    # destroys large nn.Module trees far more often than a typical script.
    # This exposed two DIFFERENT failure modes, both centered on gc.collect():
    #
    # 1. AUTOMATIC gc firing mid-construction segfaults. CPython's
    #    generational GC can trigger inline, anywhere an allocation crosses
    #    its threshold -- including mid-way through some *other* model's
    #    from_pretrained() (e.g. a HuggingFace fast tokenizer's Rust
    #    extension object, mid-init). Confirmed twice: a segfault inside a
    #    frame literally reading "Garbage-collecting", each time while a
    #    different model was under construction elsewhere on the stack.
    #    Fix: gc.disable() below, so GC only ever runs when WE call it.
    #
    # 2. But diffusers' SDXL pipeline objects hold genuine Python reference
    #    cycles (confirmed empirically: with gc entirely disabled, this
    #    process's own CUDA usage grew from a clean 0GB start to 23.3GB
    #    across just two successive generate() calls in refine()'s retry
    #    loop, then OOM'd mid-forward-pass -- each unloaded SDXL pipe's
    #    cyclic garbage was never collected, so its CUDA tensors were never
    #    released despite Models.unload() dropping our reference). Plain
    #    refcounting cannot break a cycle; only a cyclic GC pass can.
    #    Fix: free_gpu() calls gc.collect() explicitly. This is now safe
    #    specifically *because* automatic triggering is disabled -- the
    #    explicit call only ever runs at the controlled point after a
    #    model's reference is dropped and before the next model is
    #    constructed, never mid-construction.
    gc.disable()


def free_gpu() -> None:
    """Drop cyclic garbage (diffusers pipeline objects hold real reference
    cycles -- see setup()'s docstring) and clear the CUDA allocator cache.
    Safe to call frequently: with automatic gc disabled, this explicit
    gc.collect() is the ONLY place cyclic collection happens, always at a
    controlled point between phases, never mid-construction of another
    model."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass

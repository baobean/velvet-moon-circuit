import os
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from ragregen import env


def test_hf_cache_points_at_shared_cache():
    assert env.HF_CACHE == Path(
        "/mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/.cache"
    )
    assert env.HF_CACHE.is_dir()


def test_setup_exports_hf_home():
    env.setup()
    assert os.environ["HF_HOME"] == str(env.HF_CACHE)


def test_setup_enables_expandable_cuda_segments():
    env.setup()
    assert "expandable_segments:True" in os.environ[
        "PYTORCH_CUDA_ALLOC_CONF"]


def test_project_root_contains_configs_dir():
    assert (env.PROJECT_ROOT / "configs").is_dir()


def test_free_vram_is_none_without_a_card(monkeypatch):
    #: None means "unknown", and the preflight must treat unknown as
    #: permission to proceed -- otherwise every CPU test aborts.
    import ragregen.env as e
    monkeypatch.setattr(e, "_cuda_mem_get_info", lambda i: None)
    assert e.free_vram_gb(0) is None


def test_free_vram_converts_bytes_to_gigabytes(monkeypatch):
    import ragregen.env as e
    monkeypatch.setattr(e, "_cuda_mem_get_info",
                        lambda i: (8 * 1024 ** 3, 24 * 1024 ** 3))
    assert e.free_vram_gb(0) == pytest.approx(8.0)


def test_reclaim_gpu_does_not_mask_a_poisoned_cuda_context(monkeypatch):
    class Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def empty_cache():
            raise RuntimeError("unspecified launch failure")

        @staticmethod
        def ipc_collect():
            raise AssertionError("must not reach the next cleanup call")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=Cuda()))
    with pytest.warns(RuntimeWarning, match="cleanup skipped"):
        env.reclaim_gpu()


def test_exit_now_flushes_buffered_stdout_before_exiting(tmp_path):
    """A skipped flush would lose the very line that proves success.

    exit_now bypasses interpreter teardown, which also bypasses the implicit
    flush at shutdown. stdout is block-buffered when it is not a tty -- which
    is exactly how the campaign runs it -- so the flush has to be explicit.
    """
    import subprocess
    import sys as _sys
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parent.parent
    src = (f"import sys; sys.path.insert(0, {str(root)!r})\n"
           "from ragregen import env\n"
           # No newline: nothing forces this out of the buffer on its own.
           "print('[index] committed', end='')\n"
           "env.exit_now(0)\n")
    got = subprocess.run([_sys.executable, "-c", src],
                         capture_output=True, text=True, timeout=120)

    assert got.returncode == 0
    assert "[index] committed" in got.stdout, (
        f"buffered stdout was lost on exit: {got.stdout!r}")


def test_exit_now_propagates_the_exit_code(tmp_path):
    """supervise.sh reads the code: 2 means evicted, anything else is fatal."""
    import subprocess
    import sys as _sys
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parent.parent
    src = (f"import sys; sys.path.insert(0, {str(root)!r})\n"
           "from ragregen import env\n"
           "env.exit_now(2)\n")
    got = subprocess.run([_sys.executable, "-c", src],
                         capture_output=True, text=True, timeout=120)

    assert got.returncode == 2

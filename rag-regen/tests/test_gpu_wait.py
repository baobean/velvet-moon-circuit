"""The GPU acquisition gate, driven with a fake nvidia-smi. No GPU.

gpu_wait.sh blocks until the shared card has enough free VRAM. The subtlety
these tests pin down is that *waiting must be audible*: campaign.sh's watchdog
treats a stage whose log stops growing as hung and kills it, so a gate that
waits silently through a long neighbour job gets killed for doing its job.
"""
import os
import subprocess
from pathlib import Path

GPU_WAIT = Path("scripts/gpu_wait.sh").resolve()


def _fake_nvidia_smi(tmp_path, free_mib):
    """Put a fake nvidia-smi on PATH that reports `free_mib` MiB free.

    `free_mib` may be a list, one value per successive call, so a test can
    make the card fill up or free up over time.
    """
    binder = tmp_path / "bin"
    binder.mkdir(exist_ok=True)
    counter = tmp_path / "calls"
    values = free_mib if isinstance(free_mib, list) else [free_mib]
    smi = binder / "nvidia-smi"
    smi.write_text(
        "#!/usr/bin/env bash\n"
        f'vals=({" ".join(str(v) for v in values)})\n'
        f'n=$(cat "{counter}" 2>/dev/null || echo 0)\n'
        f'echo $((n + 1)) > "{counter}"\n'
        # Clamp to the last value so the sequence holds after it runs out.
        'idx=$n; [ "$idx" -ge "${#vals[@]}" ] && idx=$(( ${#vals[@]} - 1 ))\n'
        'echo "${vals[$idx]}"\n')
    smi.chmod(0o755)
    return binder


def _run(tmp_path, free_mib, *, need=8, timeout=4, interval=1, stable=1,
         command=("true",)):
    env = dict(os.environ)
    env["PATH"] = f"{_fake_nvidia_smi(tmp_path, free_mib)}:{env['PATH']}"
    return subprocess.run(
        [str(GPU_WAIT), "--need", str(need), "--interval", str(interval),
         "--stable", str(stable), "--timeout", str(timeout),
         "--log", str(tmp_path / "gpu_wait.log"), "--", *command],
        capture_output=True, text=True, timeout=90, env=env)


def test_waiting_for_a_busy_card_emits_a_heartbeat_every_poll(tmp_path):
    """A silent wait is indistinguishable from a hang.

    Regression test for the build_index stall of 2026-08-07. The neighbour
    held 19.6 GB for over nine hours; gpu_wait polled correctly but printed
    nothing while the card stayed busy, so campaign.log stopped growing and
    run_watched's 5400s stall guard would have killed a healthy gate three
    times over and marked the stage FAILED.
    """
    got = _run(tmp_path, 1024, need=8, interval=1, timeout=3)

    assert got.returncode == 4, "should time out, not acquire a busy card"
    beats = [ln for ln in got.stdout.splitlines() if "1024" in ln]
    assert len(beats) >= 2, (
        "gpu_wait printed no per-poll heartbeat while waiting; the watchdog "
        f"will read this as a stall. stdout was:\n{got.stdout}")


def test_heartbeat_reports_free_and_required_vram(tmp_path):
    """The heartbeat has to carry the numbers, or it cannot be diagnosed."""
    got = _run(tmp_path, 1024, need=8, interval=1, timeout=2)

    waiting = [ln for ln in got.stdout.splitlines() if "1024" in ln]
    assert waiting, f"no heartbeat at all:\n{got.stdout}"
    assert any("8192" in ln for ln in waiting), (
        f"heartbeat omits the requirement, so 'is 1024 nearly enough?' is "
        f"unanswerable from the log:\n{got.stdout}")


def test_a_free_card_is_still_acquired(tmp_path):
    """The heartbeat must not disturb the acquire path."""
    marker = tmp_path / "ran"
    got = _run(tmp_path, 20480, need=8, stable=2, interval=1, timeout=30,
               command=("touch", str(marker)))

    assert got.returncode == 0, got.stdout + got.stderr
    assert marker.exists(), "command never ran despite a free card"


def test_a_card_that_frees_up_is_eventually_acquired(tmp_path):
    """Busy, busy, then free -- the wait ends when the neighbour leaves."""
    marker = tmp_path / "ran"
    got = _run(tmp_path, [1024, 1024, 20480, 20480], need=8, stable=2,
               interval=1, timeout=30, command=("touch", str(marker)))

    assert got.returncode == 0, got.stdout + got.stderr
    assert marker.exists()


def test_supervise_forwards_the_gpu_index_to_gpu_wait(tmp_path):
    """On a mixed-GPU box the index is the difference between running and never.

    The pipeline stages ask for 17 GB. On a 2x16GB + 1x24GB machine only one
    card can ever satisfy that, so a hardcoded index 0 waits forever on
    hardware that physically cannot fit the job.
    """
    import subprocess

    root = Path(__file__).resolve().parent.parent
    # A fake gpu_wait that just records the arguments it was handed.
    shim = tmp_path / "scripts"
    shim.mkdir()
    seen = tmp_path / "seen.txt"
    (shim / "gpu_wait.sh").write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" > "{seen}"\n'
        "exit 0\n")
    (shim / "gpu_wait.sh").chmod(0o755)
    (shim / "supervise.sh").write_text(
        (root / "scripts" / "supervise.sh").read_text())
    (shim / "supervise.sh").chmod(0o755)

    subprocess.run(
        [str(shim / "supervise.sh"), "--gpu", "2", "--need", "17",
         "--log", str(tmp_path / "s.log"), "--", "true"],
        capture_output=True, text=True, timeout=60, cwd=tmp_path)

    assert seen.exists(), "supervise never invoked gpu_wait"
    got = seen.read_text()
    assert "--gpu 2" in got, f"gpu index not forwarded: {got!r}"

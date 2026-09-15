"""The supervisor loop, driven with fake commands. No GPU, no pipeline.

Exit 2 is StageAborted -- the card was taken, the queue is checkpointed, and
resuming is correct. Every other non-zero code is a real failure and must not
be retried forever.
"""
import subprocess
from pathlib import Path

SUPERVISE = Path("scripts/supervise.sh").resolve()


def _run(tmp_path, script_body, *, max_retries=3, gpu_wait=False):
    fake = tmp_path / "fake.sh"
    fake.write_text("#!/usr/bin/env bash\n" + script_body)
    fake.chmod(0o755)
    args = [str(SUPERVISE), "--max-retries", str(max_retries),
             "--interval", "0"]
    if not gpu_wait:
        args.append("--no-gpu-wait")
    args += [
        #: Point the log inside tmp_path so the suite never writes into the
        #: repo's untracked (non-gitignored) logs/ directory.
        "--log", str(tmp_path / "supervise.log"),
        "--", str(fake),
    ]
    return subprocess.run(args, capture_output=True, text=True, timeout=60)


def test_a_clean_exit_stops_immediately(tmp_path):
    got = _run(tmp_path, 'echo run >> "$0.log"\nexit 0\n')
    assert got.returncode == 0
    assert Path(f"{tmp_path / 'fake.sh'}.log").read_text().count("run") == 1


def test_exit_two_is_retried(tmp_path):
    #: The counter file makes the fake fail twice then succeed, which is what
    #: a contended card looks like.
    got = _run(tmp_path, f'''
n=$(cat "{tmp_path}/n" 2>/dev/null || echo 0)
echo $((n + 1)) > "{tmp_path}/n"
[ "$n" -ge 2 ] && exit 0
exit 2
''')
    assert got.returncode == 0
    assert (tmp_path / "n").read_text().strip() == "3"


def test_a_real_failure_is_not_retried(tmp_path):
    #: Exit 1 is a bug, not an eviction. Retrying it would loop on a
    #: deterministic crash and burn the whole booked block.
    got = _run(tmp_path, f'''
n=$(cat "{tmp_path}/n" 2>/dev/null || echo 0)
echo $((n + 1)) > "{tmp_path}/n"
exit 1
''')
    assert got.returncode == 1
    assert (tmp_path / "n").read_text().strip() == "1"


def test_retries_are_bounded(tmp_path):
    got = _run(tmp_path, "exit 2\n", max_retries=2)
    assert got.returncode != 0
    assert "max-retries" in (got.stdout + got.stderr)


def test_missing_gpu_wait_fails_clearly(tmp_path, monkeypatch):
    #: Run from a copy of the repo root with scripts/gpu_wait.sh absent, so
    #: the preflight check fires regardless of whether the real
    #: scripts/gpu_wait.sh happens to exist on this machine.
    fake_root = tmp_path / "fake_root"
    (fake_root / "scripts").mkdir(parents=True)
    supervise_src = SUPERVISE.read_text()
    fake_supervise = fake_root / "scripts" / "supervise.sh"
    fake_supervise.write_text(supervise_src)
    fake_supervise.chmod(0o755)
    # No scripts/gpu_wait.sh is created here -- that's the point.

    fake_cmd = fake_root / "cmd.sh"
    fake_cmd.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_cmd.chmod(0o755)

    got = subprocess.run(
        [str(fake_supervise), "--interval", "0",
         "--log", str(tmp_path / "supervise.log"),
         "--", str(fake_cmd)],
        capture_output=True, text=True, timeout=60)
    assert got.returncode == 1
    assert "gpu_wait.sh" in (got.stdout + got.stderr)

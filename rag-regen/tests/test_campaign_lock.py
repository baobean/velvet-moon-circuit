"""The campaign's cross-machine ownership guard.

logs/campaign/ and outputs/ live on shared NAS storage, so a second machine
sees this campaign's markers. Two drivers would both claim the same stage,
both write index.faiss, and interleave into one campaign.log. Ownership is a
heartbeat file because there is no cross-host PID to check.

These run against a COPY of campaign.sh in tmp_path -- the script derives its
state directory from its own location, so a copy keeps the real campaign's
markers untouched.
"""
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _sandbox(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    dst = scripts / "campaign.sh"
    dst.write_text((REPO / "scripts" / "campaign.sh").read_text())
    dst.chmod(0o755)
    (tmp_path / "logs" / "campaign").mkdir(parents=True)
    return dst


def _run(script, *args, timeout=60):
    return subprocess.run([str(script), *args], capture_output=True,
                          text=True, timeout=timeout)


def test_refuses_to_start_when_another_host_holds_a_fresh_claim(tmp_path):
    script = _sandbox(tmp_path)
    owner = tmp_path / "logs" / "campaign" / "OWNER"
    owner.write_text("some-other-box 4242 2026-08-08T10:00:00\n")

    got = _run(script)

    assert got.returncode == 9, (
        f"expected refusal, got {got.returncode}:\n{got.stdout}{got.stderr}")
    assert "some-other-box" in got.stdout
    assert "REFUSING TO START" in got.stdout


def test_takes_over_a_stale_claim(tmp_path):
    """A crashed driver on a dead machine must not block the campaign forever."""
    script = _sandbox(tmp_path)
    owner = tmp_path / "logs" / "campaign" / "OWNER"
    owner.write_text("some-other-box 4242 2026-08-08T10:00:00\n")
    # STALE_AFTER is 300s; backdate well past it.
    old = time.time() - 3600
    os.utime(owner, (old, old))

    got = _run(script)

    assert got.returncode != 9, "a stale claim must not block startup"
    assert "taking over from 'some-other-box'" in got.stdout


def test_status_never_takes_the_lock(tmp_path):
    """Checking on a run from a second machine must stay harmless."""
    script = _sandbox(tmp_path)

    got = _run(script, "status")

    assert got.returncode == 0
    assert not (tmp_path / "logs" / "campaign" / "OWNER").exists(), (
        "status claimed ownership; it must be read-only")

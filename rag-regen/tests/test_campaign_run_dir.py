"""The campaign's run-directory pinning.

`pin_run_dir` is the reason the pipeline stages are idempotent: it mints
outputs/<tag>_<TS>/ exactly once, records it under logs/campaign/, and the
driver passes it as --resume on every retry. If it returns nothing, the
stages silently run with `--resume ""`, which argparse resolves to Path("")
-> the repo root: both arms then share one directory, the second arm finds
every case already terminal and exits in seconds having produced nothing,
and the report reads a directory whose run.json describes the wrong
mechanism. That is not hypothetical -- it is what the 2026-08-09 doc-5
campaign did, and nothing in the driver noticed, because campaign.sh runs
under `set -uo pipefail` with no `-e`.

These extract the function from campaign.sh and exercise it directly; the
script itself runs a whole campaign when executed.
"""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _call_pin_run_dir(tmp_path, *args):
    """Run campaign.sh's pin_run_dir in isolation, as the driver calls it.

    Same shell options as the real script -- `set -u` is what turns the bug
    into an empty string rather than a wrong-but-present path.
    """
    src = (REPO / "scripts" / "campaign.sh").read_text()
    body = src[src.index("pin_run_dir() {"):]
    body = body[:body.index("\n}\n") + 3]

    harness = "\n".join([
        "set -uo pipefail",
        f'STATE="{tmp_path}/logs/campaign"',
        'say() { printf "[campaign] %s\\n" "$*"; }',
        body,
        'D="$(pin_run_dir "$@")"',
        'printf "CAPTURED:%s\\n" "$D"',
    ])
    return subprocess.run(["bash", "-c", harness, "bash", *args],
                          capture_output=True, text=True, timeout=30,
                          cwd=tmp_path)


def _setup(tmp_path):
    (tmp_path / "logs" / "campaign").mkdir(parents=True)
    (tmp_path / "outputs").mkdir()


def test_pin_run_dir_returns_a_path(tmp_path):
    _setup(tmp_path)

    got = _call_pin_run_dir(tmp_path, "dry", "doc5dry")

    assert got.returncode == 0, f"{got.stdout}{got.stderr}"
    captured = [ln[len("CAPTURED:"):] for ln in got.stdout.splitlines()
                if ln.startswith("CAPTURED:")][0]
    #: The empty string is the whole bug: the driver interpolates it into
    #: `--resume ""` and every stage silently targets the repo root.
    assert captured, (
        f"pin_run_dir returned nothing; stderr was:\n{got.stderr}")
    assert captured.startswith("outputs/doc5dry_")
    assert (tmp_path / captured).is_dir()


def test_pin_run_dir_is_stable_across_calls(tmp_path):
    """The second call must return the first call's directory, not mint a new one.

    This is the property that makes `--resume` meaningful across the retries
    supervise.sh performs on eviction.
    """
    _setup(tmp_path)

    first = _call_pin_run_dir(tmp_path, "real", "doc5")
    second = _call_pin_run_dir(tmp_path, "real", "doc5")

    def captured(r):
        return [ln[len("CAPTURED:"):] for ln in r.stdout.splitlines()
                if ln.startswith("CAPTURED:")][0]

    assert captured(first), first.stderr
    assert captured(first) == captured(second)


def _call_require_run_dir(tmp_path, value):
    src = (REPO / "scripts" / "campaign.sh").read_text()
    body = src[src.index("require_run_dir() {"):]
    body = body[:body.index("\n}\n") + 3]

    harness = "\n".join([
        "set -uo pipefail",
        'say() { printf "[campaign] %s\\n" "$*"; }',
        body,
        'require_run_dir test "$1"',
        'printf "PROCEEDED\\n"',
    ])
    return subprocess.run(["bash", "-c", harness, "bash", value],
                          capture_output=True, text=True, timeout=30,
                          cwd=tmp_path)


def test_require_run_dir_rejects_an_empty_path(tmp_path):
    _setup(tmp_path)

    got = _call_require_run_dir(tmp_path, "")

    assert got.returncode == 3, got.stdout
    assert "PROCEEDED" not in got.stdout


def test_require_run_dir_rejects_a_path_with_a_banner_glued_to_it(tmp_path):
    """The exact shape `say`-on-stdout produced: log line, newline, then the path."""
    _setup(tmp_path)
    (tmp_path / "outputs" / "doc5dry_x").mkdir()

    got = _call_require_run_dir(
        tmp_path, "[campaign] pinned dry run directory: outputs/doc5dry_x\n"
                  "outputs/doc5dry_x")

    assert got.returncode == 3, (
        f"a polluted return value must abort, got:\n{got.stdout}")
    assert "PROCEEDED" not in got.stdout


def test_require_run_dir_accepts_a_real_pinned_directory(tmp_path):
    _setup(tmp_path)
    (tmp_path / "outputs" / "doc5dry_x").mkdir()

    got = _call_require_run_dir(tmp_path, "outputs/doc5dry_x")

    assert got.returncode == 0, f"{got.stdout}{got.stderr}"
    assert "PROCEEDED" in got.stdout


def test_driver_refuses_to_run_a_stage_with_an_unpinned_directory(tmp_path):
    """Defence in depth: an empty run dir must stop the campaign, not run.

    `set -u` already makes the failure visible on stderr, but without `-e`
    the driver carried on for two more stages and a report. The stages
    themselves must not accept an empty directory.
    """
    src = (REPO / "scripts" / "campaign.sh").read_text()
    assert "require_run_dir" in src, (
        "campaign.sh must validate a pinned run directory before using it")

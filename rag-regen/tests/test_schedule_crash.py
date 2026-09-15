"""Crash safety, proved by killing a real process.

A test that calls save() then open() in one process proves serialisation and
nothing else. Design §R3: if the suite cannot survive a SIGKILL, the resume
guarantee is decorative.
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

from ragregen.schedule import Queue

WRITER = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, {root!r})
    from ragregen.schedule import Queue

    q = Queue.open({path!r}, ["c0","c1","c2","c3","c4"], retry_budget=3,
                   screen_run="s")
    for i in range(5):
        q.mark(f"c{{i}}", "regen", "done", attempt=1)
        print(i, flush=True)
        time.sleep(0.4)
""")


def test_sigkill_midstage_keeps_completed_cases(tmp_path):
    root = str(Path(__file__).resolve().parent.parent)
    path = tmp_path / "q.json"
    script = tmp_path / "writer.py"
    script.write_text(WRITER.format(root=root, path=str(path)))

    proc = subprocess.Popen([sys.executable, str(script)],
                            stdout=subprocess.PIPE, text=True)
    #: Wait for case 2 to report done, then kill hard -- no cleanup, no
    #: atexit, no flush. This is what a stolen GPU looks like to the process.
    for _ in range(3):
        proc.stdout.readline()
    proc.kill()
    proc.wait(timeout=10)

    data = json.loads(path.read_text())          # must parse at all
    done = {c for c, s in data["cases"].items()
            if s["stages"].get("regen@1") == "done"}
    assert {"c0", "c1", "c2"} <= done
    assert "c4" not in done


def test_no_tmp_file_is_left_behind(tmp_path):
    q = Queue.open(tmp_path / "q.json", ["a"], retry_budget=3, screen_run="s")
    q.save()

    assert [p.name for p in tmp_path.iterdir()] == ["q.json"]

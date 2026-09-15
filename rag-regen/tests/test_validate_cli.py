"""The exit-code contract of the preflight CLI.

run.sh dispatches with `exec`, so this return value is the operator's (and any
wrapping script's) only signal. 0 must mean "warnings are allowed", 2 must mean
"nothing downstream may run".
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragregen.validate import Problem  # noqa: E402
from scripts import validate_cli  # noqa: E402


@pytest.fixture
def cli(monkeypatch):
    """validate_cli with its config loading and argv stubbed out."""
    monkeypatch.setattr(sys, "argv", ["validate_cli.py"])
    monkeypatch.setattr(validate_cli.config, "load_dataset",
                        lambda p: type("DS", (), {"cases": [1, 2, 3]})())
    monkeypatch.setattr(validate_cli.config, "load_retrieval_db",
                        lambda p: type("DB", (), {"images_root": "/corpus"})())
    monkeypatch.setattr(validate_cli.config, "load_pipeline", lambda p: object())
    return monkeypatch


def test_clean_configs_exit_zero(cli):
    cli.setattr(validate_cli.validate, "validate_all", lambda *a: [])
    assert validate_cli.main() == 0


def test_warnings_alone_still_exit_zero(cli):
    cli.setattr(validate_cli.validate, "validate_all",
                lambda *a: [Problem("warning", "disk_tight", "76 GB free")])
    assert validate_cli.main() == 0


def test_any_error_exits_two(cli):
    cli.setattr(validate_cli.validate, "validate_all",
                lambda *a: [Problem("warning", "disk_tight", "76 GB free"),
                            Problem("error", "gt_ref_missing", "nope")])
    assert validate_cli.main() == 2


@pytest.mark.parametrize("exc", [FileNotFoundError("no such file"),
                                 ValueError("malformed yaml")])
def test_unloadable_config_exits_two_without_traceback(cli, exc, capsys):
    def boom(_):
        raise exc

    cli.setattr(validate_cli.config, "load_dataset", boom)
    assert validate_cli.main() == 2
    assert "[error] config:" in capsys.readouterr().out

"""Tests for the shared CLI error-handling helpers."""

import inspect
from pathlib import Path
from unittest.mock import patch

import pytest
import typer

from hfutils.cli_support import cli_guard, fail, verify_and_report
from hfutils.errors import StreamMergeError


class TestFail:
    def test_prints_error_and_exits_1_by_default(self, capsys):
        with pytest.raises(typer.Exit) as exc_info:
            fail("something broke")

        assert exc_info.value.exit_code == 1
        out = capsys.readouterr().out
        assert "Error:" in out
        assert "something broke" in out

    def test_custom_exit_code(self):
        with pytest.raises(typer.Exit) as exc_info:
            fail("verify mismatch", code=2)

        assert exc_info.value.exit_code == 2


class TestCliGuard:
    def test_catches_hfutils_error(self, capsys):
        @cli_guard
        def cmd():
            raise StreamMergeError("merge broke")

        with pytest.raises(typer.Exit) as exc_info:
            cmd()

        assert exc_info.value.exit_code == 1
        assert "merge broke" in capsys.readouterr().out

    def test_catches_oserror(self):
        @cli_guard
        def cmd():
            raise OSError("disk full")

        with pytest.raises(typer.Exit) as exc_info:
            cmd()

        assert exc_info.value.exit_code == 1

    def test_does_not_catch_plain_valueerror(self):
        """Negative-space test: cli_guard is not a blanket `except Exception`
        -- a real bug (a bare ValueError from unrelated code) must still
        surface as a traceback, not get rewritten into a clean user error."""
        @cli_guard
        def cmd():
            raise ValueError("not an hfutils condition")

        with pytest.raises(ValueError):
            cmd()

    def test_does_not_catch_plain_runtimeerror(self):
        @cli_guard
        def cmd():
            raise RuntimeError("real bug")

        with pytest.raises(RuntimeError):
            cmd()

    def test_does_not_rewrite_typer_exit(self):
        """lora convert --verify's exit code 2 must pass through untouched --
        cli_guard must not intercept typer.Exit at all."""
        @cli_guard
        def cmd():
            raise typer.Exit(2)

        with pytest.raises(typer.Exit) as exc_info:
            cmd()

        assert exc_info.value.exit_code == 2

    def test_preserves_function_signature_for_typer(self):
        """functools.wraps must be used -- Typer inspects the wrapped
        function's signature to build CLI options; losing it breaks every
        command's flags."""
        @cli_guard
        def cmd(name: str, count: int = 1):
            return name, count

        sig = inspect.signature(cmd)
        assert list(sig.parameters) == ["name", "count"]

    def test_returns_wrapped_functions_result_on_success(self):
        @cli_guard
        def cmd():
            return 42

        assert cmd() == 42


class TestVerifyAndReport:
    def test_returns_true_and_prints_verified_on_match(self, capsys):
        with patch("hfutils.cli_support.verify_output", return_value=(True, None)):
            result = verify_and_report(Path("out.safetensors"), {})

        assert result is True
        assert "Verified" in capsys.readouterr().out

    def test_returns_false_and_prints_error_on_mismatch(self, capsys):
        with patch("hfutils.cli_support.verify_output", return_value=(False, "missing tensor")):
            result = verify_and_report(Path("out.safetensors"), {})

        assert result is False
        out = capsys.readouterr().out
        assert "Error" in out
        assert "missing tensor" in out

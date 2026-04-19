"""hfutils -- local model file toolkit."""

import typer
from rich.console import Console

from hfutils import __version__

_console = Console()


def _version_callback(value: bool) -> None:
    if value:
        _console.print(__version__)
        raise typer.Exit()


app = typer.Typer(
    name="hfutils",
    help="Local model file toolkit: inspect files/trees, convert between layouts.",
    no_args_is_help=True,
)


@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version",
        callback=_version_callback, is_eager=True,
        help="Show the hfutils version and exit.",
    ),
) -> None:
    """Root callback -- handles --version, then defers to sub-commands."""


def _register_commands() -> None:
    from hfutils.commands import inspect  # noqa: F401  -- registers @app.command
    from hfutils.commands.civitai import civitai_app
    from hfutils.commands.convert import convert_app

    app.add_typer(civitai_app, name="civitai")
    app.add_typer(convert_app, name="convert")


_register_commands()

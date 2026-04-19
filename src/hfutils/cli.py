"""hfutils -- local model file toolkit."""

import typer

app = typer.Typer(
    name="hfutils",
    help="Local model file toolkit: inspect files/trees, convert between layouts.",
    no_args_is_help=True,
)


def _register_commands() -> None:
    from hfutils.commands import inspect  # noqa: F401  -- registers @app.command
    from hfutils.commands.civitai import civitai_app
    from hfutils.commands.convert import convert_app

    app.add_typer(civitai_app, name="civitai")
    app.add_typer(convert_app, name="convert")


_register_commands()

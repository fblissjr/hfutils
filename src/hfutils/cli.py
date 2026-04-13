"""hfutils -- local model file toolkit."""

import typer

app = typer.Typer(
    name="hfutils",
    help="Local model file toolkit: inspect, merge, and scan safetensors/GGUF files.",
    no_args_is_help=True,
)


def _register_commands() -> None:
    from hfutils.commands import inspect_cmd, merge, scan  # noqa: F401


_register_commands()

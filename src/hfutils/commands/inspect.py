"""`hfutils inspect` command: thin CLI wrapper over views + walker."""

from pathlib import Path

import typer
from rich.console import Console

from hfutils.cli import app
from hfutils.inspect.views import display_source, display_tree
from hfutils.inspect.walker import walk_for_models
from hfutils.sources.detect import DetectLevel, detect_source

console = Console()


@app.command()
def inspect(
    path: Path = typer.Argument(..., help="File, component dir, diffusers pipeline, or directory tree"),
    detail: bool = typer.Option(False, "--detail", "-d", help="Show architecture detection and full tensor list"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Walk a directory tree (replaces the old `scan` command)"),
) -> None:
    """Inspect model file headers and directory layouts."""
    if not path.exists():
        console.print(f"[red]Error:[/red] {path} not found")
        raise typer.Exit(1)

    if recursive:
        if not path.is_dir():
            console.print("[red]Error:[/red] --recursive requires a directory")
            raise typer.Exit(1)
        entries = walk_for_models(path)
        if not entries:
            console.print("No models found.")
            return
        display_tree(path, entries, console)
        return

    display_source(detect_source(path, DetectLevel.FULL), detail, console)

"""CivitAI commands: search, info, download."""

from pathlib import Path

import orjson
import typer
from rich.console import Console
from rich.table import Table

from hfutils.providers.civitai import CivitaiClient, DownloadInfo, parse_model_ref, primary_file
from hfutils.providers.download import download_file
from hfutils.inspect.common import format_size

civitai_app = typer.Typer(
    name="civitai",
    help="Search, inspect, and download models from CivitAI.",
    no_args_is_help=True,
)

console = Console()


@civitai_app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
) -> None:
    """Search CivitAI for models."""
    client = CivitaiClient()
    try:
        results = client.search(query, limit=limit)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if not results:
        console.print("No results found.")
        return

    table = Table(title=f"CivitAI: '{query}'")
    table.add_column("ID", justify="right", style="cyan")
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Creator")

    for item in results:
        table.add_row(
            str(item["id"]),
            item["name"],
            item.get("type", ""),
            item.get("creator", {}).get("username", ""),
        )

    console.print(table)


@civitai_app.command()
def info(
    target: str = typer.Argument(..., help="Model ID, URL, or AIR URN"),
) -> None:
    """Show model details and available versions."""
    model_id = parse_model_ref(target)
    if model_id is None:
        console.print(f"[red]Error:[/red] Could not parse model reference: {target}")
        raise typer.Exit(1)

    client = CivitaiClient()
    try:
        model = client.get_model(model_id)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"\n[bold]{model['name']}[/bold]")
    console.print(f"  ID: {model['id']}")
    if model.get("type"):
        console.print(f"  Type: {model['type']}")
    if model.get("creator", {}).get("username"):
        console.print(f"  Creator: {model['creator']['username']}")

    versions = model.get("modelVersions", [])
    if versions:
        console.print(f"\n  [bold]Versions ({len(versions)}):[/bold]")
        for i, v in enumerate(versions):
            pf = primary_file(v.get("files", []))
            size_str = format_size(int(pf.get("sizeKB", 0)) * 1024) if pf else "?"
            console.print(f"    [{i}] {v['name']}  ({size_str})")


@civitai_app.command()
def dl(
    target: str = typer.Argument(..., help="Model ID, URL, or AIR URN"),
    output: Path = typer.Option(".", "--output", "-o", help="Output directory"),
    version: int = typer.Option(0, "--version", "-v", help="Version index (0 = latest)"),
) -> None:
    """Download a model from CivitAI."""
    model_id = parse_model_ref(target)
    if model_id is None:
        console.print(f"[red]Error:[/red] Could not parse model reference: {target}")
        raise typer.Exit(1)

    client = CivitaiClient()
    try:
        dl_info = client.resolve_download(model_id, version_idx=version)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    output.mkdir(parents=True, exist_ok=True)
    dest = output / dl_info.filename

    console.print(f"\n  Model:    {dl_info.model_name}")
    console.print(f"  Version:  {dl_info.version_name}")
    console.print(f"  File:     {dl_info.filename}")
    console.print(f"  Size:     {format_size(dl_info.size_bytes)}")
    console.print(f"  Dest:     {dest}")

    if not typer.confirm("\nProceed?", default=True):
        raise typer.Abort()

    download_file(dl_info.url, dest, total_size=dl_info.size_bytes, headers=client.auth_headers)
    _write_sidecar(dl_info, dest)


def _write_sidecar(info: DownloadInfo, dest: Path) -> None:
    """Write a <file>.civitai.json sidecar with usage metadata next to the download."""
    sidecar = dest.with_name(dest.name + ".civitai.json")
    payload = {
        "model_id": info.model_id,
        "version_id": info.version_id,
        "model_name": info.model_name,
        "version_name": info.version_name,
        "base_model": info.base_model,
        "trained_words": info.trained_words,
        "description": info.description,
    }
    sidecar.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    if info.trained_words:
        console.print(f"  Triggers: {', '.join(info.trained_words)}")
    console.print(f"  Sidecar:  {sidecar}")

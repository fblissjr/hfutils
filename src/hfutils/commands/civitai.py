"""CivitAI commands: search, info, download."""

from pathlib import Path

import orjson
import typer
from rich.console import Console
from rich.table import Table

from hfutils.providers.civitai import (
    CivitaiClient,
    DownloadInfo,
    ModelRef,
    parse_model_ref,
    primary_file,
)
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


def _resolve_target(target: str, host: str | None) -> tuple[ModelRef, CivitaiClient]:
    """Parse `target` and build a client, exiting cleanly on bad input."""
    ref = parse_model_ref(target)
    if ref is None:
        console.print(f"[red]Error:[/red] Could not parse model reference: {target}")
        raise typer.Exit(1)
    return ref, CivitaiClient(host=host or ref.host)


def _api_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@civitai_app.command()
def info(
    target: str = typer.Argument(..., help="Model ID, URL (civitai.com/.red), or AIR URN"),
    host: str | None = typer.Option(None, "--host", help="API host override (e.g. civitai.com, civitai.red)"),
) -> None:
    """Show model details and available versions."""
    ref, client = _resolve_target(target, host)
    model = _api_call(client.get_model, ref.model_id)

    console.print(f"\n[bold]{model['name']}[/bold]")
    console.print(f"  ID: {model['id']}")
    if model.get("type"):
        console.print(f"  Type: {model['type']}")
    if model.get("creator", {}).get("username"):
        console.print(f"  Creator: {model['creator']['username']}")
    console.print(f"  Host: {client.host}")

    versions = model.get("modelVersions", [])
    if versions:
        console.print(f"\n  [bold]Versions ({len(versions)}):[/bold]")
        for v in versions:
            pf = primary_file(v.get("files", []))
            size_str = format_size(int(pf.get("sizeKB", 0)) * 1024) if pf else "?"
            marker = " [yellow](selected)[/yellow]" if ref.version_id == v.get("id") else ""
            console.print(f"    id={v['id']}  {v['name']}  ({size_str}){marker}")


@civitai_app.command()
def dl(
    target: str = typer.Argument(..., help="Model ID, URL (civitai.com/.red), or AIR URN"),
    output: Path = typer.Option(".", "--output", "-o", help="Output directory"),
    version: int | None = typer.Option(
        None,
        "--version",
        "-v",
        help="Specific version ID (overrides URL/AIR; default: latest)",
    ),
    host: str | None = typer.Option(None, "--host", help="API host override (e.g. civitai.com, civitai.red)"),
) -> None:
    """Download a model from CivitAI."""
    ref, client = _resolve_target(target, host)
    effective_version = version if version is not None else ref.version_id
    dl_info = _api_call(client.resolve_download, ref.model_id, version_id=effective_version)

    output.mkdir(parents=True, exist_ok=True)
    dest = output / dl_info.filename

    console.print(f"\n  Model:    {dl_info.model_name}")
    console.print(f"  Version:  {dl_info.version_name} (id={dl_info.version_id})")
    console.print(f"  File:     {dl_info.filename}")
    console.print(f"  Size:     {format_size(dl_info.size_bytes)}")
    console.print(f"  Host:     {client.host}")
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

"""Shared download logic with resume support and rich progress."""

import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from rich.console import Console

from hfutils.io.progress import make_progress

console = Console()

# Realistic User-Agent needed by some providers (CivitAI blocks default urllib agent)
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


def get_file_metadata(url: str, headers: dict | None = None) -> tuple[int, str]:
    """Get file size and filename via a HEAD request."""
    hdrs = {**DEFAULT_HEADERS, **(headers or {})}
    req = urllib.request.Request(url, method="HEAD", headers=hdrs)

    try:
        with urllib.request.urlopen(req) as resp:
            size = int(resp.headers.get("Content-Length", 0))
            cd = resp.headers.get("Content-Disposition", "")
            filename = ""
            if "filename=" in cd:
                match = re.search(r'filename="?([^";]+)"?', cd)
                if match:
                    filename = match.group(1)
            if not filename:
                filename = Path(urllib.parse.urlparse(resp.url).path).name
            return size, filename
    except Exception:
        return 0, ""


def download_file(
    url: str,
    dest: Path,
    total_size: int,
    headers: dict | None = None,
    show_progress: bool = True,
) -> bool:
    """Download a file with resume support and rich progress bar.

    Returns True on success, False on failure.
    """
    dest = Path(dest)
    hdrs = {**DEFAULT_HEADERS, **(headers or {})}
    req = urllib.request.Request(url, headers=hdrs)

    mode = "wb"
    start_byte = 0

    if dest.exists():
        start_byte = dest.stat().st_size
        if start_byte >= total_size and total_size > 0:
            console.print(f"Already downloaded: {dest.name}")
            return True
        if start_byte > 0:
            req.add_header("Range", f"bytes={start_byte}-")
            mode = "ab"
            console.print(f"Resuming from {start_byte / (1 << 20):.1f} MB...")

    try:
        with urllib.request.urlopen(req) as resp, open(dest, mode) as f:
            if show_progress and total_size > 0:
                with make_progress(console) as progress:
                    task = progress.add_task("Downloading", total=total_size, completed=start_byte)
                    while chunk := resp.read(256 * 1024):
                        f.write(chunk)
                        progress.advance(task, len(chunk))
            else:
                while chunk := resp.read(256 * 1024):
                    f.write(chunk)

        console.print("Download complete.")
        return True
    except urllib.error.HTTPError as e:
        console.print(f"[red]Download error:[/red] {e.code} {e.reason}")
        return False
    except Exception as e:
        console.print(f"[red]Download failed:[/red] {e}")
        return False

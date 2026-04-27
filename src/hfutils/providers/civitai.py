"""CivitAI API client."""

import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass

import orjson

from hfutils.providers.download import DEFAULT_HEADERS


@dataclass
class DownloadInfo:
    url: str
    filename: str
    size_bytes: int
    model_name: str
    version_name: str
    model_id: int
    version_id: int
    trained_words: list[str]
    base_model: str | None = None
    description: str | None = None


def primary_file(files: list[dict]) -> dict | None:
    """Select the primary file from a CivitAI version's file list."""
    return next((f for f in files if f.get("primary")), files[0] if files else None)


class CivitaiClient:
    BASE_URL = "https://civitai.com/api/v1"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("CIVITAI_API_KEY", "")

    @property
    def auth_headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    def _request(self, endpoint: str, params: dict[str, str] | None = None) -> dict:
        url = f"{self.BASE_URL}/{endpoint}"
        if params:
            url += "?" + urllib.parse.urlencode(params)

        req = urllib.request.Request(url, headers={**DEFAULT_HEADERS, **self.auth_headers})

        with urllib.request.urlopen(req) as resp:
            return orjson.loads(resp.read())

    def search(self, query: str, limit: int = 10) -> list[dict]:
        data = self._request("models", {"query": query, "limit": str(limit)})
        return data.get("items", [])

    def get_model(self, model_id: int) -> dict:
        return self._request(f"models/{model_id}")

    def resolve_download(self, model_id: int, version_idx: int = 0) -> DownloadInfo:
        """Get download info for a model. version_idx selects which version (0 = latest)."""
        model = self.get_model(model_id)
        versions = model.get("modelVersions", [])
        if not versions:
            msg = f"No versions found for model {model_id}"
            raise ValueError(msg)

        version = versions[version_idx]
        primary = primary_file(version.get("files", []))
        if not primary:
            msg = f"No files found for version {version['name']}"
            raise ValueError(msg)

        return DownloadInfo(
            url=version["downloadUrl"],
            filename=primary["name"],
            size_bytes=int(primary.get("sizeKB", 0)) * 1024,
            model_name=model["name"],
            version_name=version["name"],
            model_id=int(model["id"]),
            version_id=int(version["id"]),
            trained_words=list(version.get("trainedWords") or []),
            base_model=version.get("baseModel"),
            description=model.get("description"),
        )


def parse_model_ref(target: str) -> int | None:
    """Parse a model ID from various formats: numeric ID, AIR URN, or CivitAI URL."""
    if not target:
        return None
    if target.isdigit():
        return int(target)

    # AIR URN: civitai:12345
    match = re.search(r"civitai:(\d+)", target)
    if match:
        return int(match.group(1))

    # URL: https://civitai.com/models/12345/optional-slug
    match = re.search(r"civitai\.com/models/(\d+)", target)
    if match:
        return int(match.group(1))

    return None

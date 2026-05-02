"""CivitAI API client."""

import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass

import orjson

from hfutils.providers.download import DEFAULT_HEADERS

DEFAULT_HOST = "civitai.com"


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


@dataclass
class ModelRef:
    """Parsed reference to a CivitAI model, optionally with a specific version and host."""

    model_id: int
    version_id: int | None = None
    host: str = DEFAULT_HOST


def primary_file(files: list[dict]) -> dict | None:
    """Select the primary file from a CivitAI version's file list."""
    return next((f for f in files if f.get("primary")), files[0] if files else None)


class CivitaiClient:
    def __init__(self, api_key: str | None = None, host: str | None = None):
        self.api_key = api_key or os.environ.get("CIVITAI_API_KEY", "")
        self.host = host or os.environ.get("CIVITAI_HOST", DEFAULT_HOST)

    @property
    def base_url(self) -> str:
        return f"https://{self.host}/api/v1"

    @property
    def auth_headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    def _request(self, endpoint: str, params: dict[str, str] | None = None) -> dict:
        url = f"{self.base_url}/{endpoint}"
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

    def resolve_download(
        self,
        model_id: int,
        *,
        version_id: int | None = None,
    ) -> DownloadInfo:
        """Get download info for a model. If `version_id` is None, the latest version is used."""
        model = self.get_model(model_id)
        versions = model.get("modelVersions", [])
        if not versions:
            msg = f"No versions found for model {model_id}"
            raise ValueError(msg)

        if version_id is None:
            version = versions[0]
        else:
            version = next((v for v in versions if v.get("id") == version_id), None)
            if version is None:
                available = ", ".join(str(v.get("id")) for v in versions)
                msg = f"Version {version_id} not found for model {model_id}. Available: {available}"
                raise ValueError(msg)

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


_AIR_RE = re.compile(r"(?:^|:)civitai:(\d+)(?:@(\d+))?(?:$|[^\d])")
_URL_RE = re.compile(r"(civitai\.(?:com|red))/models/(\d+)")


def parse_model_ref(target: str) -> ModelRef | None:
    """Parse a model reference from numeric ID, AIR URN, or CivitAI URL.

    Supports civitai.com and civitai.red URLs, and AIR URNs of the form
    `civitai:<modelId>` or `civitai:<modelId>@<versionId>` (also the full
    `urn:air:<eco>:<type>:civitai:<modelId>@<versionId>` form).
    """
    if not target:
        return None
    if target.isdigit():
        return ModelRef(model_id=int(target))

    air = _AIR_RE.search(target)
    if air:
        ver = int(air.group(2)) if air.group(2) else None
        return ModelRef(model_id=int(air.group(1)), version_id=ver)

    url = _URL_RE.search(target)
    if url:
        host = url.group(1)
        model_id = int(url.group(2))
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(target).query)
        ver = qs.get("modelVersionId", [None])[0]
        version_id = int(ver) if ver else None
        return ModelRef(model_id=model_id, version_id=version_id, host=host)

    return None

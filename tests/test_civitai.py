"""Tests for CivitAI provider."""

from unittest.mock import patch, MagicMock

import orjson

from hfutils.providers.civitai import CivitaiClient, parse_model_ref, DownloadInfo


def _mock_response(data: dict) -> MagicMock:
    """Create a mock urllib response returning JSON."""
    body = orjson.dumps(data)
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestParseModelRef:
    def test_numeric_id(self):
        assert parse_model_ref("12345") == 12345

    def test_air_urn(self):
        assert parse_model_ref("civitai:67890") == 67890

    def test_civitai_url(self):
        assert parse_model_ref("https://civitai.com/models/12345/some-model-name") == 12345

    def test_civitai_url_with_version(self):
        assert parse_model_ref("https://civitai.com/models/12345?modelVersionId=99") == 12345

    def test_invalid_returns_none(self):
        assert parse_model_ref("not-a-model") is None

    def test_empty_returns_none(self):
        assert parse_model_ref("") is None


class TestCivitaiClient:
    def test_search(self):
        api_response = {
            "items": [
                {"id": 1, "name": "Test LoRA", "creator": {"username": "user1"}},
                {"id": 2, "name": "Another Model", "creator": {"username": "user2"}},
            ]
        }
        client = CivitaiClient(api_key="test-key")

        with patch("urllib.request.urlopen", return_value=_mock_response(api_response)):
            results = client.search("test", limit=5)

        assert len(results) == 2
        assert results[0]["name"] == "Test LoRA"

    def test_get_model(self):
        model_data = {
            "id": 12345,
            "name": "Flux LoRA",
            "modelVersions": [
                {
                    "id": 100,
                    "name": "v1.0",
                    "files": [
                        {"name": "model.safetensors", "sizeKB": 1024, "primary": True}
                    ],
                    "downloadUrl": "https://civitai.com/api/download/models/100",
                }
            ],
        }
        client = CivitaiClient()

        with patch("urllib.request.urlopen", return_value=_mock_response(model_data)):
            model = client.get_model(12345)

        assert model["name"] == "Flux LoRA"
        assert len(model["modelVersions"]) == 1

    def test_resolve_download(self):
        model_data = {
            "id": 12345,
            "name": "Flux LoRA",
            "modelVersions": [
                {
                    "id": 100,
                    "name": "v1.0",
                    "files": [
                        {"name": "model.safetensors", "sizeKB": 2048, "primary": True}
                    ],
                    "downloadUrl": "https://civitai.com/api/download/models/100",
                }
            ],
        }
        client = CivitaiClient()

        with patch("urllib.request.urlopen", return_value=_mock_response(model_data)):
            info = client.resolve_download(12345)

        assert isinstance(info, DownloadInfo)
        assert info.filename == "model.safetensors"
        assert info.size_bytes == 2048 * 1024
        assert "download" in info.url

    def test_resolve_download_specific_version(self):
        model_data = {
            "id": 12345,
            "name": "Multi Version",
            "modelVersions": [
                {
                    "id": 200,
                    "name": "v2.0",
                    "files": [{"name": "v2.safetensors", "sizeKB": 4096, "primary": True}],
                    "downloadUrl": "https://civitai.com/api/download/models/200",
                },
                {
                    "id": 100,
                    "name": "v1.0",
                    "files": [{"name": "v1.safetensors", "sizeKB": 2048, "primary": True}],
                    "downloadUrl": "https://civitai.com/api/download/models/100",
                },
            ],
        }
        client = CivitaiClient()

        with patch("urllib.request.urlopen", return_value=_mock_response(model_data)):
            info = client.resolve_download(12345, version_idx=1)

        assert info.filename == "v1.safetensors"

    def test_auth_header_sent(self):
        client = CivitaiClient(api_key="my-secret-key")

        with patch("urllib.request.urlopen", return_value=_mock_response({"items": []})) as mock_open:
            client.search("test")

        req = mock_open.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer my-secret-key"

    def test_api_key_from_env(self):
        with patch.dict("os.environ", {"CIVITAI_API_KEY": "env-key"}):
            client = CivitaiClient()
            assert client.api_key == "env-key"

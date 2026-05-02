"""Tests for CivitAI provider."""

from unittest.mock import patch, MagicMock

import orjson
import pytest
from typer.testing import CliRunner

from hfutils.cli import app
from hfutils.providers.civitai import CivitaiClient, parse_model_ref, DownloadInfo


def _mock_response(data: dict) -> MagicMock:
    """Create a mock urllib response returning JSON."""
    body = orjson.dumps(data)
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _ref(target: str):
    ref = parse_model_ref(target)
    assert ref is not None, f"expected parse to succeed for {target!r}"
    return ref


class TestParseModelRef:
    def test_numeric_id(self):
        ref = _ref("12345")
        assert ref.model_id == 12345
        assert ref.version_id is None
        assert ref.host == "civitai.com"

    def test_air_urn(self):
        ref = _ref("civitai:67890")
        assert ref.model_id == 67890
        assert ref.version_id is None

    def test_air_urn_with_version(self):
        ref = _ref("civitai:67890@1234")
        assert ref.model_id == 67890
        assert ref.version_id == 1234

    def test_air_urn_full_form_with_version(self):
        ref = _ref("urn:air:flux:lora:civitai:67890@1234")
        assert ref.model_id == 67890
        assert ref.version_id == 1234

    def test_civitai_url(self):
        ref = _ref("https://civitai.com/models/12345/some-model-name")
        assert ref.model_id == 12345
        assert ref.version_id is None
        assert ref.host == "civitai.com"

    def test_civitai_url_with_version(self):
        ref = _ref("https://civitai.com/models/12345?modelVersionId=99")
        assert ref.model_id == 12345
        assert ref.version_id == 99

    def test_civitai_url_with_slug_and_version(self):
        ref = _ref("https://civitai.com/models/12345/my-lora?modelVersionId=99")
        assert ref.model_id == 12345
        assert ref.version_id == 99

    def test_civitai_red_url(self):
        ref = _ref("https://civitai.red/models/12345/some-name")
        assert ref.model_id == 12345
        assert ref.host == "civitai.red"

    def test_civitai_red_url_with_version(self):
        ref = _ref("https://civitai.red/models/12345?modelVersionId=99")
        assert ref.model_id == 12345
        assert ref.version_id == 99
        assert ref.host == "civitai.red"

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
            "description": "A test model.",
            "modelVersions": [
                {
                    "id": 100,
                    "name": "v1.0",
                    "baseModel": "Flux.1 D",
                    "trainedWords": ["mychar", "1girl"],
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
        assert info.model_id == 12345
        assert info.version_id == 100
        assert info.base_model == "Flux.1 D"
        assert info.trained_words == ["mychar", "1girl"]
        assert info.description == "A test model."

    def test_resolve_download_missing_optional_fields(self):
        model_data = {
            "id": 12345,
            "name": "Bare LoRA",
            "modelVersions": [
                {
                    "id": 100,
                    "name": "v1.0",
                    "files": [{"name": "m.safetensors", "sizeKB": 1, "primary": True}],
                    "downloadUrl": "https://civitai.com/api/download/models/100",
                }
            ],
        }
        client = CivitaiClient()

        with patch("urllib.request.urlopen", return_value=_mock_response(model_data)):
            info = client.resolve_download(12345)

        assert info.trained_words == []
        assert info.base_model is None
        assert info.description is None

    def test_resolve_download_specific_version_by_id(self):
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
            info = client.resolve_download(12345, version_id=100)

        assert info.filename == "v1.safetensors"
        assert info.version_id == 100

    def test_resolve_download_unknown_version_id_raises(self):
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
            ],
        }
        client = CivitaiClient()

        with patch("urllib.request.urlopen", return_value=_mock_response(model_data)):
            with pytest.raises(ValueError, match="Version 999 not found"):
                client.resolve_download(12345, version_id=999)

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

    def test_default_host_is_civitai_com(self):
        client = CivitaiClient()
        assert client.host == "civitai.com"
        assert client.base_url == "https://civitai.com/api/v1"

    def test_custom_host_civitai_red(self):
        client = CivitaiClient(host="civitai.red")
        assert client.host == "civitai.red"
        assert client.base_url == "https://civitai.red/api/v1"

    def test_red_host_used_in_request(self):
        client = CivitaiClient(host="civitai.red")
        with patch("urllib.request.urlopen", return_value=_mock_response({"items": []})) as mock_open:
            client.search("test")
        req = mock_open.call_args[0][0]
        assert "civitai.red" in req.full_url


class TestDlSidecar:
    """`civitai dl` writes a <file>.civitai.json sidecar after downloading."""

    def _model_data(self) -> dict:
        return {
            "id": 12345,
            "name": "Flux LoRA",
            "description": "<p>Use mychar to trigger.</p>",
            "modelVersions": [
                {
                    "id": 100,
                    "name": "v1.0",
                    "baseModel": "Flux.1 D",
                    "trainedWords": ["mychar", "1girl"],
                    "files": [
                        {"name": "model.safetensors", "sizeKB": 1, "primary": True}
                    ],
                    "downloadUrl": "https://civitai.com/api/download/models/100",
                }
            ],
        }

    def test_writes_sidecar_with_metadata(self, tmp_path):
        runner = CliRunner()
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(self._model_data()),
        ), patch("hfutils.commands.civitai.download_file") as mock_dl:
            def _fake_dl(url, dest, total_size, headers):
                dest.write_bytes(b"x")
            mock_dl.side_effect = _fake_dl

            result = runner.invoke(
                app,
                ["civitai", "dl", "12345", "--output", str(tmp_path)],
                input="y\n",
            )

        assert result.exit_code == 0, result.output
        sidecar = tmp_path / "model.safetensors.civitai.json"
        assert sidecar.exists()

        data = orjson.loads(sidecar.read_bytes())
        assert data["model_id"] == 12345
        assert data["version_id"] == 100
        assert data["model_name"] == "Flux LoRA"
        assert data["version_name"] == "v1.0"
        assert data["base_model"] == "Flux.1 D"
        assert data["trained_words"] == ["mychar", "1girl"]
        assert "mychar" in data["description"]

    def _multi_version_data(self) -> dict:
        return {
            "id": 12345,
            "name": "Multi LoRA",
            "modelVersions": [
                {
                    "id": 200,
                    "name": "v2.0",
                    "files": [{"name": "v2.safetensors", "sizeKB": 1, "primary": True}],
                    "downloadUrl": "https://civitai.com/api/download/models/200",
                },
                {
                    "id": 100,
                    "name": "v1.0",
                    "files": [{"name": "v1.safetensors", "sizeKB": 1, "primary": True}],
                    "downloadUrl": "https://civitai.com/api/download/models/100",
                },
            ],
        }

    def test_dl_version_flag_picks_specific_version(self, tmp_path):
        runner = CliRunner()
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(self._multi_version_data()),
        ), patch("hfutils.commands.civitai.download_file") as mock_dl:
            mock_dl.side_effect = lambda url, dest, total_size, headers: dest.write_bytes(b"x")

            result = runner.invoke(
                app,
                ["civitai", "dl", "12345", "--version", "100", "--output", str(tmp_path)],
                input="y\n",
            )

        assert result.exit_code == 0, result.output
        assert (tmp_path / "v1.safetensors").exists()
        sidecar = tmp_path / "v1.safetensors.civitai.json"
        assert orjson.loads(sidecar.read_bytes())["version_id"] == 100

    def test_dl_uses_url_embedded_version(self, tmp_path):
        runner = CliRunner()
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(self._multi_version_data()),
        ), patch("hfutils.commands.civitai.download_file") as mock_dl:
            mock_dl.side_effect = lambda url, dest, total_size, headers: dest.write_bytes(b"x")

            result = runner.invoke(
                app,
                [
                    "civitai", "dl",
                    "https://civitai.com/models/12345?modelVersionId=100",
                    "--output", str(tmp_path),
                ],
                input="y\n",
            )

        assert result.exit_code == 0, result.output
        assert (tmp_path / "v1.safetensors").exists()

    def test_dl_red_url_uses_red_host(self, tmp_path):
        runner = CliRunner()
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(self._model_data()),
        ) as mock_open, patch("hfutils.commands.civitai.download_file") as mock_dl:
            mock_dl.side_effect = lambda url, dest, total_size, headers: dest.write_bytes(b"x")

            result = runner.invoke(
                app,
                [
                    "civitai", "dl",
                    "https://civitai.red/models/12345/x",
                    "--output", str(tmp_path),
                ],
                input="y\n",
            )

        assert result.exit_code == 0, result.output
        api_call = mock_open.call_args_list[0][0][0]
        assert "civitai.red" in api_call.full_url

    def test_no_sidecar_when_download_fails(self, tmp_path):
        runner = CliRunner()
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(self._model_data()),
        ), patch("hfutils.commands.civitai.download_file") as mock_dl:
            mock_dl.side_effect = RuntimeError("network exploded")

            result = runner.invoke(
                app,
                ["civitai", "dl", "12345", "--output", str(tmp_path)],
                input="y\n",
            )

        assert result.exit_code != 0
        assert not (tmp_path / "model.safetensors.civitai.json").exists()

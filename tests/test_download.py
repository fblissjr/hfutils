"""Tests for shared download module."""

import io
from unittest.mock import patch

from hfutils.providers.download import download_file, get_file_metadata


class _FakeResponse:
    """Minimal fake HTTP response for testing."""

    def __init__(self, data: bytes, headers: dict | None = None, url: str = ""):
        self._data = io.BytesIO(data)
        self.headers = headers or {}
        self.url = url
        self.status = 200

    def read(self, n=-1):
        return self._data.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class TestDownloadFile:
    def test_downloads_to_dest(self, tmp_path):
        dest = tmp_path / "model.safetensors"
        data = b"x" * 1000

        with patch("urllib.request.urlopen", return_value=_FakeResponse(data)):
            result = download_file("https://example.com/file", dest, total_size=1000)

        assert result is True
        assert dest.exists()
        assert dest.read_bytes() == data

    def test_skips_already_complete(self, tmp_path):
        dest = tmp_path / "model.safetensors"
        dest.write_bytes(b"x" * 500)

        with patch("urllib.request.urlopen") as mock_open:
            result = download_file("https://example.com/file", dest, total_size=500)

        assert result is True
        mock_open.assert_not_called()

    def test_resumes_partial(self, tmp_path):
        dest = tmp_path / "model.safetensors"
        dest.write_bytes(b"A" * 300)
        remaining = b"B" * 200

        with patch("urllib.request.urlopen", return_value=_FakeResponse(remaining)) as mock_open:
            result = download_file("https://example.com/file", dest, total_size=500)

        assert result is True
        assert len(dest.read_bytes()) == 500
        # Verify Range header was set
        req = mock_open.call_args[0][0]
        assert req.get_header("Range") == "bytes=300-"

    def test_returns_false_on_error(self, tmp_path):
        dest = tmp_path / "model.safetensors"

        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("fail")):
            result = download_file("https://example.com/file", dest, total_size=100)

        assert result is False


class TestGetFileMetadata:
    def test_extracts_filename_from_content_disposition(self):
        headers = {
            "Content-Length": "1024",
            "Content-Disposition": 'attachment; filename="model_v2.safetensors"',
        }
        resp = _FakeResponse(b"", headers=headers, url="https://example.com/dl")

        with patch("urllib.request.urlopen", return_value=resp):
            size, filename = get_file_metadata("https://example.com/dl")

        assert size == 1024
        assert filename == "model_v2.safetensors"

    def test_extracts_filename_from_url(self):
        headers = {"Content-Length": "2048"}
        resp = _FakeResponse(b"", headers=headers, url="https://example.com/files/model.gguf")

        with patch("urllib.request.urlopen", return_value=resp):
            size, filename = get_file_metadata("https://example.com/files/model.gguf")

        assert size == 2048
        assert filename == "model.gguf"

    def test_returns_zeros_on_error(self):
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("fail")):
            size, filename = get_file_metadata("https://example.com/dl")

        assert size == 0
        assert filename == ""

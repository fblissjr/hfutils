"""Tests for shared download module."""

import io
from unittest.mock import patch

from hfutils.providers.download import download_file, get_file_metadata


class _FakeResponse:
    """Minimal fake HTTP response for testing."""

    def __init__(self, data: bytes, headers: dict | None = None, url: str = "", status: int = 200):
        self._data = io.BytesIO(data)
        self.headers = headers or {}
        self.url = url
        self.status = status

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

        with patch("urllib.request.urlopen", return_value=_FakeResponse(remaining, status=206)) as mock_open:
            result = download_file("https://example.com/file", dest, total_size=500)

        assert result is True
        assert len(dest.read_bytes()) == 500
        # Verify Range header was set
        req = mock_open.call_args[0][0]
        assert req.get_header("Range") == "bytes=300-"

    def test_restarts_when_server_ignores_range(self, tmp_path):
        """A server/CDN that doesn't honor Range returns 200 with the full
        body instead of 206 with just the remainder -- appending that after
        the existing partial bytes would corrupt the file. Must restart
        from scratch instead."""
        dest = tmp_path / "model.safetensors"
        dest.write_bytes(b"A" * 300)
        full_body = b"C" * 500

        with patch("urllib.request.urlopen", return_value=_FakeResponse(full_body, status=200)):
            result = download_file("https://example.com/file", dest, total_size=500)

        assert result is True
        assert dest.read_bytes() == full_body

    def test_resume_message_uses_format_size(self, tmp_path, capsys):
        dest = tmp_path / "model.safetensors"
        dest.write_bytes(b"A" * 300)
        remaining = b"B" * 200

        with patch("urllib.request.urlopen", return_value=_FakeResponse(remaining, status=206)):
            download_file("https://example.com/file", dest, total_size=500, show_progress=False)

        out = capsys.readouterr().out
        assert "300 B" in out
        assert " MB" not in out

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

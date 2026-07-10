"""Tests for shared streaming primitives in io/progress.py."""

import pytest

from hfutils.io.progress import stream_region, unlink_on_error


class TestStreamRegion:
    def test_copies_bytes_verbatim(self, tmp_path):
        src = tmp_path / "src.bin"
        src.write_bytes(b"0123456789" * 10)
        dst = tmp_path / "dst.bin"

        with open(src, "rb") as fin, open(dst, "wb") as fout:
            fin.seek(10)
            stream_region(fin, fout, 50, src_desc=str(src))

        assert dst.read_bytes() == (b"0123456789" * 10)[10:60]

    def test_invokes_on_chunk_with_total_bytes(self, tmp_path):
        src = tmp_path / "src.bin"
        src.write_bytes(b"x" * 100)
        dst = tmp_path / "dst.bin"
        seen = []

        with open(src, "rb") as fin, open(dst, "wb") as fout:
            stream_region(fin, fout, 100, src_desc=str(src), on_chunk=seen.append)

        assert sum(seen) == 100

    def test_short_read_raises_configured_error_cls(self, tmp_path):
        src = tmp_path / "src.bin"
        src.write_bytes(b"x" * 10)
        dst = tmp_path / "dst.bin"

        class _CustomError(Exception):
            pass

        with open(src, "rb") as fin, open(dst, "wb") as fout:
            with pytest.raises(_CustomError):
                stream_region(fin, fout, 50, src_desc=str(src), error_cls=_CustomError)

    def test_default_error_cls_is_oserror(self, tmp_path):
        src = tmp_path / "src.bin"
        src.write_bytes(b"x" * 10)
        dst = tmp_path / "dst.bin"

        with open(src, "rb") as fin, open(dst, "wb") as fout:
            with pytest.raises(OSError):
                stream_region(fin, fout, 50, src_desc=str(src))


class TestUnlinkOnError:
    def test_noop_on_success(self, tmp_path):
        path = tmp_path / "out.bin"
        with unlink_on_error(path):
            path.write_bytes(b"data")

        assert path.read_bytes() == b"data"

    def test_deletes_file_on_exception(self, tmp_path):
        path = tmp_path / "out.bin"
        with pytest.raises(RuntimeError):
            with unlink_on_error(path):
                path.write_bytes(b"partial")
                raise RuntimeError("boom")

        assert not path.exists()

    def test_missing_file_on_exception_is_not_an_error(self, tmp_path):
        path = tmp_path / "never-created.bin"
        with pytest.raises(RuntimeError):
            with unlink_on_error(path):
                raise RuntimeError("boom")

        assert not path.exists()

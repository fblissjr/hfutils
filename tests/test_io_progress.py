"""Tests for shared streaming primitives in io/progress.py."""

import pytest

from hfutils.io.progress import copy_chunks, stream_region, unlink_on_error


class TestCopyChunks:
    def test_on_open_fires_once_before_any_chunk(self, tmp_path):
        src = tmp_path / "src.bin"
        src.write_bytes(b"x" * 100)
        dst = tmp_path / "dst.bin"
        calls = []

        copy_chunks(
            src, dst,
            on_chunk=lambda n: calls.append(("chunk", n)),
            on_open=lambda: calls.append(("open",)),
        )

        assert calls[0] == ("open",)
        assert all(c[0] == "chunk" for c in calls[1:])

    def test_on_open_is_optional(self, tmp_path):
        src = tmp_path / "src.bin"
        src.write_bytes(b"x" * 10)
        dst = tmp_path / "dst.bin"

        copy_chunks(src, dst)  # must not raise

        assert dst.read_bytes() == b"x" * 10


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
        with unlink_on_error(path) as mark_started:
            mark_started()
            path.write_bytes(b"data")

        assert path.read_bytes() == b"data"

    def test_deletes_file_on_exception_after_mark_started(self, tmp_path):
        path = tmp_path / "out.bin"
        with pytest.raises(RuntimeError):
            with unlink_on_error(path) as mark_started:
                path.write_bytes(b"partial")
                mark_started()
                raise RuntimeError("boom")

        assert not path.exists()

    def test_missing_file_on_exception_is_not_an_error(self, tmp_path):
        path = tmp_path / "never-created.bin"
        with pytest.raises(RuntimeError):
            with unlink_on_error(path) as mark_started:
                mark_started()
                raise RuntimeError("boom")

        assert not path.exists()

    def test_does_not_delete_pre_existing_file_when_failure_precedes_mark_started(self, tmp_path):
        """A failure that happens before this run ever started writing (e.g.
        an earlier context manager in the same `with` statement failing to
        enter) must not delete a pre-existing, unrelated file sitting at the
        destination path -- only a run that actually began writing should
        clean up after itself."""
        path = tmp_path / "out.bin"
        path.write_bytes(b"pre-existing valid content")

        with pytest.raises(RuntimeError):
            with unlink_on_error(path) as mark_started:
                raise RuntimeError("boom, before mark_started() was ever called")

        assert path.exists()
        assert path.read_bytes() == b"pre-existing valid content"

    def test_deletes_pre_existing_file_once_started_and_then_fails(self, tmp_path):
        """Once writing has actually begun (mark_started() called), a
        pre-existing file at the path has already been superseded -- a
        subsequent failure should still clean up the now-partial output."""
        path = tmp_path / "out.bin"
        path.write_bytes(b"stale content about to be overwritten")

        with pytest.raises(RuntimeError):
            with unlink_on_error(path) as mark_started:
                path.write_bytes(b"partial-new")
                mark_started()
                raise RuntimeError("boom")

        assert not path.exists()

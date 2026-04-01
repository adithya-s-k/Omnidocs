"""
tests/benchmark/test_resume_logic.py

Unit tests for resume & failure management in the benchmark runners.
No GPU, no Modal, no real models — everything is mocked.

Run with:
    cd Omnidocs
    pytest tests/benchmark/test_resume_logic.py -v
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from benchmarks.base import PageResult, PageSample
from benchmarks.multilingual.runner import (
    _load_state,
    _save_state,
    run_inference,
    run_inference_remote,
    write_md_files,
)
from benchmarks.omnidocbench.runner import (
    run_inference as omni_run_inference,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(name: str = "page_01") -> PageSample:
    """Tiny 1x1 white JPEG as a stand-in for a real page image."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (1, 1), color=(255, 255, 255)).save(buf, format="JPEG")
    return PageSample(image_bytes=buf.getvalue(), image_name=name)


def _make_extractor(markdown: str = "hello world", raise_on: str | None = None):
    """
    Returns a mock extractor whose .extract() returns a SimpleNamespace
    with a .content attribute.  If raise_on matches the image_name being
    processed the extractor raises RuntimeError instead.
    """

    class _FakeExtractor:
        def extract(self, image, output_format=None):
            return SimpleNamespace(content=markdown, plain_text=None, model_name="fake")

    class _RaisingExtractor:
        def __init__(self, bad_name):
            self._bad = bad_name

        def extract(self, image, output_format=None):
            # We can't easily check the image name here, so the caller
            # patches at the sample level — see test below.
            raise RuntimeError("simulated GPU OOM")

    if raise_on:
        return _RaisingExtractor(raise_on)
    return _FakeExtractor()


# ---------------------------------------------------------------------------
# _load_state / _save_state
# ---------------------------------------------------------------------------


class TestStateIO:
    def test_load_missing(self, tmp_path):
        """Loading state when no file exists returns empty dict."""
        state = _load_state(tmp_path)
        assert state == {}

    def test_roundtrip(self, tmp_path):
        data = {"completed_models": ["glmocr"], "inference": {"glmocr": {"en": {"written": 5}}}}
        _save_state(tmp_path, data)
        loaded = _load_state(tmp_path)
        assert loaded == data

    def test_corrupted_state_returns_empty(self, tmp_path):
        (tmp_path / "run_state.json").write_text("not json!!!", encoding="utf-8")
        assert _load_state(tmp_path) == {}


# ---------------------------------------------------------------------------
# write_md_files (multilingual)
# ---------------------------------------------------------------------------


class TestWriteMdFiles:
    def test_writes_success(self, tmp_path):
        results = [
            PageResult(image_name="p1", model="fake", markdown="text A", latency_s=0.1),
            PageResult(image_name="p2", model="fake", markdown="text B", latency_s=0.2),
        ]
        written, failed = write_md_files(results, tmp_path)
        assert written == 2
        assert failed == 0
        assert (tmp_path / "p1.md").read_text() == "text A"
        assert (tmp_path / "p2.md").read_text() == "text B"

    def test_writes_empty_for_failed(self, tmp_path):
        results = [
            PageResult(image_name="p1", model="fake", markdown="", latency_s=0.1, failed=True, error="boom"),
        ]
        written, failed = write_md_files(results, tmp_path)
        assert written == 0
        assert failed == 1
        assert (tmp_path / "p1.md").exists()
        assert (tmp_path / "p1.md").stat().st_size == 0

    def test_skips_already_existing(self, tmp_path):
        """Pages already written to disk are counted but not overwritten."""
        md = tmp_path / "p1.md"
        md.write_text("original content", encoding="utf-8")
        results = [
            PageResult(image_name="p1", model="fake", markdown="new content", latency_s=0.1),
        ]
        write_md_files(results, tmp_path)
        # File must NOT have been overwritten
        assert md.read_text() == "original content"


# ---------------------------------------------------------------------------
# run_inference (multilingual local runner) — resume behaviour
# ---------------------------------------------------------------------------


class TestRunInference:
    def test_writes_pages_immediately(self, tmp_path):
        """Each page is written to disk as soon as inference completes."""
        samples = [_make_sample("page_01"), _make_sample("page_02")]
        extractor = _make_extractor("predicted text")
        results = run_inference(extractor, samples, "fake", "en", tmp_path)

        assert len(results) == 2
        assert (tmp_path / "page_01.md").read_text() == "predicted text"
        assert (tmp_path / "page_02.md").read_text() == "predicted text"

    def test_skips_existing_non_empty_pages(self, tmp_path):
        """Pages whose .md already exists and is non-empty are skipped."""
        # Pre-write page_01 as if a previous run completed it
        (tmp_path / "page_01.md").write_text("prior result", encoding="utf-8")

        samples = [_make_sample("page_01"), _make_sample("page_02")]
        call_count = 0

        class _CountingExtractor:
            def extract(self, image, output_format=None):
                nonlocal call_count
                call_count += 1
                return SimpleNamespace(content="new result", plain_text=None, model_name="fake")

        run_inference(_CountingExtractor(), samples, "fake", "en", tmp_path)

        # Extractor should only have been called for page_02
        assert call_count == 1
        # page_01 must still contain the original text
        assert (tmp_path / "page_01.md").read_text() == "prior result"

    def test_failed_page_writes_empty_file(self, tmp_path):
        """A page that raises during inference gets an empty .md written."""
        samples = [_make_sample("bad_page")]

        class _BoomExtractor:
            def extract(self, image, output_format=None):
                raise RuntimeError("GPU exploded")

        results = run_inference(_BoomExtractor(), samples, "fake", "en", tmp_path)

        assert results[0].failed is True
        assert (tmp_path / "bad_page.md").exists()
        assert (tmp_path / "bad_page.md").stat().st_size == 0

    def test_empty_file_not_treated_as_done(self, tmp_path):
        """An empty .md (from a prior crash) is NOT skipped — page is retried."""
        (tmp_path / "page_01.md").write_text("", encoding="utf-8")

        samples = [_make_sample("page_01")]
        extractor = _make_extractor("retry result")
        results = run_inference(extractor, samples, "fake", "en", tmp_path)

        assert results[0].markdown == "retry result"
        assert (tmp_path / "page_01.md").read_text() == "retry result"


# ---------------------------------------------------------------------------
# run_inference_remote (no disk I/O)
# ---------------------------------------------------------------------------


class TestRunInferenceRemote:
    def test_returns_results_without_writing(self, tmp_path):
        samples = [_make_sample("page_01")]
        extractor = _make_extractor("remote output")
        results = run_inference_remote(extractor, samples, "fake", "en")

        assert len(results) == 1
        assert results[0].markdown == "remote output"
        # Nothing should have been written to disk
        assert not list(tmp_path.iterdir())

    def test_failed_page_recorded_not_raised(self, tmp_path):
        samples = [_make_sample("page_01")]

        class _BoomExtractor:
            def extract(self, image, output_format=None):
                raise RuntimeError("remote boom")

        results = run_inference_remote(_BoomExtractor(), samples, "fake", "en")
        assert results[0].failed is True
        assert "remote boom" in results[0].error


# ---------------------------------------------------------------------------
# run_multilingual  — model-level resume via run_state.json
# ---------------------------------------------------------------------------


class TestRunMultilingualResume:
    """
    Tests the full run_multilingual() function with all heavy deps mocked out.
    Validates that completed_models in run_state.json prevents re-running a model.
    """

    def _make_fake_sample(self):
        return _make_sample("img_01")

    @pytest.fixture()
    def patched_multilingual(self, tmp_path):
        fake_sample = self._make_fake_sample()
        samples_by_lang = {"en": [fake_sample]}
        gt_by_lang = {"en": [{"image_name": "img_01", "gt_markdown": "truth"}]}

        with (
            # patch at the SOURCE module, not the runner, because they're imported locally inside the function
            patch("benchmarks.multilingual.dataset.load_multilingual", return_value=(samples_by_lang, gt_by_lang)),
            patch("benchmarks.multilingual.runner.get_extractor", return_value=_make_extractor("predicted")),
            patch(
                "benchmarks.multilingual.evaluator.run_evaluation", return_value={"fake_model": {"en": {"cer": 0.0}}}
            ),
            patch("benchmarks.multilingual.runner.MODEL_REGISTRY", {"fake_model": lambda: None}),
        ):
            yield tmp_path

    def test_first_run_creates_state(self, patched_multilingual):
        from benchmarks.multilingual.runner import run_multilingual

        out = patched_multilingual
        run_multilingual(
            model_keys=["fake_model"],
            languages=["en"],
            max_per_language=1,
            output_dir=out,
            run_eval=False,
        )

        state = _load_state(out)
        assert "fake_model" in state["completed_models"]
        assert "fake_model" in state["inference"]

    def test_second_run_skips_completed_model(self, patched_multilingual):
        from benchmarks.multilingual.runner import run_multilingual

        out = patched_multilingual

        # Simulate a prior completed run by pre-writing the state file
        _save_state(
            out,
            {
                "completed_models": ["fake_model"],
                "inference": {"fake_model": {"en": {"written": 1, "failed": 0}}},
            },
        )
        # Also pre-write the .md so write_md_files doesn't fail
        md_dir = out / "fake_model" / "en"
        md_dir.mkdir(parents=True)
        (md_dir / "img_01.md").write_text("prior result", encoding="utf-8")

        with patch("benchmarks.multilingual.runner.get_extractor") as mock_get:
            run_multilingual(
                model_keys=["fake_model"],
                languages=["en"],
                max_per_language=1,
                output_dir=out,
                run_eval=False,
            )
            # Extractor must NOT have been loaded again
            mock_get.assert_not_called()

    def test_failed_model_not_in_completed(self, patched_multilingual):
        """A model whose extractor raises is NOT added to completed_models."""
        from benchmarks.multilingual.runner import run_multilingual

        out = patched_multilingual

        with patch("benchmarks.multilingual.runner.get_extractor", side_effect=RuntimeError("load failed")):
            run_multilingual(
                model_keys=["fake_model"],
                languages=["en"],
                max_per_language=1,
                output_dir=out,
                run_eval=False,
            )

        state = _load_state(out)
        assert "fake_model" not in state.get("completed_models", [])


# ---------------------------------------------------------------------------
# omnidocbench runner — same resume logic, different module
# ---------------------------------------------------------------------------


class TestOmniDocBenchRunInference:
    def test_skips_existing_pages(self, tmp_path):
        (tmp_path / "page_01.md").write_text("cached", encoding="utf-8")

        samples = [_make_sample("page_01.jpg")]
        call_count = 0

        class _CountingExtractor:
            def extract(self, image, output_format=None):
                nonlocal call_count
                call_count += 1
                return SimpleNamespace(content="new", plain_text=None, model_name="fake")

        omni_run_inference(_CountingExtractor(), samples, "fake", tmp_path)
        assert call_count == 0

    def test_failed_page_writes_empty_file(self, tmp_path):
        samples = [_make_sample("page_01.jpg")]

        class _BoomExtractor:
            def extract(self, image, output_format=None):
                raise RuntimeError("boom")

        results = omni_run_inference(_BoomExtractor(), samples, "fake", tmp_path)
        assert results[0].failed is True
        assert (tmp_path / "page_01.md").stat().st_size == 0

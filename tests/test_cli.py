"""Tests for CLI argument parsing and main flow."""
from __future__ import annotations

import sys
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from pathlib import Path
import argparse

from srt_translator.cli import parse_arguments, main_async


class TestParseArguments:
    """Test CLI argument parsing."""

    def test_minimal_args(self):
        """Test parsing with only input path."""
        test_args = ["srt-translator", "input.srt"]
        with patch.object(sys, "argv", test_args):
            args = parse_arguments()
        assert args.input_path == "input.srt"
        assert args.output_path is None
        assert args.glossary_path is None
        assert args.max_chars_per_entry == 300
        assert args.no_merge is False
        assert args.source_language == "en"
        assert args.api_key is None
        assert args.base_url == "https://api.deepseek.com"
        assert args.model_name is None
        assert args.concurrency == 8
        assert args.verbose is False

    def test_all_args(self):
        """Test parsing with all arguments."""
        test_args = [
            "srt-translator", "input.srt", "output.srt",
            "-g", "glossary.txt",
            "--no-merge",
            "--source-language", "ja",
            "--max-chars", "200",
            "--merge-gap", "2.0",
            "--api-key", "test-key",
            "--base-url", "https://custom.api.com",
            "--model", "custom-model",
            "--summary-model", "custom-summary-model",
            "--concurrency", "16",
            "--chunk-size", "5",
            "--resume",
            "--no-progress",
            "-v",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_arguments()
        assert args.input_path == "input.srt"
        assert args.output_path == "output.srt"
        assert args.glossary_path == "glossary.txt"
        assert args.max_chars_per_entry == 200
        assert args.no_merge is True
        assert args.source_language == "ja"
        assert args.merge_time_gap == 2.0
        assert args.api_key == "test-key"
        assert args.base_url == "https://custom.api.com"
        assert args.model_name == "custom-model"
        assert args.summary_model_name == "custom-summary-model"
        assert args.concurrency == 16
        assert args.chunk_size_for_translation == 5
        assert args.resume is True
        assert args.no_progress is True
        assert args.verbose is True

    def test_default_output_path(self):
        """Test that output path defaults to None when not specified."""
        test_args = ["srt-translator", "input.srt"]
        with patch.object(sys, "argv", test_args):
            args = parse_arguments()
        assert args.output_path is None

    def test_output_option(self):
        """Test specifying output with -o/--output."""
        test_args = ["srt-translator", "input.srt", "-o", "output.srt"]
        with patch.object(sys, "argv", test_args):
            args = parse_arguments()
        assert args.output_path == "output.srt"

    def test_rejects_duplicate_output_paths(self):
        """Test that positional output and -o cannot both be used."""
        test_args = ["srt-translator", "input.srt", "positional.srt", "-o", "option.srt"]
        with patch.object(sys, "argv", test_args), pytest.raises(SystemExit):
            parse_arguments()


class TestMainAsync:
    """Test the async main workflow."""

    @pytest.fixture
    def mock_args(self):
        """Create mock args for testing."""
        args = MagicMock(spec=argparse.Namespace)
        args.input_path = str(Path("tests/test_input.srt"))
        args.output_path = None
        args.glossary_path = None
        args.api_key = "test-key"
        args.max_chars_per_entry = 300
        args.merge_time_gap = 1.5
        args.api_key = "test-key"
        args.base_url = "https://test.api.com"
        args.model_name = "test-model"
        args.summary_model_name = "test-model"
        args.concurrency = 2
        args.chunk_size_for_translation = 10
        args.resume = False
        args.no_progress = True
        args.no_merge = False
        args.verbose = False
        args.summary_prompt = None
        args.translation_prompt = None
        return args

    @pytest.mark.asyncio
    async def test_main_async_no_input_file(self, mock_args, tmp_path):
        """Test main_async when input file doesn't exist."""
        mock_args.input_path = str(tmp_path / "nonexistent.srt")

        result = await main_async(mock_args)
        assert result == 1  # Should return error code

    @pytest.mark.asyncio
    async def test_main_async_invalid_srt(self, mock_args, tmp_path):
        """Test main_async with invalid SRT content."""
        srt_path = tmp_path / "test.srt"
        srt_path.write_text("Not a valid SRT file", encoding="utf-8")

        mock_args.input_path = str(srt_path)
        result = await main_async(mock_args)
        assert result == 1

    @pytest.mark.asyncio
    async def test_main_async_valid_srt(self, mock_args, tmp_path):
        """Test main_async with valid SRT content and mocked LLM."""
        srt_path = tmp_path / "test.srt"
        srt_path.write_text(
            "1\n00:00:01,000 --> 00:00:03,500\nHello world\n\n"
            "2\n00:00:04,000 --> 00:00:06,500\nHow are you?\n",
            encoding="utf-8-sig"
        )
        mock_args.input_path = str(srt_path)

        with patch("srt_translator.cli.create_client") as mock_create_client, \
             patch("srt_translator.cli.init_spacy_model") as mock_init_spacy, \
             patch("srt_translator.cli.merge_entries_batch") as mock_merge_entries, \
             patch("srt_translator.cli.TranslationPipeline.run", new_callable=AsyncMock) as mock_pipeline_run, \
             patch("srt_translator.cli.parse_srt") as mock_parse_srt:

            from srt_translator.models import SrtEntry
            mock_entries = [
                SrtEntry(1, "00:00:01,000", "00:00:03,500", "Hello world"),
                SrtEntry(2, "00:00:04,000", "00:00:06,500", "How are you?"),
            ]
            mock_parse_srt.return_value = mock_entries
            mock_merge_entries.return_value = mock_entries

            mock_pipeline_run.return_value = {0: "你好世界", 1: "你好吗？"}

            mock_client = MagicMock()
            mock_create_client.return_value = mock_client

            result = await main_async(mock_args)
            assert result == 0  # Success
            mock_init_spacy.assert_called_once()
            mock_merge_entries.assert_called_once()

            # Verify output file was created
            output_path = srt_path.with_name(f"translated_{srt_path.name}")
            assert output_path.exists()

    @pytest.mark.asyncio
    async def test_main_async_no_merge_skips_spacy(self, mock_args, tmp_path):
        """Test --no-merge bypasses spaCy initialization and merging."""
        srt_path = tmp_path / "test.srt"
        srt_path.write_text(
            "1\n00:00:01,000 --> 00:00:03,500\nHello world\n",
            encoding="utf-8-sig"
        )
        mock_args.input_path = str(srt_path)
        mock_args.no_merge = True

        with patch("srt_translator.cli.create_client") as mock_create_client, \
             patch("srt_translator.cli.init_spacy_model") as mock_init_spacy, \
             patch("srt_translator.cli.merge_entries_batch") as mock_merge_entries, \
             patch("srt_translator.cli.TranslationPipeline.run", new_callable=AsyncMock) as mock_pipeline_run, \
             patch("srt_translator.cli.parse_srt") as mock_parse_srt:

            from srt_translator.models import SrtEntry
            mock_entries = [
                SrtEntry(1, "00:00:01,000", "00:00:03,500", "Hello world"),
            ]
            mock_parse_srt.return_value = mock_entries
            mock_pipeline_run.return_value = {0: "你好世界"}
            mock_create_client.return_value = MagicMock()

            result = await main_async(mock_args)

            assert result == 0
            mock_init_spacy.assert_not_called()
            mock_merge_entries.assert_not_called()
            assert mock_pipeline_run.await_args.kwargs["entries"][0].text == "Hello world"

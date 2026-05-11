"""Integration tests for the API server endpoints."""
from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from pathlib import Path
from fastapi.testclient import TestClient

from api_server import app, JOB_STATE, JOB_TASKS


@pytest.fixture
def client():
    """Create a TestClient for the FastAPI app."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test the /api/health endpoint."""

    def test_health_check(self, client):
        """Test health check returns ok."""
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestTranslateEndpoint:
    """Test the /api/translate endpoint."""

    def test_translate_no_file(self, client):
        """Test translate without file returns 422."""
        response = client.post("/api/translate")
        assert response.status_code == 422  # Validation error

    def test_translate_with_mock_file(self, client, tmp_path):
        """Test translate with a mock SRT file."""
        # Create a test SRT file
        srt_content = (
            "1\n00:00:01,000 --> 00:00:03,500\nHello world\n\n"
            "2\n00:00:04,000 --> 00:00:06,500\nHow are you?\n"
        )
        test_file = tmp_path / "test.srt"
        test_file.write_text(srt_content, encoding="utf-8")

        with patch("api_server.process_translation_job", new_callable=AsyncMock) as mock_process:
            with open(test_file, "rb") as f:
                response = client.post(
                    "/api/translate",
                    files={"file": ("test.srt", f, "application/octet-stream")},
                    data={
                        "api_key": "test-key",
                        "model_name": "test-model",
                        "concurrency": "2",
                        "save_merged_subtitles": "true",
                    },
                )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "job_id" in data
            assert "expected_output" in data
            assert "expected_merged_output" in data
            assert Path(data["expected_merged_output"]).name == "merged_test.srt"
            assert mock_process.call_args.args[9] is True

            # Verify JOB_STATE was created
            job_id = data["job_id"]
            assert job_id in JOB_STATE
            assert JOB_STATE[job_id]["status"] == "pending"

            # Cleanup
            del JOB_STATE[job_id]
            JOB_TASKS.pop(job_id, None)

    def test_translate_concurrency_limit(self, client, tmp_path):
        """Test concurrency limit is enforced."""
        # Fill JOB_STATE with running jobs to hit the limit
        for i in range(6):  # MAX_CONCURRENT_JOBS is 5
            JOB_STATE[f"existing-{i}"] = {"status": "running"}

        srt_content = "1\n00:00:01,000 --> 00:00:03,500\nHello\n"
        test_file = tmp_path / "test.srt"
        test_file.write_text(srt_content, encoding="utf-8")

        with open(test_file, "rb") as f:
            response = client.post(
                "/api/translate",
                files={"file": ("test.srt", f, "application/octet-stream")},
                data={"api_key": "test-key"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "服务器繁忙" in data["error"]

        # Cleanup
        for i in range(6):
            JOB_STATE.pop(f"existing-{i}", None)


class TestQualityCheckEndpoint:
    """Test the /api/quality-check endpoint."""

    def test_quality_check_with_mock_files(self, client, tmp_path):
        original = tmp_path / "original.srt"
        translated = tmp_path / "translated.srt"
        original.write_text("1\n00:00:01,000 --> 00:00:03,000\nHello world\n", encoding="utf-8")
        translated.write_text("1\n00:00:01,000 --> 00:00:03,000\n你好\n", encoding="utf-8")

        with patch("api_server.process_quality_check_job", new_callable=AsyncMock):
            with open(original, "rb") as original_f, open(translated, "rb") as translated_f:
                response = client.post(
                    "/api/quality-check",
                    files={
                        "original_file": ("original.srt", original_f, "application/octet-stream"),
                        "translated_file": ("translated.srt", translated_f, "application/octet-stream"),
                    },
                    data={
                        "api_key": "test-key",
                        "model_name": "test-model",
                    },
                )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "job_id" in data
        assert "expected_output" in data

        job_id = data["job_id"]
        assert job_id in JOB_STATE

        del JOB_STATE[job_id]
        JOB_TASKS.pop(job_id, None)


class TestStatusEndpoint:
    """Test the /api/status endpoint."""

    def test_status_nonexistent_job(self, client):
        """Test status for nonexistent job."""
        response = client.get("/api/status/nonexistent-id")
        assert response.status_code == 200
        assert response.json()["status"] == "error"

    def test_status_existing_job(self, client):
        """Test status for existing job."""
        job_id = "test-status-job"
        JOB_STATE[job_id] = {
            "status": "running",
            "progress_pct": 50,
            "logs": [{"text": "- 测试日志", "isError": False}],
            "error": None,
            "created_at": 1000.0,
            "completed_at": None,
        }

        response = client.get(f"/api/status/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["progress"] == 50
        assert len(data["logs"]) == 1

        # Cleanup
        del JOB_STATE[job_id]


class TestCancelEndpoint:
    """Test the /api/cancel endpoint."""

    def test_cancel_nonexistent_job(self, client):
        """Test cancel for nonexistent job."""
        response = client.post("/api/cancel/nonexistent-id")
        assert response.status_code == 200
        assert response.json()["status"] == "error"

    def test_cancel_running_job(self, client):
        """Test cancel for a running job."""
        job_id = "test-cancel-job"
        JOB_STATE[job_id] = {
            "status": "running",
            "progress_pct": 30,
            "logs": [],
            "error": None,
            "created_at": 1000.0,
            "completed_at": None,
        }

        # Create a mock task that is done (simulating completed cancellation)
        mock_task = MagicMock()
        mock_task.done.return_value = True  # Already done, so cancel() won't be called
        JOB_TASKS[job_id] = mock_task

        response = client.post(f"/api/cancel/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["message"] == "任务已取消。"
        assert JOB_STATE[job_id]["status"] == "cancelled"
        assert JOB_STATE[job_id]["completed_at"] is not None

        # Cleanup
        del JOB_STATE[job_id]
        del JOB_TASKS[job_id]

    def test_cancel_completed_job(self, client):
        """Test cancel for already completed job."""
        job_id = "test-completed-job"
        JOB_STATE[job_id] = {
            "status": "completed",
            "progress_pct": 100,
            "logs": [],
            "error": None,
            "created_at": 1000.0,
            "completed_at": 2000.0,
        }

        response = client.post(f"/api/cancel/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "无法取消" in data["error"]

        # Cleanup
        del JOB_STATE[job_id]

# tests/custom_tests/ragaai_catalyst/tracers/agentic_tracing/utils/test_trace_uploader.py
import pytest
import os
import tempfile
import json
from dotenv import load_dotenv
from ragaai_catalyst import RagaAICatalyst
from ragaai_catalyst.tracers.agentic_tracing.upload.trace_uploader import (
    submit_upload_task, get_task_status, get_executor, ensure_uploader_running, save_task_status
)

# Test responses dictionary
TEST_RESPONSES = {
  "get_executor": {
    "success": True,
    "is_executor": "<class 'concurrent.futures.thread.ThreadPoolExecutor'>"
  },
  "ensure_uploader_running": {
    "success": True,
    "uploader_running": True
  },
  "save_task_status": {
    "success": True,
    "task_id": "test_task_123"
  },
  "get_task_status": {
    "success": True,
    "task_status": {
      "task_id": "test_task_123",
      "status": "pending",
      "error": None,
      "start_time": "2023-01-01T00:00:00.000000"
    }
  },
  "submit_upload_task": {
    "success": True,
    "task_id": "task_1747387164_20308_-3065570686764785958",
    "task_status": {
      "status": "processing",
      "error": None
    }
  }
}

@pytest.fixture(scope="module")
def setup_environment():
    load_dotenv()
    catalyst = RagaAICatalyst(
        access_key=os.getenv('RAGAAI_CATALYST_ACCESS_KEY'),
        secret_key=os.getenv('RAGAAI_CATALYST_SECRET_KEY'),
        base_url=os.getenv('RAGAAI_CATALYST_BASE_URL')
    )
    return catalyst

@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(b'{"test": "data"}')
    temp_file.close()
    
    yield temp_file.name
    
    # Cleanup
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)

@pytest.fixture
def test_task_status():
    """Create a test task status for testing."""
    return {
        "task_id": "test_task_123",
        "status": "pending",
        "error": None,
        "start_time": "2023-01-01T00:00:00.000000"
    }

def test_get_executor():
    """Test that get_executor returns a ThreadPoolExecutor."""
    executor = get_executor()
    assert str(type(executor)) == TEST_RESPONSES["get_executor"]["is_executor"]

def test_ensure_uploader_running():
    """Test that ensure_uploader_running returns True."""
    uploader_running = ensure_uploader_running()
    assert uploader_running == TEST_RESPONSES["ensure_uploader_running"]["uploader_running"]

def test_save_and_get_task_status(test_task_status):
    """Test saving and getting task status."""
    # Save the task status
    save_task_status(test_task_status)
    
    # Get the task status
    retrieved_status = get_task_status(test_task_status["task_id"])
    
    # Check that the task_id matches
    assert retrieved_status["task_id"] == TEST_RESPONSES["get_task_status"]["task_status"]["task_id"]
    assert retrieved_status["status"] == TEST_RESPONSES["get_task_status"]["task_status"]["status"]
    assert retrieved_status["error"] == TEST_RESPONSES["get_task_status"]["task_status"]["error"]

def test_submit_upload_task(setup_environment, temp_file):
    """Test submitting an upload task."""
    # Submit an upload task
    task_id = submit_upload_task(
        filepath=temp_file,
        hash_id="test_hash_123",
        zip_path=None,
        project_name="agentic_tracer_sk_v3",
        project_id=767,
        dataset_name="pytest_dataset",
        user_details={"test": "details"},
        base_url=setup_environment.BASE_URL
    )
    
    # Check that a task_id was returned
    assert task_id is not None
    
    # Check the task status
    task_status = get_task_status(task_id)
    assert "status" in task_status
    assert task_status["error"] is None
    
    # Note: We don't assert the exact task_id as it changes on every call
    # We only check that a task_id was returned and it has a valid status
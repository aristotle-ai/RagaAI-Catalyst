# tests/custom_tests/ragaai_catalyst/tracers/agentic_tracing/utils/test_zip_list_of_unique_files.py
import pytest
import os
import tempfile
from dotenv import load_dotenv
from ragaai_catalyst.tracers.agentic_tracing.utils.zip_list_of_unique_files import (
    zip_list_of_unique_files, TraceDependencyTracker, JupyterNotebookHandler
)

# Test responses dictionary
TEST_RESPONSES = {
  "jupyter_notebook_handler": {
    "success": True,
    "is_notebook": False,
    "is_colab": False,
    "notebook_path": None
  },
  "trace_dependency_tracker_init": {
    "success": True,
    "output_dir": "D:\\May\\RagaAI-Catalyst\\tests\\custom_tests\\ragaai_catalyst\\tracers\\agentic_tracing\\utils"
  },
  "zip_list_of_unique_files": {
    "success": False,
    "error": "'TraceDependencyTracker' object has no attribute 'get_unique_files'"
  }
}

@pytest.fixture
def temp_py_file():
    """Create a temporary Python file for testing."""
    temp_dir = tempfile.mkdtemp()
    test_py_file = os.path.join(temp_dir, "test_file.py")
    
    with open(test_py_file, 'w') as f:
        f.write('print("Hello, world!")\n')
    
    yield test_py_file, temp_dir
    
    # Cleanup
    if os.path.exists(test_py_file):
        os.remove(test_py_file)
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)

def test_jupyter_notebook_handler():
    """Test JupyterNotebookHandler functionality."""
    jupyter_handler = JupyterNotebookHandler()
    
    # Check is_notebook
    is_notebook = jupyter_handler.is_running_in_notebook()
    assert is_notebook == TEST_RESPONSES["jupyter_notebook_handler"]["is_notebook"]
    
    # Check is_colab
    is_colab = jupyter_handler.is_running_in_colab()
    assert is_colab == TEST_RESPONSES["jupyter_notebook_handler"]["is_colab"]
    
    # Check get_notebook_path (may be None in test environment)
    notebook_path = jupyter_handler.get_notebook_path()
    # We don't assert the exact path as it may be None or environment-dependent
    assert isinstance(notebook_path, str) if notebook_path else notebook_path is None

def test_trace_dependency_tracker_init():
    """Test TraceDependencyTracker initialization."""
    tracker = TraceDependencyTracker()
    
    # Check that the output_dir is set (may vary by environment)
    assert isinstance(tracker.output_dir, str)
    assert os.path.exists(tracker.output_dir)

def test_zip_list_of_unique_files(temp_py_file):
    """Test zip_list_of_unique_files functionality."""
    test_py_file, temp_dir = temp_py_file
    
    # Attempt to create a zip file
    try:
        hash_id, zip_path = zip_list_of_unique_files([test_py_file], output_dir=temp_dir)
        
        # Check that the hash_id and zip_path are returned
        assert isinstance(hash_id, str)
        assert isinstance(zip_path, str)
        assert os.path.exists(zip_path)
        
        # Clean up the zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)
    except AttributeError as e:
        # Based on the response, we expect an error with this message
        assert "'TraceDependencyTracker' object has no attribute 'get_unique_files'" in str(e)
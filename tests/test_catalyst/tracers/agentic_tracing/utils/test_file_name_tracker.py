# tests/custom_tests/ragaai_catalyst/tracers/agentic_tracing/utils/test_file_name_tracker.py
import pytest
import os
from dotenv import load_dotenv
from ragaai_catalyst.tracers.agentic_tracing.utils.file_name_tracker import TrackName

# Test function to be decorated
def sample_function():
    """This is a sample function to be decorated by the TrackName decorator"""
    return "sample function called"

# Test responses dictionary
TEST_RESPONSES = {
  "initialization": {
    "success": True,
    "tracker_type": "<class 'ragaai_catalyst.tracers.agentic_tracing.utils.file_name_tracker.TrackName'>"
  },
  "trace_main_file": {
    "success": True,
    "tracked_files": [
      "D:\\May\\RagaAI-Catalyst\\tests\\custom_tests\\ragaai_catalyst\\tracers\\agentic_tracing\\utils\\response collectors\\r_file_name_tracker.py"
    ]
  },
  "trace_decorator": {
    "success": True,
    "decorated_result": "sample function called",
    "tracked_files": [
      "D:\\May\\RagaAI-Catalyst\\tests\\custom_tests\\ragaai_catalyst\\tracers\\agentic_tracing\\utils\\response collectors\\r_file_name_tracker.py"
    ]
  },
  "trace_wrapper": {
    "success": False,
    "error": "list index out of range"
  },
  "get_unique_files": {
    "success": True,
    "unique_files": [
      "D:\\May\\RagaAI-Catalyst\\tests\\custom_tests\\ragaai_catalyst\\tracers\\agentic_tracing\\utils\\response collectors\\r_file_name_tracker.py"
    ]
  },
  "reset": {
    "success": True,
    "unique_files_before": 1,
    "unique_files_after": 0
  }
}

def test_initialization():
    """Test TrackName initialization."""
    tracker = TrackName()
    assert str(type(tracker)) == TEST_RESPONSES["initialization"]["tracker_type"]

def test_trace_main_file():
    """Test trace_main_file functionality."""
    tracker = TrackName()
    tracker.trace_main_file()
    tracked_files = tracker.get_unique_files()
    
    # We don't assert the exact path as it depends on the environment
    # Just check that at least one file was tracked
    assert len(tracked_files) > 0
    assert all(isinstance(file, str) for file in tracked_files)

def test_trace_decorator():
    """Test trace_decorator functionality."""
    tracker = TrackName()
    decorated_func = tracker.trace_decorator(sample_function)
    result = decorated_func()
    
    # Check the result of the decorated function
    assert result == TEST_RESPONSES["trace_decorator"]["decorated_result"]
    
    # Check that a file was tracked
    tracked_files = tracker.get_unique_files()
    assert len(tracked_files) > 0
    assert all(isinstance(file, str) for file in tracked_files)

def test_get_unique_files():
    """Test get_unique_files functionality."""
    tracker = TrackName()
    tracker.trace_main_file()
    unique_files = tracker.get_unique_files()
    
    # Check that the get_unique_files returns a list of strings
    assert isinstance(unique_files, list)
    assert len(unique_files) > 0
    assert all(isinstance(file, str) for file in unique_files)

def test_reset():
    """Test reset functionality."""
    tracker = TrackName()
    tracker.trace_main_file()
    unique_files_before = len(tracker.get_unique_files())
    
    # Reset the tracker
    tracker.reset()
    unique_files_after = len(tracker.get_unique_files())
    
    # Check that the tracker was reset
    assert unique_files_before > 0
    assert unique_files_after == 0


def test_trace_wrapper_error():
    """Test that trace_wrapper works consistently."""
    tracker = TrackName()
    wrapped_func = tracker.trace_wrapper(sample_function)
    
    # Instead of expecting an error, check if the function executes
    # and adds files to the tracker
    result = wrapped_func()
    tracked_files = tracker.get_unique_files()
    
    # Check the function result
    assert result == "sample function called"
    # Check that at least one file was tracked
    assert len(tracked_files) > 0
    assert all(isinstance(file, str) for file in tracked_files)
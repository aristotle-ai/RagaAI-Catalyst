import pytest
import logging
import os
from unittest.mock import Mock
from ragaai_catalyst.tracers.distributed import (
    init_tracing, 
    get_current_tracer, 
    get_current_catalyst,
    current_span
)
from ragaai_catalyst import RagaAICatalyst
from ragaai_catalyst.tracers.tracer import Tracer

@pytest.fixture
def mock_tracer():
    tracer = Mock(spec=Tracer)
    tracer.current_agent_name = Mock()
    tracer.current_agent_name.get.return_value = None
    tracer.current_llm_call_name = Mock()
    tracer.current_llm_call_name.get.return_value = None
    tracer.current_tool_name = Mock()
    tracer.current_tool_name.get.return_value = None
    tracer.span = Mock()
    return tracer

@pytest.fixture
def mock_catalyst():
    return Mock(spec=RagaAICatalyst)

def test_init_tracing_invalid_object_types(caplog):
    """Test init_tracing with invalid object types logs error"""
    caplog.set_level(logging.ERROR)
    
    # Use non-Tracer and non-RagaAICatalyst objects
    init_tracing(tracer="not a tracer", catalyst="not a catalyst")
    
    # Check that error was logged
    assert "Both Tracer and Catalyst objects must be instances of Tracer and RagaAICatalyst" in caplog.text
    
    # Verify global vars were not set
    assert get_current_tracer() is None
    assert get_current_catalyst() is None

def test_init_tracing_missing_objects(caplog):
    """Test init_tracing with missing objects logs error"""
    caplog.set_level(logging.ERROR)
    
    # Call with no tracer and catalyst
    init_tracing()
    
    # Check that error was logged
    assert "Both Tracer and Catalyst objects must be provided" in caplog.text
    
    # Verify global vars were not set
    assert get_current_tracer() is None
    assert get_current_catalyst() is None

def test_current_span_no_active_span(caplog, mock_tracer):
    """Test current_span with no active span logs error"""
    caplog.set_level(logging.ERROR)
    
    # Set up module globals
    from ragaai_catalyst.tracers import distributed
    distributed._global_tracer = mock_tracer
    
    # Call current_span with no active agent name
    result = current_span()
    
    # Check that error was logged
    assert "No active span found" in caplog.text
    
    # Verify span was still returned
    assert result is not None
    assert result == mock_tracer.span.return_value
    mock_tracer.span.assert_called_once_with(None)
    
    # Reset module globals
    distributed._global_tracer = None
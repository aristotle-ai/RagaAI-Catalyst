import pytest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer
from ragaai_catalyst.tracers.exporters.ragaai_trace_exporter import RAGATraceExporter

# Test responses dictionary instead of loading from JSON file
TEST_RESPONSES = {
    "exporter_initialization": {
        "success": True,
        "exporter_attrs": {
            "tracer_type": "test_type",
            "project_name": "response_collection",
            "dataset_name": "test_dataset",
            "timeout": 120
        }
    },
    "export_parent_span": {
        "success": True,
        "result": "SpanExportResult.SUCCESS",
        "trace_spans_keys": [],
        "trace_id": "5dfe689c9b5b47b2b8f27874a1b16a48"
    },
    "export_child_span": {
        "success": True,
        "result": "SpanExportResult.SUCCESS",
        "trace_spans_keys": [
            "5dfe689c9b5b47b2b8f27874a1b16a48"
        ],
        "trace_id": "5dfe689c9b5b47b2b8f27874a1b16a48"
    },
    "prepare_trace_direct": {
        "success": False,
        "trace_details_keys": None,
        "trace_file_exists": False,
        "code_zip_exists": False
    },
    "shutdown": {
        "success": True,
        "trace_spans_empty": True,
        "process_complete_trace_called": True
    }
}

@pytest.fixture(scope="module")
def setup_environment():
    """Setup environment variables and initialize catalyst and tracer"""
    load_dotenv()
    
    catalyst = RagaAICatalyst(
        access_key=os.getenv('RAGAAI_CATALYST_ACCESS_KEY'),
        secret_key=os.getenv('RAGAAI_CATALYST_SECRET_KEY'),
        base_url=os.getenv('RAGAAI_CATALYST_BASE_URL')
    )
    
    tracer = Tracer(
        project_name='agentic_tracer_sk_v3',
        dataset_name='pytest_dataset',
        tracer_type="agentic/crewai",
    )
    
    init_tracing(catalyst=catalyst, tracer=tracer)
    return catalyst

@pytest.fixture
def mock_files():
    """Create temporary test files"""
    temp_dir = tempfile.mkdtemp()
    files = []
    for i in range(3):
        file_path = os.path.join(temp_dir, f"test_file_{i}.py")
        with open(file_path, 'w') as f:
            f.write(f'print("Hello World from file {i}")\n')
        files.append(file_path)
    return files

@pytest.fixture
def exporter(mock_files):
    """Create RAGATraceExporter instance with test configuration"""
    return RAGATraceExporter(
        tracer_type="test_type",
        files_to_zip=mock_files,
        project_name="response_collection",
        project_id="test_project_id",
        dataset_name="test_dataset",
        user_details={"name": "Test User"},
        base_url=os.getenv('RAGAAI_CATALYST_BASE_URL'),
        custom_model_cost=None,
        timeout=120
    )

class MockSpan:
    """Mock span class that implements the required interface"""
    def __init__(self, span_data):
        self.span_data = span_data
    
    def to_json(self):
        return json.dumps(self.span_data)

def test_exporter_initialization(exporter):
    """Test RAGATraceExporter initialization"""
    expected_attrs = TEST_RESPONSES["exporter_initialization"]["exporter_attrs"]
    
    assert exporter.tracer_type == expected_attrs["tracer_type"]
    assert exporter.project_name == expected_attrs["project_name"]
    assert exporter.dataset_name == expected_attrs["dataset_name"]
    assert exporter.timeout == expected_attrs["timeout"]
    assert TEST_RESPONSES["exporter_initialization"]["success"] is True

def test_export_parent_span(exporter):
    """Test export method with parent span"""
    # Create mock parent span using the trace_id from test responses
    trace_id = TEST_RESPONSES["export_parent_span"]["trace_id"]
    parent_span_data = {
        "name": "parent_span",
        "context": {
            "trace_id": trace_id,
            "span_id": "parent_span_id"
        },
        "parent_id": None,
        "kind": "INTERNAL"
    }
    parent_mock_span = MockSpan(parent_span_data)
    
    # Patch process_complete_trace to avoid actual processing
    with patch.object(exporter, 'process_complete_trace'):
        result = exporter.export([parent_mock_span])
        
        assert str(result) == TEST_RESPONSES["export_parent_span"]["result"]
        assert TEST_RESPONSES["export_parent_span"]["success"] is True

def test_export_child_span(exporter):
    """Test export method with child span"""
    # Create mock child span using the trace_id from test responses
    trace_id = TEST_RESPONSES["export_child_span"]["trace_id"]
    parent_span_id = "parent_span_id"
    child_span_data = {
        "name": "child_span",
        "context": {
            "trace_id": trace_id,
            "span_id": "child_span_id"
        },
        "parent_id": parent_span_id,
        "kind": "INTERNAL"
    }
    child_mock_span = MockSpan(child_span_data)
    
    # Add the trace_id to trace_spans to simulate parent span already processed
    exporter.trace_spans[trace_id] = []
    
    with patch.object(exporter, 'process_complete_trace'):
        result = exporter.export([child_mock_span])
        
        assert str(result) == TEST_RESPONSES["export_child_span"]["result"]
        assert TEST_RESPONSES["export_child_span"]["success"] is True
        assert trace_id in exporter.trace_spans

def test_prepare_trace_direct(exporter):
    """Test prepare_trace method directly"""
    # Create mock spans
    trace_id = "test_trace_id"
    mock_spans = [
        {
            "name": "test_span",
            "context": {"trace_id": trace_id},
            "kind": "INTERNAL"
        }
    ]
    
    # Mock the internal functions that are called within prepare_trace
    mock_ragaai_trace = {
        "start_time": "2024-01-01T00:00:00",
        "end_time": "2024-01-01T00:00:01",
        "metadata": {"system_info": {}},
        "data": [{}]
    }
    
    with patch('ragaai_catalyst.tracers.utils.trace_json_converter.convert_json_format', 
               return_value=mock_ragaai_trace):
        with patch('ragaai_catalyst.tracers.agentic_tracing.utils.trace_utils.format_interactions',
                  return_value={"workflow": []}):
            result = exporter.prepare_trace(mock_spans, trace_id)
            
            # Assert based on the test responses
            if TEST_RESPONSES["prepare_trace_direct"]["success"]:
                assert result is not None
                assert "trace_file_path" in result
                assert "code_zip_path" in result
                assert "hash_id" in result
            else:
                assert result is None

def test_shutdown(exporter):
    """Test shutdown method"""
    # Add a mock trace to trace_spans
    trace_id = "shutdown_test_trace_id"
    mock_span = {"name": "shutdown_span"}
    exporter.trace_spans[trace_id] = [mock_span]
    
    with patch.object(exporter, 'process_complete_trace') as mock_process:
        exporter.shutdown()
        
        assert len(exporter.trace_spans) == 0
        assert mock_process.called
        assert TEST_RESPONSES["shutdown"]["success"] is True
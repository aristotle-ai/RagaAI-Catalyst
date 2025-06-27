# RagaAI-Catalyst Test Suite

This directory contains the test suite for the RagaAI-Catalyst library, focusing on error handling, safeguarding, and robustness of the workflow components. The tests ensure that the library functions correctly when handling edge cases, invalid inputs, and various error conditions.

## Directory Structure

This directory contains the following test files:

- `test_api_utils.py` - Tests for API utility functions
- `test_dataset.py` - Tests for the Dataset class operations
- `test_distributed.py` - Tests for distributed tracing functionality
- `test_dynamic_trace_exporter.py` - Tests for dynamic trace exporter components
- `test_evaluation.py` - Tests for the Evaluation class
- `test_file_span_exporter.py` - Tests for file span exporter
- `test_llm_utils.py` - Tests for LLM utility functions
- `test_raga_exporter.py` - Tests for the RagaExporter class
- `test_ragaai_trace_Exporter.py` - Tests for the RagaAI trace exporter
- `test_span_attributes.py` - Tests for span attribute handling
- `test_trace_json_convertor.py` - Tests for JSON trace conversion utilities
- `test_tracer.py` - Tests for the core Tracer class
- `test_upload_code.py` - Tests for code upload functionality
- `test_utils.py` - Tests for general utility functions

## Key Features Tested

### Authentication and Initialization
- Tests for proper handling of missing credentials
- Tests for token management and expiration
- Tests for proper initialization of various components

### Error Handling
- Tests for handling of invalid inputs and parameters
- Tests for appropriate error responses when resources don't exist
- Tests for proper error logging and reporting

### Data Management
- Tests for dataset creation, modification, and deletion
- Tests for schema validation and handling
- Tests for row and column operations

### Tracing Functionality
- Tests for span creation and attribute handling
- Tests for distributed tracing across components
- Tests for trace export in various formats

### Evaluation
- Tests for evaluation metrics and configurations
- Tests for threshold handling and validation

### Code Upload
- Tests for code upload process
- Tests for presigned URL generation and usage
- Tests for hash validation and management

## Running the Tests

These tests require proper environment setup with the following environment variables:
```bash
RAGAAI_CATALYST_ACCESS_KEY
RAGAAI_CATALYST_SECRET_KEY
RAGAAI_CATALYST_TOKEN
RAGAAI_CATALYST_BASE_URL
```

The tests can be run using pytest with the following command:

```bash
pytest tests/test_catalyst/workflow-safeguarding/final\ files/
```

## Best Practices Demonstrated

The test suite demonstrates several testing best practices:

- Use of fixtures for common setup and teardown
- Mocking of external services and dependencies
- Comprehensive logging verification
- Error condition testing
- Boundary and edge case testing

These tests ensure that the RagaAI-Catalyst library provides robust error handling and proper operation across a wide range of usage scenarios.
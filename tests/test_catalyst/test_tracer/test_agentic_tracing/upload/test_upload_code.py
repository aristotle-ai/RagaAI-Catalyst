
import pytest
import os
import tempfile
import zipfile
from dotenv import load_dotenv
from ragaai_catalyst import RagaAICatalyst
from ragaai_catalyst.tracers.agentic_tracing.upload.upload_code import (
    upload_code, _fetch_dataset_code_hashes, _fetch_presigned_url
)

# Test responses from upload_code_responses.json
TEST_RESPONSES = {
    "fetch_code_hashes": {
        "success": True,
        "code_hashes": [
            "new_test_hash",
            "test_hash"
        ]
    },
    "fetch_presigned_url": {
        "success": True,
        "url_received": True
    },
    "upload_code": {
        "success": True,
        "result": "Code already exists"
    }
}

@pytest.fixture(scope="module")
def setup_environment():
    """Setup the test environment with RagaAI credentials"""
    load_dotenv()
    catalyst = RagaAICatalyst(
        access_key=os.getenv('RAGAAI_CATALYST_ACCESS_KEY'),
        secret_key=os.getenv('RAGAAI_CATALYST_SECRET_KEY'),
        base_url=os.getenv('RAGAAI_CATALYST_BASE_URL')
    )
    return catalyst

@pytest.fixture
def test_zip_file():
    """Create a temporary test zip file"""
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "test.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        test_file = os.path.join(temp_dir, "test.py")
        with open(test_file, 'w') as f:
            f.write("print('test')")
        zipf.write(test_file, "test.py")
    return zip_path

def test_fetch_code_hashes_exists(setup_environment):
    """Test fetching code hashes for existing dataset"""
    code_hashes = _fetch_dataset_code_hashes(
        project_name='agentic_tracer_sk_v3',
        dataset_name='pytest_dataset',
        base_url=os.getenv('RAGAAI_CATALYST_BASE_URL')
    )
    assert isinstance(code_hashes, list)
    # Should match the code_hashes from upload_code_responses.json
    assert code_hashes == TEST_RESPONSES["fetch_code_hashes"]["code_hashes"]

def test_fetch_presigned_url(setup_environment):
    """Test fetching presigned URL"""
    url = _fetch_presigned_url(
        project_name='agentic_tracer_sk_v3',
        dataset_name='pytest_dataset',
        base_url=os.getenv('RAGAAI_CATALYST_BASE_URL')
    )
    assert isinstance(url, str)
    assert len(url) > 0
    assert TEST_RESPONSES["fetch_presigned_url"]["url_received"] == True

def test_upload_code_new(setup_environment, test_zip_file):
    """Test uploading new code"""
    result = upload_code(
        hash_id="new_test_hash",
        zip_path=test_zip_file,
        project_name='agentic_tracer_sk_v3',
        dataset_name='pytest_dataset',
        base_url=os.getenv('RAGAAI_CATALYST_BASE_URL')
    )
    # Since hash already exists in code_hashes, should return "Code already exists"
    assert result == TEST_RESPONSES["upload_code"]["result"]

def test_upload_code_exists(setup_environment, test_zip_file):
    """Test uploading existing code"""
    result = upload_code(
        hash_id="test_hash",
        zip_path=test_zip_file,
        project_name='agentic_tracer_sk_v3',
        dataset_name='pytest_dataset',
        base_url=os.getenv('RAGAAI_CATALYST_BASE_URL')
    )
    assert result == TEST_RESPONSES["upload_code"]["result"]

def test_complete_flow(setup_environment, test_zip_file):
    """Test the complete flow in sequence"""
    # 1. First verify we can fetch existing hashes
    code_hashes = _fetch_dataset_code_hashes(
        project_name='agentic_tracer_sk_v3',
        dataset_name='pytest_dataset',
        base_url=os.getenv('RAGAAI_CATALYST_BASE_URL')
    )
    assert code_hashes == TEST_RESPONSES["fetch_code_hashes"]["code_hashes"]
    
    # 2. Get presigned URL
    url = _fetch_presigned_url(
        project_name='agentic_tracer_sk_v3',
        dataset_name='pytest_dataset',
        base_url=os.getenv('RAGAAI_CATALYST_BASE_URL')
    )
    assert isinstance(url, str)
    assert len(url) > 0
    
    # 3. Try uploading existing hash
    result = upload_code(
        hash_id="test_hash",
        zip_path=test_zip_file,
        project_name='agentic_tracer_sk_v3',
        dataset_name='pytest_dataset',
        base_url=os.getenv('RAGAAI_CATALYST_BASE_URL')
    )
    assert result == TEST_RESPONSES["upload_code"]["result"]
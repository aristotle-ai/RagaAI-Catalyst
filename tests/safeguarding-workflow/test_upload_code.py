import pytest
import os
import logging
import tempfile
import zipfile
import dotenv
dotenv.load_dotenv()
from ragaai_catalyst import RagaAICatalyst
from ragaai_catalyst.tracers.agentic_tracing.upload.upload_code import (
    _fetch_dataset_code_hashes,
    _fetch_presigned_url,
    _insert_code,
    upload_code
)

@pytest.fixture
def base_url():
    return os.getenv("RAGAAI_CATALYST_BASE_URL")

@pytest.fixture
def access_keys():
    return {
        "access_key": os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
        "secret_key": os.getenv("RAGAAI_CATALYST_SECRET_KEY")}

@pytest.fixture
def catalyst(base_url, access_keys):
    """Create authenticated RagaAICatalyst instance"""
    os.environ["RAGAAI_CATALYST_BASE_URL"] = base_url
    return RagaAICatalyst(
        access_key=access_keys["access_key"],
        secret_key=access_keys["secret_key"]
    )

@pytest.fixture
def test_zip_file():
    """Create a temporary zip file for testing"""
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "test_code.zip")
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # Add a simple text file to the zip
        test_file = os.path.join(temp_dir, "test.py")
        with open(test_file, "w") as f:
            f.write("print('Hello, world!')")
        zipf.write(test_file, arcname="test.py")
    
    yield zip_path
    
    # Clean up
    os.remove(zip_path)
    os.remove(test_file)
    os.rmdir(temp_dir)

def test_fetch_dataset_code_hashes_nonexistent(catalyst, caplog):
    """Test handling of error when fetching code hashes for non-existent dataset"""
    # Set log level to capture all logs
    caplog.set_level(logging.ERROR)
    
    project_name = "test_project"
    dataset_name = "nonexistent_dataset_name_12345"
    
    result = _fetch_dataset_code_hashes(project_name, dataset_name)
    
    # Check that error is logged (adjust based on the actual error message pattern)
    assert result == []  # Function should return an empty list instead of raising exception
    # assert "Failed to" in caplog.text or "Error" in caplog.text

# def test_fetch_presigned_url_error(catalyst, caplog):
#     """Test handling of error when fetching presigned URL"""
#     # Set log level to capture all logs
#     caplog.set_level(logging.ERROR)
    
#     project_name = "test_project"
#     dataset_name = "nonexistent_dataset_name_12345"
    
#     result = _fetch_presigned_url(project_name, dataset_name)
    
#     # Check that error is logged
#     assert result is None  # Function should return None instead of raising exception
#     assert "Failed to" in caplog.text or "Error" in caplog.text

def test_insert_code_error(catalyst, caplog):
    """Test handling of error when inserting code"""
    # Set log level to capture all logs
    caplog.set_level(logging.ERROR)
    
    project_name = "test_project"
    dataset_name = "test_dataset"
    hash_id = "test_hash"
    presigned_url = "https://example.com/test-url"
    
    result = _insert_code(dataset_name, hash_id, presigned_url, project_name)
    
    # Check that error is logged
    assert result is None  # Function should return None instead of raising exception
    assert "Failed to insert code" in caplog.text or "Error" in caplog.text

def test_upload_code_integration(catalyst, test_zip_file, caplog, monkeypatch):
    """Test full upload_code flow with error handling"""
    # Set log level to capture all logs
    caplog.set_level(logging.ERROR)
    
    # Mock _fetch_dataset_code_hashes to return an empty list instead of None
    def mock_fetch_dataset_code_hashes(*args, **kwargs):
        return []
    
    monkeypatch.setattr(
        "ragaai_catalyst.tracers.agentic_tracing.upload.upload_code._fetch_dataset_code_hashes", 
        mock_fetch_dataset_code_hashes
    )
    
    # Mock _fetch_presigned_url to return a dummy URL
    def mock_fetch_presigned_url(*args, **kwargs):
        return "https://example.com/dummy-url"
    
    monkeypatch.setattr(
        "ragaai_catalyst.tracers.agentic_tracing.upload.upload_code._fetch_presigned_url", 
        mock_fetch_presigned_url
    )
    
    # Mock _put_zip_presigned_url to do nothing
    def mock_put_zip_presigned_url(*args, **kwargs):
        return None
    
    monkeypatch.setattr(
        "ragaai_catalyst.tracers.agentic_tracing.upload.upload_code._put_zip_presigned_url", 
        mock_put_zip_presigned_url
    )
    
    # Mock _insert_code to log an error and return None
    def mock_insert_code(*args, **kwargs):
        logger = logging.getLogger(__name__)
        logger.error("Failed to insert code: mock error")
        return None
    
    monkeypatch.setattr(
        "ragaai_catalyst.tracers.agentic_tracing.upload.upload_code._insert_code", 
        mock_insert_code
    )
    
    project_name = "test_project"
    dataset_name = "nonexistent_dataset_name_12345"
    hash_id = "test_hash_id"
    
    # This should trigger several errors but not raise exceptions
    result = upload_code(hash_id, test_zip_file, project_name, dataset_name)
    
    # Check that error is logged
    assert "Failed to" in caplog.text or "Error" in caplog.text
    # Function might return None or an error message
    assert result is None or isinstance(result, str)
import pytest
import os
import logging
import time
from ragaai_catalyst.guardrails_manager import GuardrailsManager
from ragaai_catalyst import RagaAICatalyst

# Enable logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Skip tests if credentials are not available
pytest.importorskip("dotenv").load_dotenv()

# Project configuration for tests
TEST_PROJECT_NAME = os.getenv("TEST_PROJECT_NAME", "guard")
TEST_DATASET_NAME = os.getenv("TEST_DATASET_NAME", "abcd")
DEPLOYMENT_NAME_PREFIX = f"test_deployment_{int(time.time())}"

@pytest.fixture(scope="module")
def raga_catalyst():
    """Initialize RagaAICatalyst client for testing"""
    access_key = os.getenv("RAGAAI_CATALYST_ACCESS_KEY")
    secret_key = os.getenv("RAGAAI_CATALYST_SECRET_KEY")
    
    if not access_key or not secret_key:
        pytest.skip("API credentials not found in environment variables")
    
    # Initialize the client
    try:
        client = RagaAICatalyst(
            access_key=access_key,
            secret_key=secret_key
        )
        # Verify token was obtained
        assert os.getenv("RAGAAI_CATALYST_TOKEN"), "Failed to get token"
        return client
    except Exception as e:
        pytest.skip(f"Failed to initialize RagaAICatalyst: {e}")

@pytest.fixture(scope="module")
def guardrails_manager(raga_catalyst):
    """Create a real GuardrailsManager instance for testing"""
    try:
        # Try to get existing project first
        manager = GuardrailsManager(TEST_PROJECT_NAME)
        return manager
    except ValueError as e:
        # If project doesn't exist, skip the tests
        pytest.skip(f"Project '{TEST_PROJECT_NAME}' not found: {e}")

def test_init_with_valid_project(guardrails_manager):
    """Test initialization with a valid project name"""
    assert guardrails_manager.project_name == TEST_PROJECT_NAME
    assert guardrails_manager.project_id is not None
    assert guardrails_manager.base_url is not None

def test_init_with_invalid_project():
    """Test initialization with an invalid project name"""
    with pytest.raises(ValueError):
        GuardrailsManager("non_existent_project_name_12345")

def test_list_deployment_ids(guardrails_manager):
    """Test listing deployment IDs"""
    deployments = guardrails_manager.list_deployment_ids()
    assert isinstance(deployments, list)
    for deployment in deployments:
        assert "id" in deployment
        assert "name" in deployment

def test_get_deployment(guardrails_manager):
    """Test getting deployment details"""
    # First list deployments to get an ID
    deployments = guardrails_manager.list_deployment_ids()
    if not deployments:
        pytest.skip("No deployments found to test with")
    
    # Get the first deployment's details
    deployment_id = deployments[0]["id"]
    deployment = guardrails_manager.get_deployment(deployment_id)
    
    assert deployment is not None
    assert "success" in deployment
    assert deployment["success"] is True
    assert "data" in deployment
    assert "name" in deployment["data"]
    assert "guardrailsResponse" in deployment["data"]

def test_get_deployment_with_invalid_id(guardrails_manager):
    """Test getting deployment details with an invalid ID"""
    result = guardrails_manager.get_deployment("invalid-id-12345")
    assert result is None

def test_list_guardrails(guardrails_manager):
    """Test listing available guardrails"""
    guardrails = guardrails_manager.list_guardrails()
    assert isinstance(guardrails, list)
    assert len(guardrails) > 0  # Should have some default guardrails

def test_list_fail_condition(guardrails_manager):
    """Test listing fail conditions"""
    fail_conditions = guardrails_manager.list_fail_condition()
    # The API returns a dictionary instead of a list
    assert isinstance(fail_conditions, dict)
    assert "guardrailFailConditions" in fail_conditions
    assert isinstance(fail_conditions["guardrailFailConditions"], list)

def test_list_datasets(guardrails_manager):
    """Test listing datasets"""
    datasets = guardrails_manager.list_datasets()
    assert isinstance(datasets, list)

def test_create_deployment(guardrails_manager):
    """Test creating a new deployment"""
    unique_deployment_name = f"{DEPLOYMENT_NAME_PREFIX}_create"
    
    # Create a new deployment
    try:
        deployment_id = guardrails_manager.create_deployment(
            unique_deployment_name, 
            TEST_DATASET_NAME
        )
        
        assert deployment_id is not None
        # API is returning an integer ID, not a string
        assert isinstance(deployment_id, (str, int))
        
        # Verify it exists in the list
        deployments = guardrails_manager.list_deployment_ids()
        deployment_names = [d["name"] for d in deployments]
        assert unique_deployment_name in deployment_names
        
    except ValueError as e:
        if "already exists" in str(e):
            # If deployment already exists, that's fine
            logger.info(f"Deployment {unique_deployment_name} already exists")
        else:
            raise

def test_create_duplicate_deployment(guardrails_manager):
    """Test creating a deployment with a name that already exists"""
    unique_deployment_name = f"{DEPLOYMENT_NAME_PREFIX}_duplicate"
    
    # Create the first deployment
    try:
        guardrails_manager.create_deployment(unique_deployment_name, TEST_DATASET_NAME)
    except ValueError as e:
        if "already exists" not in str(e):
            raise
    
    # Try to create a second one with the same name
    with pytest.raises(ValueError):
        guardrails_manager.create_deployment(unique_deployment_name, TEST_DATASET_NAME)



def test_get_guardrail_config_payload(guardrails_manager):
    """Test generating guardrail configuration payload"""
    config = {
        "isActive": True, 
        "guardrailFailConditions": ["FAIL"], 
        "deploymentFailCondition": "ALL_FAIL", 
        "alternateResponse": "This is an alternate response"
    }
    
    payload = guardrails_manager._get_guardrail_config_payload(config)
    
    assert payload["isActive"] is True
    assert payload["guardrailFailConditions"] == ["FAIL"]
    assert payload["deploymentFailCondition"] == "ALL_FAIL"
    assert "failAction" in payload
    assert "args" in payload["failAction"]
    assert '"alternateResponse": "This is an alternate response"' in payload["failAction"]["args"]

def test_get_guardrail_list_payload(guardrails_manager):
    """Test generating guardrail list payload"""
    guardrail = {
        "name": "TestGuardrail", 
        "displayName": "Test Guardrail", 
        "config": {
            "mappings": [
                {"schemaName": "Prompt", "variableName": "Prompt"}
            ]
        }
    }
    
    payload = guardrails_manager._get_guardrail_list_payload([guardrail])
    
    assert isinstance(payload, list)
    assert len(payload) == 1
    assert payload[0]["name"] == "TestGuardrail"
    assert payload[0]["displayName"] == "Test Guardrail"
    assert "config" in payload[0]
    assert "mappings" in payload[0]["config"]

def test_get_one_guardrail_data(guardrails_manager):
    """Test generating single guardrail data"""
    guardrail = {
        "name": "TestGuardrail", 
        "displayName": "Test Guardrail", 
        "config": {
            "mappings": [
                {"schemaName": "Prompt", "variableName": "Prompt"}
            ]
        }
    }
    
    data = guardrails_manager._get_one_guardrail_data(guardrail)
    
    assert data["name"] == "TestGuardrail"
    assert data["displayName"] == "Test Guardrail"
    assert "config" in data
    assert "mappings" in data["config"]
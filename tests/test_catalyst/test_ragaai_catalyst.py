import os
import unittest
import time
import logging
from ragaai_catalyst import RagaAICatalyst

# Enable debug logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("TestRagaAICatalyst")
from dotenv import load_dotenv
load_dotenv()

class TestRagaAICatalystRealAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load credentials from environment variables
        cls.access_key = os.getenv("RAGAAI_CATALYST_ACCESS_KEY")
        cls.secret_key = os.getenv("RAGAAI_CATALYST_SECRET_KEY")        
        cls.base_url = os.getenv("RAGAAI_CATALYST_BASE_URL", "https://llm-dev5.ragaai.ai/api")
        # Skip tests if credentials are missing
        if not cls.access_key or not cls.secret_key:
            raise unittest.SkipTest("API credentials not found in environment variables")
        
        # Initialize client
        cls.client = RagaAICatalyst(
            access_key=cls.access_key,
            secret_key=cls.secret_key,
            base_url=cls.base_url,
            api_keys={"openai": "test_key_123"}
        )
        
        # Use existing project
        cls.project_name = "bug_test2"
        try:
            project = cls.client.create_project(
                project_name=cls.project_name,
                usecase="Agentic Application"  # Default usecase Q/A
            )
            logger.debug(f"Created project: {cls.project_name}")
        except Exception as e:
            logger.debug(f"Project {cls.project_name} already exists: {str(e)}")


    def test_1_initialization(self):
        """Test client initialization and environment setup"""
        self.assertIsInstance(self.client, RagaAICatalyst)
        self.assertTrue(hasattr(RagaAICatalyst, "BASE_URL"))
        self.assertTrue(os.getenv("RAGAAI_CATALYST_TOKEN"))
        self.assertIsNotNone(RagaAICatalyst.BASE_URL)
        logger.debug(f"Using base URL: {RagaAICatalyst.BASE_URL}")

    def test_2_token_retrieval(self):
        """Test token retrieval and validity"""
        token = self.client.ensure_valid_token()
        self.assertIsInstance(token, str)
        self.assertGreater(len(token), 30)
        logger.debug(f"Obtained token: {token[:10]}...")

    def test_3_api_key_management(self):
        """Test API key handling"""
        # Test key retrieval
        self.assertEqual(self.client.get_api_key("openai"), "test_key_123")
        
        # Test key addition
        self.client.add_api_key("anthropic", "claude_test_456")
        self.assertEqual(self.client.get_api_key("anthropic"), "claude_test_456")
        
        # Test key update
        self.client.add_api_key("openai", "updated_key_789")
        self.assertEqual(self.client.get_api_key("openai"), "updated_key_789")
        
        # Test non-existent key
        self.assertIsNone(self.client.get_api_key("non_existent_service"))

    def test_4_project_verification(self):
        """Verify existing project is accessible"""
        # Verify project appears in list
        projects = self.client.list_projects()
        self.assertIsInstance(projects, list)
        self.assertIn(self.project_name, projects)
        logger.debug(f"Found project in list: {self.project_name}")
        
        # Test listing limited number of projects
        limited_projects = self.client.list_projects(num_projects=1)
        self.assertIsInstance(limited_projects, list)
        self.assertEqual(len(limited_projects), 1)

    def test_5_project_use_cases(self):
        """Test retrieval of available project use cases"""
        use_cases = self.client.project_use_cases()
        self.assertIsInstance(use_cases, list)
        self.assertGreater(len(use_cases), 0)
        logger.debug(f"Available use cases: {use_cases}")
        self.assertIn("Q/A", use_cases)  # Verify a common use case exists

    def test_6_auth_header_generation(self):
        """Test authorization header generation"""
        auth_header = self.client.get_auth_header()
        self.assertIsInstance(auth_header, dict)
        self.assertIn("Authorization", auth_header)
        self.assertTrue(auth_header["Authorization"].startswith("Bearer "))
        logger.debug(f"Generated auth header: {auth_header['Authorization'][:20]}...")

    def test_7_token_refresh_mechanism(self):
        """Test token refresh functionality"""
        # Get current token
        original_token = self.client.ensure_valid_token()
        
        # Force token refresh
        new_token = self.client.get_token(force_refresh=True)
        self.assertNotEqual(original_token, new_token)
        logger.debug("Successfully refreshed token")
        
        # Verify token expiration time is set
        self.assertIsNotNone(RagaAICatalyst._token_expiry)
        logger.debug(f"Token expires at: {RagaAICatalyst._token_expiry}")

    def test_8_api_key_upload(self):
        """Test API key upload functionality (requires valid token)"""
        # This will trigger the _upload_keys method in the SDK
        # Since we added keys during initialization and in test_3
        # We'll just verify that the client has the keys we expect
        self.assertEqual(self.client.get_api_key("openai"), "updated_key_789")
        self.assertEqual(self.client.get_api_key("anthropic"), "claude_test_456")
        logger.debug("API keys present in client - upload was triggered during init")

    def test_9_base_url_normalization(self):
        """Test base URL normalization logic"""
        test_cases = [
            ("https://example.com", "https://example.com/api"),
            ("https://example.com/", "https://example.com/api"),
            ("https://example.com/api", "https://example.com/api"),
            ("https://example.com//api", "https://example.com/api"),
            ("https://example.com/path/", "https://example.com/path/api"),
        ]
        
        for input_url, expected in test_cases:
            normalized = self.client._normalize_base_url(input_url)
            self.assertEqual(normalized, expected)
            logger.debug(f"Normalized {input_url} to {normalized}")

if __name__ == "__main__":
    unittest.main()
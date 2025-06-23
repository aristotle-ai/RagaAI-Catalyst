import logging
import os
import threading

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.exceptions import PoolError, MaxRetryError, NewConnectionError
from requests.exceptions import ConnectionError, Timeout
from http.client import RemoteDisconnected
from ragaai_catalyst import RagaAICatalyst
import requests

logger = logging.getLogger(__name__)


class SessionManager:
    """Shared session manager with connection pooling for HTTP requests"""
    _instance = None
    _session = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:  # Thread-safe singleton
                if cls._instance is None:  # Double-check locking
                    logger.info("Creating new SessionManager singleton instance")
                    cls._instance = super(SessionManager, cls).__new__(cls)
                    cls._instance._initialize_session()
                else:
                    logger.debug("SessionManager instance already exists, returning existing instance")
        else:
            logger.debug("SessionManager instance exists, returning existing instance")
        return cls._instance

    def _initialize_session(self):
        """Initialize session with connection pooling and retry strategy"""
        logger.info("Initializing HTTP session with connection pooling and retry strategy")
        self._session = requests.Session()

        retry_strategy = Retry(
            total=3,  # number of retries
            connect=3,  # number of retries for connection-related errors
            read=3,  # number of retries for read-related errors
            backoff_factor=0.5,  # wait 0.5, 1, 2... seconds between retries
            status_forcelist=[500, 502, 503, 504]  # HTTP status codes to retry on
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=5,  # number of connection pools to cache (per host)
            pool_maxsize=50,  # maximum number of connections in each pool
            pool_block=True  # Block/wait when pool is full rather than raising error
        )

        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        # Set session-level configuration to handle connection issues
        self._session.headers.update({
            'Connection': 'keep-alive',
            'User-Agent': 'RagaAI-Catalyst/1.0'
        })

        logger.info("HTTP session initialized successfully with adapters mounted for http:// and https://")

        # Warm up connection pool using RagaAICatalyst.BASE_URL
        if os.getenv("RAGAAI_CATALYST_BASE_URL") is not None:
            base_url = os.getenv("RAGAAI_CATALYST_BASE_URL")
            logger.info(f"Warming up connection pool using RagaAICatalyst.BASE_URL: {base_url}")
            self.warm_up_connections(base_url)
        else:
            logger.warning(f"RAGAAI_CATALYST_BASE_URL not available, skipping connection warmup")

    @property
    def session(self):
        if self._session is None:
            logger.warning("Session accessed but not initialized, reinitializing...")
            self._initialize_session()
        return self._session

    def warm_up_connections(self, base_url, num_connections=3):
        """
        Warm up the connection pool by making lightweight requests to healthcheck endpoint.
        This can help prevent RemoteDisconnected errors on initial requests.
        """
        if not self._session:
            return

        # Construct healthcheck URL
        healthcheck_url = f"{base_url.rstrip('/')}/healthcheck"
        logger.info(f"Warming up connection pool with {num_connections} connections to {healthcheck_url}")

        for i in range(num_connections):
            try:
                # Make a lightweight HEAD request to the healthcheck endpoint to warm up the connection
                response = self._session.head(healthcheck_url, timeout=10)
                logger.info(f"Warmup connection {i+1}: Status {response.status_code}")
            except Exception as e:
                logger.warning(f"Warmup connection {i+1} failed (this may be normal): {e}")
                # Ignore other failures during warmup as they're expected
                continue

        logger.info("Connection pool warmup completed")

    def close(self):
        """Close the session"""
        if self._session:
            logger.info("Closing HTTP session")
            self._session.close()
            self._session = None
            logger.info("HTTP session closed successfully")
        else:
            logger.debug("Close called but session was already None")

    def handle_request_exceptions(self, e, operation_name):
        """Handle common request exceptions with appropriate logging"""
        logger.error(f"Exception occurred during {operation_name}")
        if isinstance(e, (PoolError, MaxRetryError)):
            logger.error(f"Connection pool exhausted during {operation_name}: {e}")
        elif isinstance(e, NewConnectionError):
            logger.error(f"Failed to establish new connection during {operation_name}: {e}")
        elif isinstance(e, RemoteDisconnected):
            logger.error(f"Remote connection closed unexpectedly during {operation_name}: {e}")
        elif isinstance(e, ConnectionError):
            logger.error(f"Connection error during {operation_name}: {e}")
        elif isinstance(e, Timeout):
            logger.error(f"Request timeout during {operation_name}: {e}")
        else:
            logger.error(f"Unexpected error during {operation_name}: {e}")

    def make_request_with_retry(self, method, url, **kwargs):
        """
        Make HTTP request with additional retry logic for RemoteDisconnected errors
        that may not be caught by urllib3's retry mechanism.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._session.request(method, url, **kwargs)
                return response
            except (RemoteDisconnected, ConnectionError) as e:
                logger.warning(f"Connection error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    # Re-raise the exception on the last attempt
                    raise
                # Wait before retrying (exponential backoff)
                import time
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)


# Global session manager instance
logger.info("Creating global SessionManager instance")
session_manager = SessionManager()
logger.info(f"Global SessionManager instance created with ID: {id(session_manager)}")

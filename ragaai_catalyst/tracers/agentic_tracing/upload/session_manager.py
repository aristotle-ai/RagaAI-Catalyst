import logging
import threading

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.exceptions import PoolError, MaxRetryError, NewConnectionError
from requests.exceptions import ConnectionError, Timeout, RequestException
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
            backoff_factor=0.5,  # wait 0.5, 1, 2... seconds between retries
            status_forcelist=[500, 502, 503, 504]  # HTTP status codes to retry on
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=1,  # number of connections to keep in the pool
            pool_maxsize=50,  # maximum number of connections in the pool
            pool_block=False  # don't block when pool is full
        )
        logger.debug(f"Configured HTTP adapter: pool_connections={adapter.config.get('pool_connections', 1)}, "
                    f"pool_maxsize={adapter.config.get('pool_maxsize', 50)}, "
                    f"pool_block={adapter.config.get('pool_block', False)}")

        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        logger.info("HTTP session initialized successfully with adapters mounted for http:// and https://")

    @property
    def session(self):
        if self._session is None:
            logger.warning("Session accessed but not initialized, reinitializing...")
            self._initialize_session()
        return self._session

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
        elif isinstance(e, ConnectionError):
            logger.error(f"Connection error during {operation_name}: {e}")
        elif isinstance(e, Timeout):
            logger.error(f"Request timeout during {operation_name}: {e}")
        else:
            logger.error(f"Unexpected error during {operation_name}: {e}")


# Global session manager instance
logger.info("Creating global SessionManager instance")
session_manager = SessionManager()
logger.info(f"Global SessionManager instance created with ID: {id(session_manager)}")

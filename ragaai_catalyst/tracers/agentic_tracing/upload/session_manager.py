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
                    cls._instance = super(SessionManager, cls).__new__(cls)
                    cls._instance._initialize_session()
        return cls._instance

    def _initialize_session(self):
        """Initialize session with connection pooling and retry strategy"""
        self._session = requests.Session()
        retry_strategy = Retry(
            total=3,  # number of retries
            backoff_factor=0.5,  # wait 0.5, 1, 2... seconds between retries
            status_forcelist=[500, 502, 503, 504]  # HTTP status codes to retry on
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=2,  # number of connections to keep in the pool
            pool_maxsize=50,  # maximum number of connections in the pool
            pool_block=True
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    @property
    def session(self):
        return self._session

    def close(self):
        """Close the session"""
        if self._session:
            self._session.close()

    def handle_request_exceptions(self, e, operation_name):
        """Handle common request exceptions with appropriate logging"""
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
session_manager = SessionManager()

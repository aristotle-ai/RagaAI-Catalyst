"""
trace_uploader.py - A dedicated process for handling trace uploads
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
import tempfile
from pathlib import Path
import multiprocessing
import queue
from datetime import datetime
import atexit
import glob
from logging.handlers import RotatingFileHandler
import concurrent.futures
from typing import Dict, Any, Optional
import threading
import uuid


# Set up logging
log_dir = os.path.join(tempfile.gettempdir(), "ragaai_logs")
os.makedirs(log_dir, exist_ok=True)

# Define maximum file size (e.g., 5 MB) and backup count
max_file_size = 5 * 1024 * 1024  # 5 MB
backup_count = 1  # Number of backup files to keep

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            os.path.join(log_dir, "trace_uploader.log"),
            maxBytes=max_file_size,
            backupCount=backup_count
        )
    ]
)
logger = logging.getLogger("trace_uploader")

try:
    from ragaai_catalyst.tracers.agentic_tracing.upload.upload_agentic_traces import UploadAgenticTraces
    from ragaai_catalyst.tracers.agentic_tracing.upload.upload_code import upload_code
    # from ragaai_catalyst.tracers.agentic_tracing.upload.upload_trace_metric import upload_trace_metric
    from ragaai_catalyst.tracers.agentic_tracing.utils.create_dataset_schema import create_dataset_schema_with_trace
    from ragaai_catalyst.tracers.agentic_tracing.upload.session_manager import session_manager
    from ragaai_catalyst import RagaAICatalyst
    IMPORTS_AVAILABLE = True
except ImportError:
    logger.warning("RagaAI Catalyst imports not available - running in test mode")
    IMPORTS_AVAILABLE = False
    session_manager = None

# Define task queue directory
QUEUE_DIR = os.path.join(tempfile.gettempdir(), "ragaai_tasks")
os.makedirs(QUEUE_DIR, exist_ok=True)

# Status codes
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# Global executor for handling uploads
_executor = None
_executor_lock = threading.Lock()
# Dictionary to track futures and their associated task IDs
_futures: Dict[str, Any] = {}
_futures_lock = threading.Lock()

# Dataset creation cache to avoid redundant API calls
_dataset_cache: Dict[str, Dict[str, Any]] = {}
_dataset_cache_lock = threading.Lock()
DATASET_CACHE_DURATION = 600  # 10 minutes in seconds

_cleanup_lock = threading.Lock()
_last_cleanup = 0
CLEANUP_INTERVAL = 300  # 5 minutes

# Thread-safe counter for task IDs
_task_counter = 0
_task_counter_lock = threading.Lock()

def get_executor(max_workers=None):
    """Get or create the thread pool executor"""
    global _executor
    with _executor_lock:
        if _executor is None:
            # Calculate optimal worker count
            if max_workers is None:
                max_workers = min(8, (os.cpu_count() or 1) * 4)

            logger.info(f"Creating ThreadPoolExecutor with {max_workers} workers")
            _executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="trace_uploader"
            )
            # atexit.register(shutdown)
        return _executor


def generate_unique_task_id():
    """Generate a thread-safe unique task ID"""
    global _task_counter

    with _task_counter_lock:
        _task_counter += 1
        counter = _task_counter

    unique_id = str(uuid.uuid4())[:8]  # Short UUID
    return f"task_{int(time.time())}_{os.getpid()}_{counter}_{unique_id}"

def _generate_dataset_cache_key(dataset_name: str, project_name: str, base_url: str) -> str:
    """Generate a unique cache key for dataset creation"""
    return f"{dataset_name}#{project_name}#{base_url}"

def _is_dataset_cached(cache_key: str) -> bool:
    """Check if dataset creation is cached and still valid"""
    with _dataset_cache_lock:
        if cache_key not in _dataset_cache:
            return False

        cache_entry = _dataset_cache[cache_key]
        cache_time = cache_entry.get('timestamp', 0)
        current_time = time.time()

        # Check if cache is still valid (within 10 minutes)
        if current_time - cache_time <= DATASET_CACHE_DURATION:
            logger.info(f"Dataset creation cache hit for key: {cache_key}")
            return True
        else:
            # Cache expired, remove it
            logger.info(f"Dataset creation cache expired for key: {cache_key}")
            del _dataset_cache[cache_key]
            return False

def _cache_dataset_creation(cache_key: str, response: Any) -> None:
    """Cache successful dataset creation"""
    with _dataset_cache_lock:
        _dataset_cache[cache_key] = {
            'timestamp': time.time(),
            'response': response
        }

def _cleanup_expired_cache_entries() -> None:
    """Remove expired cache entries"""
    current_time = time.time()
    with _dataset_cache_lock:
        expired_keys = []
        for cache_key, cache_entry in _dataset_cache.items():
            cache_time = cache_entry.get('timestamp', 0)
            if current_time - cache_time > DATASET_CACHE_DURATION:
                expired_keys.append(cache_key)

        for key in expired_keys:
            del _dataset_cache[key]

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired dataset cache entries")

def process_upload(task_id: str, filepath: str, hash_id: str, zip_path: str, 
                  project_name: str, project_id: str, dataset_name: str, 
                  user_details: Dict[str, Any], base_url: str, tracer_type, timeout=120, fail_on_trace_error=True) -> Dict[str, Any]:
    """
    Process a single upload task
    
    Args:
        task_id: Unique identifier for the task
        filepath: Path to the trace file
        hash_id: Hash ID for the code
        zip_path: Path to the code zip file
        project_name: Project name
        project_id: Project ID
        dataset_name: Dataset name
        user_details: User details dictionary
        base_url: Base URL for API calls
        fail_on_trace_error: If True, raise exception when trace upload fails
    Returns:
        Dict containing status and any error information
    """
    # Correct base_url
    base_url = base_url[0] if isinstance(base_url, tuple) else base_url

    logger.info(f"Processing upload task {task_id}")
    result = {
        "task_id": task_id,
        "status": STATUS_PROCESSING,
        "error": None,
        "start_time": datetime.now().isoformat()
    }
    
    # Save initial status to file
    # with open(filepath, 'r') as f:
    #     data = json.load(f)
    # with open(os.path.join(os.getcwd(), 'agentic_traces.json'), 'w') as f:
    #     json.dump(data, f, default=str, indent=2)
    save_task_status(result)
    
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            error_msg = f"Task filepath does not exist: {filepath}"
            logger.error(error_msg)
            result["status"] = STATUS_FAILED
            result["error"] = error_msg
            save_task_status(result)
            return result

        if not IMPORTS_AVAILABLE:
            logger.warning(f"Test mode: Simulating processing of task {task_id}")
            # time.sleep(2)  # Simulate work
            result["status"] = STATUS_COMPLETED
            save_task_status(result)
            return result
            
        # Step 1: Create dataset schema (with caching)
        logger.info(f"Creating dataset schema for {dataset_name} with base_url: {base_url} and timeout: {timeout}")

        # Generate cache key and check if dataset creation is already cached
        cache_key = _generate_dataset_cache_key(dataset_name, project_name, base_url)

        if _is_dataset_cached(cache_key):
            logger.info(f"Dataset schema creation skipped (cached) for {dataset_name}")
        else:
            try:
                # Clean up expired cache entries periodically
                # _cleanup_expired_cache_entries()

                response = create_dataset_schema_with_trace(
                    dataset_name=dataset_name,
                    project_name=project_name,
                    base_url=base_url,
                    user_details=user_details,
                    timeout=timeout
                )

                if response is None:
                    logger.error(f"Dataset schema creation failed for {dataset_name} - received None response")
                elif hasattr(response, 'status_code') and response.status_code in [200, 201]:
                    logger.info(f"Dataset schema created successfully: {response.status_code}")
                    _cache_dataset_creation(cache_key, response)
                    logger.info(f"Response cached successfully for dataset: {dataset_name} and key: {cache_key}")
                else:
                    logger.warning(f"Dataset schema creation returned unexpected response: {response}")

            except Exception as e:
                logger.error(f"Error creating dataset schema: {e}")
                # Continue with other steps
            
        # Step 2: Upload trace metrics
        # if filepath and os.path.exists(filepath):
        #     logger.info(f"Uploading trace metrics for {filepath} with base_url: {base_url} and timeout: {timeout}")
        #     try:
        #         response = upload_trace_metric(
        #             json_file_path=filepath,
        #             dataset_name=dataset_name,
        #             project_name=project_name,
        #             base_url=base_url,
        #             timeout=timeout
        #         )
        #         logger.info(f"Trace metrics uploaded: {response}")
        #     except Exception as e:  
        #         logger.error(f"Error uploading trace metrics: {e}")
        #         # Continue with other uploads
        # else:
        #     logger.warning(f"Trace file {filepath} not found, skipping metrics upload")
        
        # Step 3: Upload agentic traces
        if filepath and os.path.exists(filepath):
            logger.info(f"Uploading agentic traces for {filepath} with base_url: {base_url} and timeout: {timeout}")
            try:
                upload_traces = UploadAgenticTraces(
                    json_file_path=filepath,
                    project_name=project_name,
                    project_id=project_id,
                    dataset_name=dataset_name,
                    user_detail=user_details,
                    base_url=base_url,   
                    timeout=timeout
                )
                upload_success = upload_traces.upload_agentic_traces()
                if upload_success:
                    logger.info("Agentic traces uploaded successfully")
                else:
                    error_msg = "Agentic traces upload failed"
                    logger.error(error_msg)

                    if fail_on_trace_error:
                        result["status"] = STATUS_FAILED
                        result["error"] = error_msg
                        result["end_time"] = datetime.now().isoformat()
                        save_task_status(result)
            except Exception as e:
                logger.error(f"Error uploading agentic traces: {e}")

                # Continue with code upload
        else:
            error_msg = f"Trace file {filepath} not found"
            logger.warning(f"Trace file {filepath} not found, skipping traces upload")
            if fail_on_trace_error:
                result["status"] = STATUS_FAILED
                result["error"] = error_msg
                result["end_time"] = datetime.now().isoformat()
                save_task_status(result)
                logger.error(error_msg)
        
        # Step 4: Upload code hash
        if tracer_type.startswith("agentic/"):
            logger.info(f"Tracer type '{tracer_type}' matches agentic pattern, proceeding with code upload")
            if hash_id and zip_path and os.path.exists(zip_path):
                logger.info(f"Uploading code hash {hash_id} with base_url: {base_url} and timeout: {timeout}")
                try:
                    response = upload_code(
                        hash_id=hash_id,
                        zip_path=zip_path,
                        project_name=project_name,
                        dataset_name=dataset_name,
                        base_url=base_url,
                        timeout=timeout
                    )
                    if response is None:
                        error_msg = "Code hash not uploaded"
                        logger.error(error_msg)
                    else:
                        logger.info(f"Code hash uploaded successfully: {response}")
                except Exception as e:
                    logger.error(f"Error uploading code hash: {e}")
            else:
                logger.warning(f"Code zip {zip_path} not found, skipping code upload")

        # Mark task as completed
        result["status"] = STATUS_COMPLETED
        result["end_time"] = datetime.now().isoformat()
        logger.info(f"Task {task_id} completed successfully")

    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}")
        result["status"] = STATUS_FAILED
        result["error"] = str(e)
        result["end_time"] = datetime.now().isoformat()
        
    # Save final status
    save_task_status(result)
    return result

def cleanup_completed_futures():
    """Remove completed futures to prevent memory leaks"""
    global _futures, _last_cleanup

    current_time = time.time()
    if current_time - _last_cleanup < CLEANUP_INTERVAL:
        return

    with _cleanup_lock:
        if current_time - _last_cleanup < CLEANUP_INTERVAL:
            return  # Double-check after acquiring lock
        with _futures_lock:
            completed_tasks = []
            for task_id, future in _futures.items():
                if future.done():
                    completed_tasks.append(task_id)

            for task_id in completed_tasks:
                del _futures[task_id]

            _last_cleanup = current_time

            if completed_tasks:
                logger.info(f"Cleaned up {len(completed_tasks)} completed futures")

def save_task_status(task_status: Dict[str, Any]):
    """Save task status to a file"""
    task_id = task_status["task_id"]
    status_path = os.path.join(QUEUE_DIR, f"{task_id}_status.json")
    with open(status_path, "w") as f:
        json.dump(task_status, f, indent=2)

def submit_upload_task(filepath, hash_id, zip_path, project_name, project_id, dataset_name, user_details, base_url,
                       tracer_type, timeout=120):
    """
    Submit a new upload task using futures.
    
    Args:
        filepath: Path to the trace file
        hash_id: Hash ID for the code
        zip_path: Path to the code zip file
        project_name: Project name
        project_id: Project ID
        dataset_name: Dataset name
        user_details: User details dictionary
        base_url: Base URL for API calls
        
    Returns:
        str: Task ID
    """
    logger.info(f"Submitting new upload task for file: {filepath}")
    logger.debug(f"Task details - Project: {project_name}, Dataset: {dataset_name}, Hash: {hash_id}, Base_URL: {base_url}")
    
    # Verify the trace file exists
    if not os.path.exists(filepath):
        logger.error(f"Trace file not found: {filepath}")
        return None

    # Create absolute path to the trace file
    filepath = os.path.abspath(filepath)
    logger.debug(f"Using absolute filepath: {filepath}")

    # Generate a thread-safe unique task ID
    task_id = generate_unique_task_id()
    logger.debug(f"Generated task ID: {task_id}")

    # Function to handle synchronous processing
    def do_sync_processing():
        try:
            logger.info(f"Processing task {task_id} synchronously...")
            result = process_upload(
                task_id=task_id,
                filepath=filepath,
                hash_id=hash_id,
                zip_path=zip_path,
                project_name=project_name,
                project_id=project_id,
                dataset_name=dataset_name,
                user_details=user_details,
                base_url=base_url,
                tracer_type = tracer_type,
                timeout=timeout,
                fail_on_trace_error=True
            )
            logger.info(f"Synchronous processing completed for {task_id}: {result}")
            return task_id
        except Exception as e:
            logger.error(f"Error in synchronous processing: {e}")
            return None
    
    # Try to get the executor
    executor = get_executor()
    if executor is None:
        logger.warning("Executor is None or shutdown in progress, processing synchronously")
        return do_sync_processing()
    # Cleanup completed futures periodically
    # cleanup_completed_futures()
    # Try to submit the task to the executor
    try:
        # Cleanup completed futures periodically
        future = executor.submit(
            process_upload,
            task_id=task_id,
            filepath=filepath,
            hash_id=hash_id,
            zip_path=zip_path,
            project_name=project_name,
            project_id=project_id,
            dataset_name=dataset_name,
            user_details=user_details,
            base_url=base_url,
            tracer_type=tracer_type,
            timeout=timeout,
            fail_on_trace_error=True
        )
        
        # Store the future for later status checks
        with _futures_lock:
            _futures[task_id] = future
        
        # Create initial status
        initial_status = {
            "task_id": task_id,
            "status": STATUS_PENDING,
            "error": None,
            "start_time": datetime.now().isoformat()
        }
        save_task_status(initial_status)
        
        return task_id
    except RuntimeError as e:
        if any(msg in str(e) for msg in
               ("cannot schedule new futures after shutdown", "cannot schedule new futures after interpreter shutdown")):
            logger.warning(f"Executor already shut down, falling back to synchronous processing: {e}")
            return do_sync_processing()
        else:
            logger.error(f"Error submitting task: {e}")
            return None
    except Exception as e:
        logger.error(f"Error submitting task: {e}")
        return None

def get_task_status(task_id):
    """
    Get the status of a task by ID.
    
    Args:
        task_id: Task ID to check
        
    Returns:
        dict: Task status information
    """
    logger.debug(f"Getting status for task {task_id}")
    
    # Check if we have a future for this task
    with _futures_lock:
        future = _futures.get(task_id)
    
    # If we have a future, check its status
    if future:
        if future.done():
            try:
                # Get the result (this will re-raise any exception that occurred)
                result = future.result(timeout=0)
                return result
            except concurrent.futures.TimeoutError:
                return {"status": STATUS_PROCESSING, "error": None}
            except Exception as e:
                logger.error(f"Error retrieving future result for task {task_id}: {e}")
                return {"status": STATUS_FAILED, "error": str(e)}
        else:
            return {"status": STATUS_PROCESSING, "error": None}
    
    # If we don't have a future, try to read from the status file
    status_path = os.path.join(QUEUE_DIR, f"{task_id}_status.json")
    if os.path.exists(status_path):
        try:
            with open(status_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading status file for task {task_id}: {e}")
            return {"status": "unknown", "error": f"Error reading status: {e}"}
    
    return {"status": "unknown", "error": "Task not found"}


def get_upload_queue_status():
    """Get detailed status of upload queue"""
    logger.info("Reached get queue status")
    # with _executor_lock:
    executor = get_executor()

    if executor is None:
        return {
            "total_submitted": 0,
            "pending_uploads": 0,
            "completed_uploads": 0,
            "failed_uploads": 0,
            "thread_pool_queue_size": 0,
            "active_workers": 0,
            "max_workers": 0
        }
    # with _futures_lock:
    pending_count = len([f for f in _futures.values() if not f.done()])
    completed_count = len([f for f in _futures.values() if f.done() and not f.exception()])
    failed_count = len([f for f in _futures.values() if f.done() and f.exception()])

    # Try to get thread pool queue size (may not be available in all Python versions)
    queue_size = 0
    try:
        if hasattr(executor, '_work_queue'):
            queue_size = executor._work_queue.qsize()
    except:
        pass

    return {
        "total_submitted": len(_futures),
        "pending_uploads": pending_count,
        "completed_uploads": completed_count,
        "failed_uploads": failed_count,
        "thread_pool_queue_size": queue_size,
        "active_workers": getattr(executor, '_threads', set()).__len__() if executor else 0,
        "max_workers": executor._max_workers if executor else 0
    }


def shutdown(timeout=120):
    """Enhanced shutdown with manual timeout and progress reporting"""
    logger.debug("Reached shutdown of executor")
    global _executor, _futures
    with _executor_lock:
        if _executor is None:
            logger.debug("Executor is none in shutdown")
            return

    # Log current state
    status = get_upload_queue_status()
    logger.debug(f"Queue status: {status}")
    logger.debug(f"Shutting down uploader. Pending uploads: {status['pending_uploads']}")

    if status['pending_uploads'] > 0:
        logger.debug(f"Waiting up to {timeout}s for {status['pending_uploads']} uploads to complete...")

        start_time = time.time()
        last_report = start_time

        while time.time() - start_time < timeout:
            # Check if all futures are done
            with _futures_lock:
                pending_futures = [f for f in _futures.values() if not f.done()]

                if not pending_futures:
                    logger.info("All uploads completed successfully")
                    break

                # Report progress every 10 seconds
                current_time = time.time()
                if current_time - last_report >= 10:
                    elapsed = current_time - start_time
                    remaining = timeout - elapsed
                    logger.info(f"Still waiting for {len(pending_futures)} uploads to complete. "
                                f"Time remaining: {remaining:.1f}s")
                    last_report = current_time

                # Sleep briefly to avoid busy waiting
                time.sleep(0.5)
        else:
            # Timeout reached
            logger.info("Executor timeout reached")
            with _futures_lock:
                pending_futures = [f for f in _futures.values() if not f.done()]
                logger.debug(f"Shutdown timeout reached. {len(pending_futures)} uploads still pending.")
    else:
        logger.info(f"No pending uploads")

    # Shutdown the executor
    try:
        _executor.shutdown(wait=False)  # Don't wait here since we already waited above
        logger.info("Executor shutdown initiated")
    except Exception as e:
        logger.error(f"Error during executor shutdown: {e}")

    _executor = None

    # Close the session manager to clean up HTTP connections
    if session_manager is not None:
        try:
            session_manager.close()
            logger.info("Session manager closed successfully")
        except Exception as e:
            logger.error(f"Error closing session manager: {e}")

# Register shutdown handler
atexit.register(shutdown)

# For backward compatibility
def ensure_uploader_running():
    """
    Ensure the uploader is running.
    This is a no-op in the futures implementation, but kept for API compatibility.
    """
    get_executor()  # Just ensure the executor is created
    return True

# For backward compatibility with the old daemon mode
def run_daemon():
    """
    Run the uploader as a daemon process.
    This is a no-op in the futures implementation, but kept for API compatibility.
    """
    logger.info("Daemon mode not needed in futures implementation")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trace uploader process")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon process")
    args = parser.parse_args()
    
    if args.daemon:
        logger.info("Daemon mode not needed in futures implementation")
    else:
        logger.info("Interactive mode not needed in futures implementation")

import json
import logging
import os
import tempfile
from dataclasses import asdict
from typing import Optional, Callable, Dict, List

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from ragaai_catalyst.tracers.agentic_tracing.upload.trace_uploader import (
    submit_upload_task,
)
from ragaai_catalyst.tracers.agentic_tracing.utils.system_monitor import SystemMonitor
from ragaai_catalyst.tracers.agentic_tracing.utils.trace_utils import (
    format_interactions,
)
from ragaai_catalyst.tracers.agentic_tracing.utils.zip_list_of_unique_files import (
    zip_list_of_unique_files,
)
from ragaai_catalyst.tracers.utils.trace_json_converter import convert_json_format

logger = logging.getLogger("RagaAICatalyst")
logging_level = (
    logger.setLevel(logging.DEBUG) if os.getenv("DEBUG") == "1" else logging.INFO
)


class TracerJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, bytes):
            try:
                return obj.decode("utf-8")
            except UnicodeDecodeError:
                return str(obj)  # Fallback to string representation
        if hasattr(obj, "to_dict"):  # Handle objects with to_dict method
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            # Filter out None values and handle nested serialization
            return {
                k: v
                for k, v in obj.__dict__.items()
                if v is not None and not k.startswith("_")
            }
        try:
            # Try to convert to a basic type
            return str(obj)
        except:
            return None  # Last resort: return None instead of failing


class RAGATraceExporter(SpanExporter):
    def __init__(
            self,
            project_name: str,
            dataset_name: str,
            base_url: str,
            tracer_type: str,
            files_to_zip: Optional[List[str]] = None,
            project_id: Optional[str] = None,
            user_details: Optional[Dict] = None,
            custom_model_cost: Optional[dict] = None,
            timeout: int = 120,
            post_processor: Optional[Callable] = None,
            max_upload_workers: int = 30,
            user_context: Optional[str] = None,
            user_gt: Optional[str] = None,
            external_id: Optional[str] = None
    ):
        self.trace_spans = dict()
        self.tmp_dir = tempfile.gettempdir()
        self.tracer_type = tracer_type
        self.files_to_zip = files_to_zip
        self.project_name = project_name
        self.project_id = project_id
        self.dataset_name = dataset_name
        self.user_details = user_details
        self.base_url = base_url
        self.custom_model_cost = custom_model_cost
        self.system_monitor = SystemMonitor(dataset_name)
        self.timeout = timeout
        self.post_processor = post_processor
        self.max_upload_workers = max_upload_workers
        self.user_context = user_context
        self.user_gt = user_gt
        self.external_id = external_id

    def export(self, spans):
        for span in spans:
            try:
                span_json = json.loads(span.to_json())
                trace_id = span_json.get("context").get("trace_id")
                if trace_id is None:
                    logger.error("Trace ID is None")
                
                if trace_id not in self.trace_spans:
                    self.trace_spans[trace_id] = list()

                if span_json.get("attributes").get("openinference.span.kind", None) is None:
                    span_json["attributes"]["openinference.span.kind"] = "UNKNOWN"

                # Extract dataset name from span attributes for proper isolation
                dataset_name = self._get_dataset_from_span(span_json)

                self.trace_spans[trace_id].append(span_json)

                if span_json["parent_id"] is None:
                    trace = self.trace_spans[trace_id]
                    try:
                        self.process_complete_trace(trace, trace_id, dataset_name)
                    except Exception as e:
                        logger.error(f"Error processing complete trace: {e}")
                    try:
                        del self.trace_spans[trace_id]
                    except Exception as e:
                        logger.error(f"Error deleting trace: {e}")
            except Exception as e:
                logger.warning(f"Error processing span: {e}")
                continue

        return SpanExportResult.SUCCESS

    def _get_dataset_from_span(self, span_json):
        """
        Extract dataset name from a single span's attributes.
        
        Args:
            span_json: Single span dictionary
            
        Returns:
            str: Dataset name if found, fallback to original dataset_name otherwise
        """
        try:
            attributes = span_json.get('attributes', {})
            dataset = attributes.get('ragaai.dataset')
            
            if dataset:
                logger.debug(f"Found dataset '{dataset}' in span: {span_json.get('name', 'unnamed')}")
                return dataset
            else:
                # Fallback to original dataset if ragaai.dataset not found
                logger.debug(f"No ragaai.dataset found in span: {span_json.get('name', 'unnamed')}, using fallback: {self.dataset_name}")
                return self.dataset_name
                
        except Exception as e:
            logger.error(f"Error extracting dataset from span: {e}")
            return self.dataset_name

    def shutdown(self):
        # Process any remaining traces during shutdown
        logger.debug("Reached shutdown of exporter")
        for trace_id, spans in self.trace_spans.items():
            self.process_complete_trace(spans, trace_id)
        self.trace_spans.clear()

    def process_complete_trace(self, spans, trace_id, dataset_name=None):
        """
        Process a complete trace with the specified dataset.
        
        Args:
            spans: List of span dictionaries for this trace
            trace_id: The trace ID
            dataset_name: The dataset name for this trace (from span attributes)
        """
        # Use the dataset name from span attributes if provided, otherwise fall back to detection
        if dataset_name is None:
            dataset_name = self._get_dataset_from_spans(spans)
        
        if dataset_name and dataset_name != self.dataset_name:
            # Temporarily route to the target dataset
            logger.info(f"Routing trace {trace_id} to dataset: {dataset_name}")
            
            # Store original values
            original_dataset = self.dataset_name
            original_user_details = self.user_details.copy()
            
            try:
                # Update dataset for this trace
                self.dataset_name = dataset_name
                self.user_details["dataset_name"] = dataset_name
                
                # Process with updated dataset
                self._process_trace_with_current_dataset(spans, trace_id, self.dataset_name)
                
            finally:
                # Restore original values
                self.dataset_name = original_dataset
                self.user_details = original_user_details
        else:
            # Use original dataset
            logger.debug(f"Trace {trace_id} using original dataset: {self.dataset_name}")
            self._process_trace_with_current_dataset(spans, trace_id, self.dataset_name)

    def _process_trace_with_current_dataset(self, spans, trace_id, dataset_name):
        """
        Process the trace with the current dataset (original logic from process_complete_trace).
        """
        # Convert the trace to ragaai trace format
        try:
            ragaai_trace_details = self.prepare_trace(spans, trace_id)
        except Exception as e:
            print(f"Error converting trace {trace_id}: {e}")
            return  # Exit early if conversion fails

        # Check if trace details are None (conversion failed)
        if ragaai_trace_details is None:
            logger.error(f"Cannot upload trace {trace_id}: conversion failed and returned None")
            return  # Exit early if conversion failed

        # Upload the trace if upload_trace function is provided
        try:
            if self.post_processor != None:
                ragaai_trace_details['trace_file_path'] = self.post_processor(ragaai_trace_details['trace_file_path'])
            self.upload_trace(ragaai_trace_details, trace_id, dataset_name)
        except Exception as e:
            print(f"Error uploading trace {trace_id}: {e}")

    def _get_dataset_from_spans(self, spans):
        """
        Extract dataset name from span attributes.
        
        Args:
            spans: List of span dictionaries
            
        Returns:
            str: Dataset name if found, None otherwise
        """
        try:
            # Look through all spans for the ragaai.dataset attribute
            for span in spans:
                attributes = span.get('attributes', {})
                dataset = attributes.get('ragaai.dataset')
                
                if dataset:
                    logger.debug(f"Found dataset '{dataset}' in span: {span.get('name', 'unnamed')}")
                    return dataset
            
            # No dataset attribute found
            logger.debug("No ragaai.dataset attribute found in any span")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting dataset from spans: {e}")
            return None

    def prepare_trace(self, spans, trace_id):
        try:
            try:
                ragaai_trace = convert_json_format(spans, self.custom_model_cost, self.user_context, self.user_gt,
                                                   self.external_id)
            except Exception as e:
                print(f"Error in convert_json_format function: {trace_id}: {e}")
                return None

            try:
                interactions = format_interactions(ragaai_trace)
                ragaai_trace["workflow"] = interactions['workflow']
            except Exception as e:
                print(f"Error in format_interactions function: {trace_id}: {e}")
                return None

            try:
                # Add source code hash
                hash_id, zip_path = zip_list_of_unique_files(
                    self.files_to_zip, output_dir=self.tmp_dir
                )
            except Exception as e:
                print(f"Error in zip_list_of_unique_files function: {trace_id}: {e}")
                return None

            try:
                ragaai_trace["metadata"]["system_info"] = asdict(self.system_monitor.get_system_info())
                ragaai_trace["metadata"]["resources"] = asdict(self.system_monitor.get_resources())
            except Exception as e:
                print(f"Error in get_system_info or get_resources function: {trace_id}: {e}")
                return None

            try:
                ragaai_trace["metadata"]["system_info"]["source_code"] = hash_id
            except Exception as e:
                print(f"Error in adding source code hash: {trace_id}: {e}")
                return None

            try:
                ragaai_trace["data"][0]["start_time"] = ragaai_trace["start_time"]
                ragaai_trace["data"][0]["end_time"] = ragaai_trace["end_time"]
            except Exception as e:
                print(f"Error in adding start_time or end_time: {trace_id}: {e}")
                return None

            try:
                ragaai_trace["project_name"] = self.project_name
            except Exception as e:
                print(f"Error in adding project name: {trace_id}: {e}")
                return None

            try:
                # Add tracer type to the trace
                ragaai_trace["tracer_type"] = self.tracer_type
            except Exception as e:
                print(f"Error in adding tracer type: {trace_id}: {e}")
                return None

            # Add user passed metadata to the trace
            try:
                logger.debug("Started adding user passed metadata")

                metadata = (
                    self.user_details.get("trace_user_detail", {}).get("metadata", {})
                    if self.user_details else {}
                )

                if isinstance(metadata, dict):
                    for key, value in metadata.items():
                        if key not in {"log_source", "recorded_on"}:
                            ragaai_trace.setdefault("metadata", {})[key] = value

                logger.debug("Completed adding user passed metadata")
            except Exception as e:
                print(f"Error in adding metadata: {trace_id}: {e}")
                return None

            try:
                # Save the trace_json 
                trace_file_path = os.path.join(self.tmp_dir, f"{trace_id}.json")
                with open(trace_file_path, "w") as file:
                    json.dump(ragaai_trace, file, cls=TracerJSONEncoder, indent=2)
                with open(os.path.join(os.getcwd(), f"{trace_id}.json"), 'w') as f:
                    json.dump(ragaai_trace, f, cls=TracerJSONEncoder, indent=2)
            except Exception as e:
                print(f"Error in saving trace json: {trace_id}: {e}")
                return None

            return {
                'trace_file_path': trace_file_path,
                'code_zip_path': zip_path,
                'hash_id': hash_id
            }
        except Exception as e:
            print(f"Error converting trace {trace_id}: {str(e)}")
            return None

    def upload_trace(self, ragaai_trace_details, trace_id, dataset_name):
        filepath = ragaai_trace_details['trace_file_path']
        hash_id = ragaai_trace_details['hash_id']
        zip_path = ragaai_trace_details['code_zip_path']
        self.upload_task_id = submit_upload_task(
            filepath=filepath,
            hash_id=hash_id,
            zip_path=zip_path,
            project_name=self.project_name,
            project_id=self.project_id,
            dataset_name=dataset_name,
            user_details=self.user_details,
            base_url=self.base_url,
            tracer_type=self.tracer_type,
            timeout=self.timeout
        )

        logger.info(f"Submitted upload task with ID: {self.upload_task_id}")

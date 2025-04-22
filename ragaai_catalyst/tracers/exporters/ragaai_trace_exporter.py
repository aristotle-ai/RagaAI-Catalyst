import os
import json
import tempfile
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
import logging
from dataclasses import asdict
from ragaai_catalyst.tracers.utils.trace_json_converter import convert_json_format
from ragaai_catalyst.tracers.agentic_tracing.tracers.base import TracerJSONEncoder
from ragaai_catalyst.tracers.agentic_tracing.utils.system_monitor import SystemMonitor
from ragaai_catalyst.tracers.agentic_tracing.upload.trace_uploader import submit_upload_task
from ragaai_catalyst.tracers.agentic_tracing.utils.zip_list_of_unique_files import zip_list_of_unique_files
from ragaai_catalyst.tracers.agentic_tracing.utils.trace_utils import format_interactions
from ragaai_catalyst.tracers.utils.rag_trace_json_converter import rag_trace_json_converter
from ragaai_catalyst.tracers.utils.convert_langchain_callbacks_output import convert_langchain_callbacks_output
from ragaai_catalyst.tracers.upload_traces import UploadTraces
import datetime
import logging
import asyncio
import concurrent.futures
from functools import partial

logger = logging.getLogger("RagaAICatalyst")
logging_level = (
    logger.setLevel(logging.DEBUG) if os.getenv("DEBUG") == "1" else logging.INFO
)


class RAGATraceExporter(SpanExporter):
    def __init__(self, tracer_type, files_to_zip, project_name, project_id, dataset_name, user_details, base_url, custom_model_cost, timeout=120, post_processor = None, max_upload_workers = 30,user_context = None, external_id=None):
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
        self.external_id = external_id

    def export(self, spans):
        for span in spans:
            try:
                span_json = json.loads(span.to_json())
                trace_id = span_json.get("context").get("trace_id")
                if trace_id is None:
                    raise Exception("Trace ID is None")

                if trace_id not in self.trace_spans:
                    self.trace_spans[trace_id] = list()

                self.trace_spans[trace_id].append(span_json)

                if span_json["parent_id"] is None:
                    trace = self.trace_spans[trace_id]
                    try:
                        self.process_complete_trace(trace, trace_id)
                    except Exception as e:
                        raise Exception(f"Error processing complete trace: {e}")
                    try:
                        del self.trace_spans[trace_id]
                    except Exception as e:
                        raise Exception(f"Error deleting trace: {e}")
            except Exception as e:
                logger.warning(f"Error processing span: {e}")
                continue

        return SpanExportResult.SUCCESS

    def shutdown(self):
        # Process any remaining traces during shutdown
        for trace_id, spans in self.trace_spans.items():
            self.process_complete_trace(spans, trace_id)
        self.trace_spans.clear()

    def process_complete_trace(self, spans, trace_id):
        # Convert the trace to ragaai trace format
        try:
            if self.tracer_type == "langchain":
                ragaai_trace_details, additional_metadata = self.prepare_rag_trace(spans, trace_id)
            else:
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
            if self.post_processor!=None:
                ragaai_trace_details['trace_file_path'] = self.post_processor(ragaai_trace_details['trace_file_path'])
            if self.tracer_type == "langchain":
                # Check if we're already in an event loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # We're in a running event loop (like in Colab/Jupyter)
                        # Create a future and run the coroutine
                        future = asyncio.ensure_future(self.upload_rag_trace(ragaai_trace_details, additional_metadata, trace_id))
                        # We don't wait for it to complete as this would block the event loop
                        logger.info(f"Scheduled async upload for trace {trace_id} in existing event loop")
                    else:
                        # No running event loop, use asyncio.run()
                        asyncio.run(self.upload_rag_trace(ragaai_trace_details, additional_metadata, trace_id))
                except RuntimeError:
                    # No event loop exists, create one
                    asyncio.run(self.upload_rag_trace(ragaai_trace_details, additional_metadata, trace_id))
            else:
                self.upload_trace(ragaai_trace_details, trace_id)
        except Exception as e:
            print(f"Error uploading trace {trace_id}: {e}")

    def prepare_trace(self, spans, trace_id):
        try:
            try:
                ragaai_trace = convert_json_format(spans, self.custom_model_cost)   
            except Exception as e:
                print(f"Error in convert_json_format function: {trace_id}: {e}")
                return None
            
            # try:
            #     interactions = format_interactions(ragaai_trace)         
            #     ragaai_trace["workflow"] = interactions['workflow']
            # except Exception as e:
            #     print(f"Error in format_interactions function: {trace_id}: {e}")
            #     return None

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
                # Save the trace_json 
                trace_file_path = os.path.join(self.tmp_dir, f"{trace_id}.json")
                with open(trace_file_path, "w") as file:
                    json.dump(ragaai_trace, file, cls=TracerJSONEncoder, indent=2)
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

    def upload_trace(self, ragaai_trace_details, trace_id):
        filepath = ragaai_trace_details['trace_file_path']
        hash_id = ragaai_trace_details['hash_id']
        zip_path = ragaai_trace_details['code_zip_path']
        self.upload_task_id = submit_upload_task(
                filepath=filepath,
                hash_id=hash_id,
                zip_path=zip_path,
                project_name=self.project_name,
                project_id=self.project_id,
                dataset_name=self.dataset_name,
                user_details=self.user_details,
                base_url=self.base_url,
                timeout=self.timeout
            )

        logger.info(f"Submitted upload task with ID: {self.upload_task_id}")
    
    async def upload_rag_trace(self, ragaai_trace, additional_metadata, trace_id):
        try:
            ragaai_trace[0]['external_id'] = self.external_id
            trace_file_path = os.path.join(self.tmp_dir, f"{trace_id}.json")
            with open(trace_file_path, 'w') as f:
                json.dump(ragaai_trace, f, indent=2)
            logger.info(f"Trace file saved at {trace_file_path}")

            # Create a ThreadPoolExecutor with max_workers=30
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_upload_workers) as executor:
                # Create a partial function with all the necessary arguments
                upload_func = partial(
                    UploadTraces(
                        json_file_path=trace_file_path,
                        project_name=self.project_name,
                        project_id=self.project_id,
                        dataset_name=self.dataset_name,
                        user_detail=self.user_details,
                        base_url=self.base_url
                    ).upload_traces,
                    additional_metadata_keys=additional_metadata
                )
                
                # Implement retry logic - attempt upload up to 3 times
                max_retries = 3
                retry_count = 0
                last_exception = None
                
                while retry_count < max_retries:
                    try:
                        # Submit the task to the executor and get a future
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(executor, upload_func)
                        
                        logger.info(f"Successfully uploaded rag trace {trace_id} on attempt {retry_count + 1}")
                        return  # Exit the method if upload is successful
                    except Exception as e:
                        retry_count += 1
                        last_exception = e
                        logger.warning(f"Attempt {retry_count} to upload rag trace {trace_id} failed: {str(e)}")
                        
                        if retry_count < max_retries:
                            # Add a small delay before retrying (exponential backoff)
                            await asyncio.sleep(2 ** retry_count)  # 2, 4, 8 seconds
                
                # If we've exhausted all retries, log the error
                logger.error(f"Failed to upload rag trace {trace_id} after {max_retries} attempts. Last error: {str(last_exception)}")
        except Exception as e:
            logger.error(f"Error preparing rag trace {trace_id} for upload: {str(e)}")
    
    def prepare_rag_trace(self, spans, trace_id):
        try:            
            ragaai_trace, additional_metadata = rag_trace_json_converter(spans, self.custom_model_cost, trace_id, self.user_details, self.tracer_type,self.user_context)
            ragaai_trace["metadata"]["recorded_on"] = datetime.datetime.now().astimezone().isoformat()
            ragaai_trace["metadata"]["log_source"] = "langchain_tracer"

            if True:
                converted_ragaai_trace = convert_langchain_callbacks_output(ragaai_trace, self.project_name, ragaai_trace["metadata"], ragaai_trace["pipeline"])
            else:
                converted_ragaai_trace = ragaai_trace
            
            return converted_ragaai_trace, additional_metadata
            
        except Exception as e:
            logger.error(f"Error converting trace {trace_id}: {str(e)}")
            return None

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
from ragaai_catalyst.tracers.utils.convert_llama_instru_callback import convert_llamaindex_instrumentation_to_callback
from ragaai_catalyst.tracers.utils.convert_langchain_callbacks_output import convert_langchain_callbacks_output
from ragaai_catalyst.tracers.upload_traces import UploadTraces
import datetime
import logging

logger = logging.getLogger("RagaAICatalyst")
logging_level = (
    logger.setLevel(logging.DEBUG) if os.getenv("DEBUG") == "1" else logging.INFO
)


class RAGATraceExporter(SpanExporter):
    def __init__(self, tracer_type, files_to_zip, project_name, project_id, dataset_name, user_details, base_url, custom_model_cost, timeout=120):
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

    def export(self, spans):
        for span in spans:
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
                    logger.debug(f"Processing span for trace {trace_id}, parent_id: {span_json['parent_id']}")
                    self.process_complete_trace(trace, trace_id)
                except Exception as e:
                    raise Exception(f"Error processing complete trace: {e}")
                try:
                    del self.trace_spans[trace_id]
                except Exception as e:
                    raise Exception(f"Error deleting trace: {e}")

        return SpanExportResult.SUCCESS

    def shutdown(self):
        # Process any remaining traces during shutdown
        for trace_id, spans in self.trace_spans.items():
            self.process_complete_trace(spans, trace_id)
        self.trace_spans.clear()

    def process_complete_trace(self, spans, trace_id):
        # Convert the trace to ragaai trace format
        try:
            if self.tracer_type == "rag/langchain":
                ragaai_trace_details, additional_metadata = self.prepare_rag_trace(spans, trace_id)
            elif self.tracer_type == "rag/llamaindex":
                ragaai_trace_details, additional_metadata = self.prepare_rag_trace(spans, trace_id)
            else:
                ragaai_trace_details = self.prepare_trace(spans, trace_id)
        except Exception as e:
            print(f"Error converting trace {trace_id}: {e}")
        
        # Upload the trace if upload_trace function is provided
        try:
            if self.tracer_type == "rag/langchain":
                self.upload_rag_trace(ragaai_trace_details, additional_metadata, trace_id)
            elif self.tracer_type == "rag/llamaindex":
                self.upload_rag_trace(ragaai_trace_details, additional_metadata, trace_id)

            else:
                None
                # self.upload_trace(ragaai_trace_details, trace_id)
        except Exception as e:
            print(f"Error uploading trace {trace_id}: {e}")

    def prepare_trace(self, spans, trace_id):
        try:
            ragaai_trace = convert_json_format(spans, self.custom_model_cost)   
            interactions = format_interactions(ragaai_trace)         
            ragaai_trace["workflow"] = interactions['workflow']

            # Add source code hash
            hash_id, zip_path = zip_list_of_unique_files(
                self.files_to_zip, output_dir=self.tmp_dir
            )

            ragaai_trace["metadata"]["system_info"] = asdict(self.system_monitor.get_system_info())
            ragaai_trace["metadata"]["resources"] = asdict(self.system_monitor.get_resources())
            ragaai_trace["metadata"]["system_info"]["source_code"] = hash_id

            ragaai_trace["data"][0]["start_time"] = ragaai_trace["start_time"]
            ragaai_trace["data"][0]["end_time"] = ragaai_trace["end_time"]

            ragaai_trace["project_name"] = self.project_name
            
            # Save the trace_json 
            trace_file_path = os.path.join(self.tmp_dir, f"{trace_id}.json")
            with open(trace_file_path, "w") as file:
                json.dump(ragaai_trace, file, cls=TracerJSONEncoder, indent=2)

            return {
                'trace_file_path': trace_file_path,
                'code_zip_path': zip_path,
                'hash_id': hash_id
            }
        except Exception as e:
            logger.error(f"Error converting trace {trace_id}: {str(e)}")
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
    
    def prepare_rag_trace(self, spans, trace_id):
        try:
            with open(os.path.join(os.getcwd(), "llamaindex_traces_debug.json"), "w") as f:
                json.dump(spans, f, cls=TracerJSONEncoder, indent=4)
            ragaai_trace, additional_metadata = rag_trace_json_converter(spans, self.custom_model_cost, trace_id, self.user_details, self.tracer_type)            
            ragaai_trace["metadata"]["recorded_on"] = datetime.datetime.now().astimezone().isoformat()
            if self.tracer_type == "rag/langchain": 
                ragaai_trace["metadata"]["log_source"] = "langchain_tracer"
            elif self.tracer_type == "rag/llamaindex": 
                ragaai_trace["metadata"]["log_source"] = "llamaindex_tracer"
            if True:
                converted_ragaai_trace = convert_langchain_callbacks_output(ragaai_trace, self.project_name, ragaai_trace["metadata"], ragaai_trace["pipeline"])
            else:
                converted_ragaai_trace = ragaai_trace
            
            return converted_ragaai_trace, additional_metadata
            
        except Exception as e:
            logger.error(f"Error converting trace {trace_id}: {str(e)}")
            return None
    
    def upload_rag_trace(self, ragaai_trace, additional_metadata, trace_id):
        if ragaai_trace and len(ragaai_trace) > 0:
            for trace in ragaai_trace[0].get('traces', []):
                attributes = trace.get('attributes', {})
                
                # Extract prompt
                if 'traceloop.entity.input' in attributes:
                    input_data = json.loads(attributes['traceloop.entity.input'])
                    if 'kwargs' in input_data and 'input' in input_data['kwargs']:
                        prompt = input_data['kwargs']['input']
                
                # Extract context
                if 'traceloop.entity.input' in attributes:
                    input_data = json.loads(attributes['traceloop.entity.input'])
                    if 'kwargs' in input_data and 'context' in input_data['kwargs']:
                        context = input_data['kwargs']['context']
                
                # Extract response
                if 'gen_ai.completion.0.content' in attributes:
                    response = attributes['gen_ai.completion.0.content']
        if prompt is not None and response is not None and context is not None:
            try:
                trace_file_path = os.path.join(os.getcwd(), "final_rag_traces.json")
                with open(trace_file_path, 'w') as f:
                    json.dump(ragaai_trace, f, indent=2)
                UploadTraces(json_file_path=trace_file_path,
                            project_name=self.project_name,
                            project_id=self.project_id,
                            dataset_name=self.dataset_name,
                            user_detail=self.user_details,
                            base_url=self.base_url
                            ).upload_traces(additional_metadata_keys=additional_metadata)
            except Exception as e:
                logger.error(f"Error uploading rag trace {trace_id}: {str(e)}")

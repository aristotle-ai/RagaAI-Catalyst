from configparser import InterpolationMissingOptionError
import json
from datetime import datetime
from typing import Any, Optional, Dict, List, ClassVar
from pydantic import Field
# from treelib import Tree

from llama_index.core.instrumentation.span import SimpleSpan
from llama_index.core.instrumentation.span_handlers.base import BaseSpanHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.span_handlers import SimpleSpanHandler

from llama_index.core.instrumentation.events.agent import (
    AgentChatWithStepStartEvent,
    AgentChatWithStepEndEvent,
    AgentRunStepStartEvent,
    AgentRunStepEndEvent,
    AgentToolCallEvent,
)
from llama_index.core.instrumentation.events.chat_engine import (
    StreamChatErrorEvent,
    StreamChatDeltaReceivedEvent,
)
from llama_index.core.instrumentation.events.embedding import (
    EmbeddingStartEvent,
    EmbeddingEndEvent,
)
from llama_index.core.instrumentation.events.llm import (
    LLMPredictEndEvent,
    LLMPredictStartEvent,
    LLMStructuredPredictEndEvent,
    LLMStructuredPredictStartEvent,
    LLMCompletionEndEvent,
    LLMCompletionStartEvent,
    LLMChatEndEvent,
    LLMChatStartEvent,
    LLMChatInProgressEvent,
)
from llama_index.core.instrumentation.events.query import (
    QueryStartEvent,
    QueryEndEvent,
)
from llama_index.core.instrumentation.events.rerank import (
    ReRankStartEvent,
    ReRankEndEvent,
)
from llama_index.core.instrumentation.events.retrieval import (
    RetrievalStartEvent,
    RetrievalEndEvent,
)
from llama_index.core.instrumentation.events.span import (
    SpanDropEvent,
)
from llama_index.core.instrumentation.events.synthesis import (
    SynthesizeStartEvent,
    SynthesizeEndEvent,
    GetResponseEndEvent,
    GetResponseStartEvent,
)

import uuid

from .utils.extraction_logic_llama_index import extract_llama_index_data
from .utils.convert_llama_instru_callback import convert_llamaindex_instrumentation_to_callback

class EventHandler(BaseEventHandler):
    """LlamaIndex event handler for RagaAI Catalyst.

    This event handler captures all LlamaIndex events and stores them for later processing.
    It supports a wide range of event types:
    
    1. LLM Events:
       - LLMPredictStartEvent/EndEvent
       - LLMStructuredPredictStartEvent/EndEvent
       - LLMCompletionStartEvent/EndEvent
       - LLMChatStartEvent/EndEvent/InProgressEvent
    
    2. Agent Events:
       - AgentChatWithStepStartEvent/EndEvent
       - AgentRunStepStartEvent/EndEvent
       - AgentToolCallEvent
       
    3. Retrieval Events:
       - RetrievalStartEvent/EndEvent
       
    4. Query Events:
       - QueryStartEvent/EndEvent
       
    5. Embedding Events:
       - EmbeddingStartEvent/EndEvent
       
    6. Reranking Events:
       - ReRankStartEvent/EndEvent
       
    7. Synthesis Events:
       - SynthesizeStartEvent/EndEvent
       - GetResponseStartEvent/EndEvent
       
    8. Streaming Events:
       - StreamChatErrorEvent
       - StreamChatDeltaReceivedEvent
       
    9. Span Events:
       - SpanDropEvent

    Captured events include metadata about inputs, outputs, timestamps, and relationships
    between different operations in the LlamaIndex workflow.
    """

    events: List[BaseEvent] = []
    current_trace: List[Dict[str, Any]] = []  # Store events for the current trace
    # Keep track of parent-child relationships between events and spans
    event_parents: Dict[str, str] = {}
    span_events: Dict[str, List[str]] = {}


    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "EventHandler"

    def handle(self, event: BaseEvent) -> None:
        """Logic for handling all LlamaIndex event types.
        
        This method captures the event details and extracts relevant information
        based on the event type. Different event types have different attributes
        and require specialized extraction logic.
        
        Args:
            event: The LlamaIndex event to process
        """
        # Track the event
        self.events.append(event)
        
        # Record parent-child relationships
        if hasattr(event, "span_id") and event.span_id:
            # Add this event to its span's list
            if event.span_id not in self.span_events:
                self.span_events[event.span_id] = []
            self.span_events[event.span_id].append(event.id_)
            
            # If this event has a parent span, record it
            if hasattr(event, "parent_span_id") and event.parent_span_id:
                self.event_parents[event.id_] = event.parent_span_id
        
        # Build the event metadata
        event_details = {
            "id": event.id_,
            "type": event.class_name(),
            "timestamp": event.timestamp.isoformat() if hasattr(event, "timestamp") else datetime.now().isoformat(),
            "span_id": event.span_id if hasattr(event, "span_id") else None,
            "parent_span_id": event.parent_span_id if hasattr(event, "parent_span_id") else None,
        }
        
        # === LLM Events ===
        if isinstance(event, (LLMPredictStartEvent, LLMStructuredPredictStartEvent, 
                            LLMCompletionStartEvent, LLMChatStartEvent)):
            # Handle start of LLM operations
            if hasattr(event, "model_name"):
                event_details["model_name"] = event.model_name
            if hasattr(event, "prompt"):
                event_details["prompt"] = event.prompt
            
        elif isinstance(event, (LLMPredictEndEvent, LLMStructuredPredictEndEvent, 
                                LLMCompletionEndEvent, LLMChatEndEvent)):
            # Handle end of LLM operations
            if hasattr(event, "response") and event.response:
                if hasattr(event.response, "message"):
                    event_details["response"] = {
                        "role": event.response.message.role,
                        "content": event.response.message.content
                    }
                elif hasattr(event.response, "text"):
                    event_details["response"] = {"text": event.response.text}
                    
                # Add token usage if available
                if hasattr(event.response, "additional_kwargs") and event.response.additional_kwargs:
                    if "token_usage" in event.response.additional_kwargs:
                        event_details["token_usage"] = event.response.additional_kwargs["token_usage"]
                        
            # Add params if available
            if hasattr(event, "params"):
                event_details["params"] = event.params
            
        # === LLM Streaming Events ===
        elif isinstance(event, StreamChatDeltaReceivedEvent):
            # Handle streaming chat events (token by token)
            if hasattr(event, "delta") and event.delta:
                if hasattr(event.delta, "content"):
                    event_details["delta"] = {"content": event.delta.content}
                
        elif isinstance(event, StreamChatErrorEvent):
            # Handle streaming error events
            if hasattr(event, "error"):
                event_details["error"] = str(event.error)
        
        # === Agent Events ===
        elif isinstance(event, (AgentChatWithStepStartEvent, AgentRunStepStartEvent)):
            # Handle agent step start
            if hasattr(event, "agent"):
                event_details["agent_type"] = type(event.agent).__name__
            if hasattr(event, "query"):
                event_details["query"] = event.query
            
        elif isinstance(event, (AgentChatWithStepEndEvent, AgentRunStepEndEvent)):
            # Handle agent step end
            if hasattr(event, "response"):
                event_details["response"] = event.response

        elif isinstance(event, AgentToolCallEvent):
            # Handle tool calls by agents
            if hasattr(event, "tool_name"):
                event_details["tool_name"] = event.tool_name
            if hasattr(event, "input"):
                event_details["input"] = event.input
            if hasattr(event, "output"):
                event_details["output"] = event.output
            
        # === Retrieval Events ===
        elif isinstance(event, RetrievalStartEvent):
            # Handle retrieval start
            if hasattr(event, "query"):
                event_details["query"] = event.query
            
        elif isinstance(event, RetrievalEndEvent):
            # Handle retrieval end
            if hasattr(event, "nodes"):
                nodes_data = []
                for node in event.nodes:
                    node_data = {"text": node.text}
                    if hasattr(node, "metadata"):
                        node_data["metadata"] = node.metadata
                    nodes_data.append(node_data)
                event_details["nodes"] = nodes_data
                
        # === Embedding Events ===
        elif isinstance(event, EmbeddingStartEvent):
            # Handle embedding start
            if hasattr(event, "texts"):
                event_details["texts"] = event.texts[:10]  # Limit to first 10 texts
                event_details["text_count"] = len(event.texts)
            
        elif isinstance(event, EmbeddingEndEvent):
            # Handle embedding end
            if hasattr(event, "embeddings"):
                event_details["embedding_count"] = len(event.embeddings)
                # Don't include actual embeddings as they're large and not very human-readable
                
        # === Reranking Events ===
        elif isinstance(event, ReRankStartEvent):
            # Handle reranking start
            if hasattr(event, "query"):
                event_details["query"] = event.query
            if hasattr(event, "docs"):
                event_details["doc_count"] = len(event.docs)
            
        elif isinstance(event, ReRankEndEvent):
            # Handle reranking end
            if hasattr(event, "ranked_nodes"):
                event_details["ranked_doc_count"] = len(event.ranked_nodes)
                
        # === Synthesis Events ===
        elif isinstance(event, (SynthesizeStartEvent, GetResponseStartEvent)):
            # Handle synthesis/response start
            if hasattr(event, "query"):
                event_details["query"] = event.query
            
        elif isinstance(event, (SynthesizeEndEvent, GetResponseEndEvent)):
            # Handle synthesis/response end
            if hasattr(event, "response"):
                event_details["response"] = event.response
                
        # === Span Events ===
        elif isinstance(event, SpanDropEvent):
            # Handle span drop events
            if hasattr(event, "error"):
                event_details["error"] = str(event.error)
        
        # Add timing information
        event_details["timestamp"] = datetime.now().isoformat()
        
        # Store the event
        self.current_trace.append(event_details)


    def _get_events_by_span(self) -> Dict[str, List[BaseEvent]]:
        events_by_span: Dict[str, List[BaseEvent]] = {}
        for event in self.events:
            if event.span_id in events_by_span:
                events_by_span[event.span_id].append(event)
            else:
                events_by_span[event.span_id] = [event]
        return events_by_span

    # def _get_event_span_trees(self) -> List[Tree]:
    #     events_by_span = self._get_events_by_span()

    #     trees = []
    #     tree = Tree()

    #     for span, sorted_events in events_by_span.items():
    #         # create root node i.e. span node
    #         tree.create_node(
    #             tag=f"{span} (SPAN)",
    #             identifier=span,
    #             parent=None,
    #             data=sorted_events[0].timestamp,
    #         )

    #         for event in sorted_events:
    #             tree.create_node(
    #                 tag=f"{event.class_name()}: {event.id_}",
    #                 identifier=event.id_,
    #                 parent=event.span_id,
    #                 data=event.timestamp,
    #             )

    #         trees.append(tree)
    #         tree = Tree()
    #     return trees

    # def print_event_span_trees(self) -> None:
    #     """Method for viewing trace trees."""
    #     trees = self._get_event_span_trees()
    #     for tree in trees:
    #         print(
    #             tree.show(
    #                 stdout=False, sorting=True, key=lambda node: node.data
    #             )
    #         )
    #         print("")


class SpanHandler(BaseSpanHandler[SimpleSpan]):
    # span_dict = {}
    span_dict: ClassVar[Dict[str, List[SimpleSpan]]] = {}

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "SpanHandler"

    def new_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[SimpleSpan]:
        """Create a span."""
        # logic for creating a new MyCustomSpan
        if id_ not in self.span_dict:
            self.span_dict[id_] = []
        self.span_dict[id_].append(
            SimpleSpan(id_=id_, parent_id=parent_span_id)
        )

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Logic for preparing to exit a span."""
        pass
        # if id in self.span_dict:
        #    return self.span_dict[id].pop()

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Any:
        """Logic for preparing to drop a span."""
        pass
        # if id in self.span_dict:
        #    return self.span_dict[id].pop()


class LlamaIndexInstrumentationTracer:
    def __init__(self, user_detail):
        """Initialize the LlamaIndexTracer with handlers but don't start tracing yet."""
        # Initialize the root dispatcher
        self.root_dispatcher = get_dispatcher()

        # Initialize handlers
        self.json_event_handler = EventHandler()
        self.span_handler = SpanHandler()
        self.simple_span_handler = SimpleSpanHandler()

        self.is_tracing = False  # Flag to check if tracing is active

        self.user_detail = user_detail

    def start(self):
        """Start tracing by registering handlers."""
        if self.is_tracing:
            print("Tracing is already active.")
            return

        # Register handlers
        self.root_dispatcher.add_span_handler(self.span_handler)
        self.root_dispatcher.add_span_handler(self.simple_span_handler)
        self.root_dispatcher.add_event_handler(self.json_event_handler)

        self.is_tracing = True
        print("Tracing started.")

    def stop(self):
        """Stop tracing by unregistering handlers."""
        if not self.is_tracing:
            print("Tracing is not active.")
            return

        # Get converted callback data from stopping instrumentation
        # This will provide events in the standardized component format
        if hasattr(self, "handler_refs") and self.handler_refs:
            converted_callbacks = stop_llamaindex_instrumentation(self, self.handler_refs)
            if converted_callbacks:
                # Write converted data to file for reference and debugging
                with open('llamaindex_components.json', 'w') as f:
                    json.dump(converted_callbacks, f, default=str, indent=4)
                
                # Set tracing active flag to False
                self.is_tracing = False
                return converted_callbacks
                
        # If we didn't get valid converted data, create a simple structure
        final_trace = {
            "project_id": self.user_detail["project_id"],
            "trace_id": str(uuid.uuid4()),
            "session_id": None,
            "trace_type": "llamaindex",
            "metadata": self.user_detail["trace_user_detail"]["metadata"],
            "pipeline": self.user_detail["trace_user_detail"]["pipeline"],
            "components": []  # Empty components means no events were captured
        }
        
        # Set tracing active flag to False
        self.is_tracing = False
        return [final_trace]


def init_llamaindex_instrumentation(tracer):
    """
    Initialize LlamaIndex instrumentation with the provided tracer.
    
    This function registers event and span handlers with the LlamaIndex
    dispatcher to capture and trace all LlamaIndex events. These events include:
    - LLM events (chat, predict, completion)
    - Agent events (chat steps, tool calls)
    - Tool events
    - Retrieval events
    - Query events
    - Embedding events
    - Reranking events
    - Synthesis events
    
    The function is automatically called when using a Tracer as a context manager:
    
    ```python
    with tracer:
        # Your LlamaIndex code here
    ```
    
    Args:
        tracer: The Tracer instance to use for instrumentation
        
    Returns:
        The initialized handler references that can be used to stop instrumentation
    """
    from llama_index.core.instrumentation import get_dispatcher
    
    # Initialize handlers
    json_event_handler = EventHandler()
    span_handler = SpanHandler()
    
    # Get the root dispatcher
    root_dispatcher = get_dispatcher()
    
    # Register handlers
    root_dispatcher.register_event_handler(json_event_handler)
    root_dispatcher.register_span_handler(span_handler)
    
    # Patch the Workflow class for better tracing
    workflow_originals = None
    try:
        from .integrations.workflow import patch_workflow_class
        workflow_originals = patch_workflow_class()
    except (ImportError, Exception) as e:
        print(f"Warning: Could not patch Workflow class: {e}")
    
    # Store a reference to handlers and return it for later cleanup
    return {
        "event_handler": json_event_handler,
        "span_handler": span_handler,
        "dispatcher": root_dispatcher,
        "workflow_originals": workflow_originals
    }

def stop_llamaindex_instrumentation(tracer, handler_refs):
    """
    Stop LlamaIndex instrumentation and clean up resources.
    
    This function unregisters the event and span handlers from the LlamaIndex
    dispatcher and converts the collected events to the format expected by
    the tracing system.
    
    This function is automatically called when exiting a Tracer context manager:
    
    ```python
    with tracer:
        # Your LlamaIndex code here
    # Instrumentation is stopped and cleaned up here
    ```
    
    Args:
        tracer: The Tracer instance used for instrumentation
        handler_refs: The handler references returned by init_llamaindex_instrumentation
    
    Returns:
        Converted callback data if successful, None otherwise
    """
    if not handler_refs:
        return
    
    # Extract handlers from references
    event_handler = handler_refs.get("event_handler")
    span_handler = handler_refs.get("span_handler")
    dispatcher = handler_refs.get("dispatcher")
    workflow_originals = handler_refs.get("workflow_originals")
    
    # Unregister handlers
    if dispatcher and event_handler:
        dispatcher.unregister_event_handler(event_handler)
    
    if dispatcher and span_handler:
        dispatcher.unregister_span_handler(span_handler)
        
    # Unpatch Workflow class if it was patched
    if workflow_originals:
        try:
            from .integrations.workflow import unpatch_workflow_class
            unpatch_workflow_class(workflow_originals)
        except (ImportError, Exception) as e:
            print(f"Warning: Could not unpatch Workflow class: {e}")
    
    # Convert collected data to callback format if needed
    if event_handler and hasattr(event_handler, "events") and event_handler.events:
        try:
            # Pass the entire event_handler object to preserve parent-child relationships
            callbacks_data = convert_llamaindex_instrumentation_to_callback(
                [event_handler], tracer.user_detail
            )
            return callbacks_data
        except Exception as e:
            print(f"Error converting instrumentation data: {e}")
            return None

def extract_llama_index_data(data):
    """
    Extract LlamaIndex instrumentation data from the captured events.
    
    This function takes the raw event data from LlamaIndex instrumentation
    and extracts the key information needed for tracing and analysis.
    
    Args:
        data: The raw data array containing LlamaIndex events
        
    Returns:
        Structured event data in a standardized format suitable for processing
    """
    if not data or not isinstance(data, list) or not data[0].get("components"):
        return []
    
    # Extract events from the EventHandler
    from .utils.convert_llama_instru_callback import convert_llamaindex_instrumentation_to_callback
    return data
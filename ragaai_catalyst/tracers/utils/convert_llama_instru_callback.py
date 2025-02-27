import uuid
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

def convert_llamaindex_instrumentation_to_callback(events, user_detail=None):
    """
    Convert LlamaIndex events to standardized component format for tracing.
    
    This function transforms LlamaIndex events into the standardized component formats
    used by the RagaAI Catalyst tracing system, ensuring all data is properly captured
    and formatted according to the required schema.
    
    Args:
        events: List of LlamaIndex event objects captured during operation
        user_detail: Optional user details for the trace
        
    Returns:
        Dictionary with standardized trace data format
    """
    if not events:
        return None
    
    # Check if we received event_handler instead of raw events
    if len(events) == 1 and hasattr(events[0], 'events'):
        # We got an event handler, extract the events and parent tracking
        event_handler = events[0]
        events = event_handler.events
        event_parents = getattr(event_handler, 'event_parents', {})
        span_events = getattr(event_handler, 'span_events', {})
    else:
        # No parent tracking available
        event_parents = {}
        span_events = {}
    
    project_id = user_detail.get("project_id", "default_project") if user_detail else "default_project"
    trace_id = str(uuid.uuid4())
    
    # Initialize the trace data structure
    trace_data = {
        "trace_id": trace_id,
        "project_id": project_id,
        "session_id": None,
        "trace_type": "llamaindex",
        "metadata": user_detail.get("trace_user_detail", {}).get("metadata", {}) if user_detail else {},
        "pipeline": user_detail.get("trace_user_detail", {}).get("pipeline", []) if user_detail else [],
        "components": []
    }
    
    # Create a mapping to track component IDs by event ID for parent-child relationships
    component_ids = {}
    created_components = {}
    
    # First pass: create all components
    for event in events:
        event_type = event.__class__.__name__
        event_id = getattr(event, "id_", str(uuid.uuid4()))
        component = None
        
        # Handle based on event type
        if "LLM" in event_type:
            # Convert to LLM component format
            component = create_llm_component_from_event(event, trace_id)
        elif "Agent" in event_type:
            # Convert to Agent component format
            component = create_agent_component_from_event(event, trace_id)
        elif "Tool" in event_type or "Retrieve" in event_type:
            # Convert to Tool component format
            component = create_tool_component_from_event(event, trace_id)
        else:
            # For other types, create a custom component
            component = create_custom_component_from_event(event, trace_id)
            
        if component:
            component_ids[event_id] = component["id"]
            created_components[component["id"]] = component
            trace_data["components"].append(component)
    
    # Second pass: establish parent-child relationships
    for event_id, parent_id in event_parents.items():
        if event_id in component_ids and parent_id in component_ids:
            child_component_id = component_ids[event_id]
            parent_component_id = component_ids[parent_id]
            
            if child_component_id in created_components:
                created_components[child_component_id]["parent_id"] = parent_component_id
                
                # If parent is an agent and child is a tool, add to children list
                parent = created_components.get(parent_component_id)
                child = created_components.get(child_component_id)
                
                if parent and child and parent.get("type") == "agent" and child.get("type") in ["tool", "llm"]:
                    if "data" in parent and "children" in parent["data"]:
                        parent["data"]["children"].append(child_component_id)
    
    return [trace_data]

def create_llm_component_from_event(event, trace_id):
    """
    Create an LLM component from a LlamaIndex event.
    
    Maps LlamaIndex LLM events to the standardized LLM component format.
    """
    component_id = str(uuid.uuid4())
    hash_id = str(uuid.uuid4())
    event_payload = getattr(event, "payload", {})
    
    # Extract event-specific details
    is_streaming = "Stream" in event.__class__.__name__ or "InProgress" in event.__class__.__name__
    is_delta = isinstance(event, StreamChatDeltaReceivedEvent) if 'StreamChatDeltaReceivedEvent' in globals() else False
    
    # Extract input data
    input_data = {}
    if hasattr(event, "messages"):
        input_data["messages"] = event.messages
    elif hasattr(event, "prompt"):
        input_data["prompt"] = event.prompt
    else:
        # Try to get from payload
        input_data = {
            "input": str(event_payload.get("messages", event_payload.get("prompt", "")))
        }
    
    # Extract output data
    output_data = {}
    if hasattr(event, "response"):
        if hasattr(event.response, "message"):
            output_data["response"] = {
                "role": event.response.message.role if hasattr(event.response.message, "role") else "assistant",
                "content": event.response.message.content
            }
        elif hasattr(event.response, "text"):
            output_data["response"] = {"text": event.response.text}
        else:
            output_data["response"] = str(event.response)
    elif hasattr(event, "completion"):
        output_data["completion"] = event.completion
    elif hasattr(event, "delta") and is_streaming:
        # Handle streaming delta content
        if hasattr(event.delta, "content"):
            output_data["delta"] = {"content": event.delta.content}
        elif hasattr(event.delta, "delta"):
            output_data["delta"] = {"content": event.delta.delta}
        else:
            output_data["delta"] = {"content": str(event.delta)}
    else:
        # Try to get from payload
        output_data = {
            "output": str(event_payload.get("response", event_payload.get("completion", "")))
        }
    
    # Extract token usage
    usage = {}
    if hasattr(event, "token_usage"):
        usage = event.token_usage
    else:
        # Try to get from response additional_kwargs
        if hasattr(event, "response") and hasattr(event.response, "additional_kwargs"):
            token_usage = event.response.additional_kwargs.get("token_usage", {})
            if token_usage:
                usage = {
                    "prompt_tokens": token_usage.get("prompt_tokens", 0),
                    "completion_tokens": token_usage.get("completion_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0)
                }
        
        # Try to get from payload
        if not usage:
            token_usage = event_payload.get("token_usage", {})
            if token_usage:
                usage = {
                    "prompt_tokens": token_usage.get("prompt_tokens", 0),
                    "completion_tokens": token_usage.get("completion_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0)
                }
    
    # Extract cost data
    cost = {}
    # Try to estimate cost if not available directly
    if usage:
        # Default cost rates (could be refined based on model)
        input_cost_rate = 0.0001  # $0.0001 per token
        output_cost_rate = 0.0002  # $0.0002 per token
        cost = {
            "prompt_cost": usage.get("prompt_tokens", 0) * input_cost_rate,
            "completion_cost": usage.get("completion_tokens", 0) * output_cost_rate,
            "total_cost": (usage.get("prompt_tokens", 0) * input_cost_rate) + 
                         (usage.get("completion_tokens", 0) * output_cost_rate)
        }
    
    # Extract model details
    model = ""
    if hasattr(event, "model_name"):
        model = event.model_name
    elif hasattr(event, "llm"):
        model = getattr(event.llm, "model_name", str(event.llm))
    else:
        # Try from payload
        model = event_payload.get("model", "")
    
    # Extract parameters
    parameters = {}
    if hasattr(event, "params"):
        parameters = event.params
    else:
        # Common LLM parameters to look for
        for param in ["temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty"]:
            if hasattr(event, param):
                parameters[param] = getattr(event, param)
    
    # Set timing information
    start_time = getattr(event, "start_time", datetime.now().astimezone()).isoformat()
    if "Start" in event.__class__.__name__:
        end_time = None  # End event will come later
    elif is_delta and is_streaming:
        # For streaming deltas, we don't have a proper end time yet
        end_time = None
    else:
        end_time = getattr(event, "end_time", datetime.now().astimezone()).isoformat()
    
    # Error handling
    error = None
    if hasattr(event, "error") and event.error:
        error = {
            "type": type(event.error).__name__,
            "message": str(event.error),
            "details": {}
        }
    
    # Create the standardized component
    return {
        "id": component_id,
        "hash_id": hash_id,
        "source_hash_id": None,
        "type": "llm",
        "name": f"{model}_call" if model else "llm_call",
        "start_time": start_time,
        "end_time": end_time or datetime.now().astimezone().isoformat(),
        "error": error,
        "parent_id": None,  # Would be set based on context if available
        "info": {
            "model": model,
            "version": "1.0.0",
            "memory_used": 0,  # Not typically tracked by LlamaIndex
            "cost": cost,
            "tokens": usage,
            "tags": ["llamaindex", "streaming" if is_streaming else "non-streaming"],
            **parameters,
        },
        "extra_info": parameters,
        "data": {
            "input": input_data,
            "output": output_data,
            "memory_used": 0,
        },
        "metrics": [],
        "network_calls": [],
        "interactions": []
    }

def create_agent_component_from_event(event, trace_id):
    """
    Create an Agent component from a LlamaIndex event.
    
    Maps LlamaIndex Agent events to the standardized Agent component format.
    """
    component_id = str(uuid.uuid4())
    hash_id = str(uuid.uuid4())
    event_payload = getattr(event, "payload", {})
    
    # Extract input/output data
    input_data = {}
    if hasattr(event, "query"):
        input_data["query"] = event.query
    else:
        # Try to get from payload
        input_data = {
            "input": event_payload.get("query", event_payload.get("input", ""))
        }
    
    output_data = {}
    if hasattr(event, "response"):
        output_data["response"] = event.response
    else:
        # Try to get from payload
        output_data = {
            "output": event_payload.get("response", "")
        }
    
    # Set timing information
    start_time = getattr(event, "start_time", datetime.now().astimezone()).isoformat()
    if "Start" in event.__class__.__name__:
        end_time = None  # End event will come later
    else:
        end_time = getattr(event, "end_time", datetime.now().astimezone()).isoformat()
    
    # Extract agent type
    agent_type = "unknown"
    if hasattr(event, "agent_type"):
        agent_type = event.agent_type
    elif hasattr(event, "agent"):
        agent_type = event.agent.__class__.__name__
    else:
        # Try from class name
        agent_type = event.__class__.__name__.replace("Event", "").replace("Agent", "")
    
    # Error handling
    error = None
    if hasattr(event, "error") and event.error:
        error = {
            "type": type(event.error).__name__,
            "message": str(event.error),
            "details": {}
        }
    
    # Create the standardized component
    return {
        "id": component_id,
        "hash_id": hash_id,
        "source_hash_id": None,
        "type": "agent",
        "name": f"{agent_type}_agent",
        "start_time": start_time,
        "end_time": end_time or datetime.now().astimezone().isoformat(),
        "error": error,
        "parent_id": None,  # Would be set based on context if available
        "info": {
            "agent_type": agent_type,
            "version": "1.0.0",
            "capabilities": ["reasoning", "tool_use"],
            "memory_used": 0,
            "tags": ["llamaindex"],
        },
        "data": {
            "input": input_data,
            "output": output_data,
            "children": [],  # Would be populated with tool calls
        },
        "metrics": [],
        "network_calls": [],
        "interactions": []
    }

def create_tool_component_from_event(event, trace_id):
    """
    Create a Tool component from a LlamaIndex event.
    
    Maps LlamaIndex Tool and Retrieval events to the standardized Tool component format.
    """
    component_id = str(uuid.uuid4())
    hash_id = str(uuid.uuid4())
    event_payload = getattr(event, "payload", {})
    
    # Extract tool type from event
    tool_type = "unknown"
    if "Retrieve" in event.__class__.__name__:
        tool_type = "retriever"
    elif "ReRank" in event.__class__.__name__:
        tool_type = "reranker"
    elif "Embed" in event.__class__.__name__:
        tool_type = "embedding"
    elif hasattr(event, "tool_name"):
        tool_type = event.tool_name
    else:
        # Try to infer from class name
        tool_type = event.__class__.__name__.replace("Event", "").replace("Tool", "")
    
    # Extract input/output data
    input_data = {}
    if hasattr(event, "query"):
        input_data["query"] = event.query
    elif hasattr(event, "inputs"):
        input_data["inputs"] = event.inputs
    else:
        # Try to get from payload
        input_data = {
            "input": event_payload.get("query", event_payload.get("inputs", ""))
        }
    
    output_data = {}
    if hasattr(event, "nodes"):
        output_data["nodes"] = event.nodes
    elif hasattr(event, "response"):
        output_data["response"] = event.response
    elif hasattr(event, "result"):
        output_data["result"] = event.result
    else:
        # Try to get from payload
        output_data = {
            "output": event_payload.get("nodes", event_payload.get("response", event_payload.get("result", "")))
        }
    
    # Set timing information
    start_time = getattr(event, "start_time", datetime.now().astimezone()).isoformat()
    if "Start" in event.__class__.__name__:
        end_time = None  # End event will come later
    else:
        end_time = getattr(event, "end_time", datetime.now().astimezone()).isoformat()
    
    # Error handling
    error = None
    if hasattr(event, "error") and event.error:
        error = {
            "type": type(event.error).__name__,
            "message": str(event.error),
            "details": {}
        }
    
    # Create the standardized component
    return {
        "id": component_id,
        "hash_id": hash_id,
        "source_hash_id": None,
        "type": "tool",
        "name": f"{tool_type}_tool",
        "start_time": start_time,
        "end_time": end_time or datetime.now().astimezone().isoformat(),
        "error": error,
        "parent_id": None,  # Would be set based on context if available
        "info": {
            "tool_type": tool_type,
            "version": "1.0.0",
            "memory_used": 0,
            "tags": ["llamaindex"],
        },
        "data": {
            "input": input_data,
            "output": output_data,
            "memory_used": 0,
        },
        "metrics": [],
        "network_calls": [],
        "interactions": []
    }

def create_custom_component_from_event(event, trace_id):
    """
    Create a Custom component from any other LlamaIndex event.
    
    Maps miscellaneous LlamaIndex events to the standardized Custom component format.
    """
    component_id = str(uuid.uuid4())
    hash_id = str(uuid.uuid4())
    event_payload = getattr(event, "payload", {})
    
    # Extract event type for the name
    custom_type = event.__class__.__name__.replace("Event", "")
    
    # Extract any available input/output data
    input_data = {}
    output_data = {}
    
    # Try to gather input data from common attributes
    for attr in ["query", "prompt", "messages", "inputs"]:
        if hasattr(event, attr):
            input_data[attr] = getattr(event, attr)
    
    # If no attributes found, use the entire payload as input
    if not input_data and event_payload:
        input_data = {"payload": event_payload}
    
    # Try to gather output data from common attributes
    for attr in ["response", "result", "completion", "output", "nodes"]:
        if hasattr(event, attr):
            output_data[attr] = getattr(event, attr)
    
    # Set timing information
    start_time = getattr(event, "start_time", datetime.now().astimezone()).isoformat()
    if "Start" in event.__class__.__name__:
        end_time = None  # End event will come later
    else:
        end_time = getattr(event, "end_time", datetime.now().astimezone()).isoformat()
    
    # Error handling
    error = None
    if hasattr(event, "error") and event.error:
        error = {
            "type": type(event.error).__name__,
            "message": str(event.error),
            "details": {}
        }
    
    # Create the standardized component
    return {
        "id": component_id,
        "hash_id": hash_id,
        "source_hash_id": None,
        "type": "custom",
        "name": f"{custom_type}_operation",
        "start_time": start_time,
        "end_time": end_time or datetime.now().astimezone().isoformat(),
        "error": error,
        "parent_id": None,  # Would be set based on context if available
        "info": {
            "custom_type": custom_type,
            "version": "1.0.0",
            "memory_used": 0
        },
        "data": {
            "input": input_data,
            "output": output_data,
            "memory_used": 0,
            "variable_traces": []
        },
        "network_calls": [],
        "interactions": []
    }
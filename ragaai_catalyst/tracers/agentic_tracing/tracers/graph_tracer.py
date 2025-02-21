from typing import Optional, Any, Dict, List
import uuid
from datetime import datetime
import psutil
from .base import BaseTracer
from ..utils.unique_decorator import generate_unique_hash
import sys
import functools
import wrapt

class GraphTracerMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_graph_id = None
        self.auto_instrument_graph = False

    def instrument_graph_calls(self):
        """Enable graph execution instrumentation"""
        self.auto_instrument_graph = True
        
        if "langgraph.graph" in sys.modules:
            self.patch_langgraph_graph(sys.modules["langgraph.graph"])
            
    def patch_langgraph_graph(self, module):
        """Patch the LangGraph graph class to track execution"""
        original_graph = module.StateGraph.compile
        
        @functools.wraps(original_graph) 
        def wrapped_compile(graph_self, *args, **kwargs):
            graph = original_graph(graph_self, *args, **kwargs)
            original_stream = graph.stream
            
            @functools.wraps(original_stream)
            def wrapped_stream(*args, **kwargs):
                for arg in args:
                    if arg.get("configurable"):
                        config = arg
                events = original_stream(*args, **kwargs)
                events_object = []
                processed_events = []
                for event in events:
                    events_object.append(event)
                    for message in event['messages']:
                        if hasattr(message, 'content') and message.content:  # Ensure 'message' has the 'content' attribute and it's not empty
                            processed_event = {
                                "id": getattr(message, 'id', None),  # Safely get 'id' or return None if not found
                                "message": message.content,
                                "tool_calls": getattr(message, 'tool_calls', None),  # Safely get 'tool_calls' or return None if not found
                                "additional_kwargs": getattr(message, 'additional_kwargs', None),  # Safely get 'additional_kwargs' or return None
                                "response_metadata": getattr(message, 'response_metadata', None),  # Safely get 'response_metadata' or return None
                                "type": type(message).__name__,  # Get the class name of the message
                                "usage_metadata": getattr(message, 'usage_metadata', None)  # Safely get 'usage_metadata' or return None
                            }
                            processed_events.append(processed_event)
                state = list(graph.get_state(config))
                graph_info = graph.get_graph()
                nodes, edges = graph_info.nodes, graph_info.edges
                metadata = {
                    "state": state,
                    "events": events_object,
                    "config": config,
                    "nodes": nodes,
                    "edges": edges,
                }
                
                self.trace_graph(name="graph", metadata=metadata)
                return events_object
            graph.stream = wrapped_stream
            return graph

        module.StateGraph.compile = wrapped_compile

    def create_graph_component(self, component_id: str, hash_id: str, name: str, 
                            state: Dict, nodes: Dict, edges: Dict, events: List,
                            start_time: str, metadata: Dict = None) -> Dict:
        end_time = datetime.now().astimezone().isoformat()
        
        sanitized_state = self.sanitize_graph_data(state)
        sanitized_nodes = self.sanitize_graph_data(nodes)
        sanitized_edges = self.sanitize_graph_data(edges)
        sanitized_events = self.sanitize_graph_data(events)
        
        component = {
            "id": component_id,
            "hash_id": hash_id,
            "type": "langgraph/graph",
            "name": name,
            "start_time": start_time,
            "end_time": end_time,
            "info": None,
            "source_hash_id": None,
            "parent_id": None,
            "data": {
                "state": sanitized_state,
                "node_info": sanitized_nodes,
                "edge_info": sanitized_edges,
                "events": sanitized_events,
                "metadata": metadata or {}
            }
        }
        
        return component

    def trace_graph(
        self,
        name: str = None,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Trace a graph execution step with detailed node and memory information"""
        
        if not self.is_active or not self.auto_instrument_graph:
            return

        component_id = str(uuid.uuid4())
        start_time = datetime.now().astimezone().isoformat()
        
        # Enhanced metadata with node information
        state = metadata.get("state", {})
        events = metadata.get("events", [])
        nodes = metadata.get("nodes", [])
        edges = metadata.get("edges", [])
        del metadata['state']
        del metadata['nodes']
        del metadata['edges']
        del metadata['events']
        enhanced_metadata = metadata or {}

        graph_component = self.create_graph_component(
            component_id=component_id,
            hash_id=generate_unique_hash(name, enhanced_metadata),
            name=name,
            state=state,
            nodes=nodes,
            edges = edges,
            events = events,
            start_time=start_time,
            metadata=enhanced_metadata
        )
    
        if tags:
            graph_component["tags"] = tags

        self.add_component(graph_component)
        
    def sanitize_graph_data(self, data: Any) -> Any:
        """
        Sanitize graph data by converting complex objects to serializable format.

        Args:
            data: Any data structure that needs sanitization
            
        Returns:
            Sanitized data structure with complex objects converted to strings or appropriate structures
        """
        if isinstance(data, dict):
            return {k: self.sanitize_graph_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_graph_data(item) for item in data]
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        elif hasattr(data, 'to_dict'):
            return self.sanitize_graph_data(data.to_dict())
        elif hasattr(data, '__dict__'):
            # This block is modified to handle specific custom types like HumanMessage, AIMessage, etc.
            sanitized = {attr: self.sanitize_graph_data(getattr(data, attr)) for attr in vars(data) if not attr.startswith('_')}
            sanitized['type'] = data.__class__.__name__
            return sanitized
        else:
            return str(data)  # As a last resort, convert to string if no other method is available.


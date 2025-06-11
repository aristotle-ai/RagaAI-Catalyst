import contextvars
from typing import Optional, Dict
import json
from datetime import datetime
import uuid
import os
import builtins
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

from ..data.data_structure import (
    Trace,
    Metadata,
    SystemInfo,
    OSInfo,
    EnvironmentInfo,
    Resources,
    CPUResource,
    MemoryResource,
    DiskResource,
    NetworkResource,
    ResourceInfo,
    MemoryInfo,
    DiskInfo,
    NetworkInfo,
    Component,
    LLMComponent,
    AgentComponent,
    ToolComponent,
    NetworkCall,
    Interaction,
    Error,
)

from ....ragaai_catalyst import RagaAICatalyst


class AgenticTracing():
    def __init__(self, user_detail, auto_instrumentation=None, timeout=120):
        self.project_name = user_detail["project_name"]
        self.project_id = user_detail["project_id"]
        self.trace_user_detail = user_detail["trace_user_detail"]
        self.base_url = f"{RagaAICatalyst.BASE_URL}"
        self.timeout = timeout
        
        # Add warning flag
        self._warning_shown = False

        self.tools: Dict[str, Tool] = {}
        self.call_depth = contextvars.ContextVar("call_depth", default=0)
        self.current_component_id = contextvars.ContextVar(
            "current_component_id", default=None
        )

        # Handle auto_instrumentation
        if auto_instrumentation is None:
            # Default behavior: everything enabled
            self.is_active = True
            self.auto_instrument_llm = True
            self.auto_instrument_tool = True
            self.auto_instrument_agent = True
            self.auto_instrument_user_interaction = True
            self.auto_instrument_file_io = True
            self.auto_instrument_network = True
            self.auto_instrument_custom = True
        else:
            # Set global active state
            self.is_active = True

            # Set individual components
            if isinstance(auto_instrumentation, dict):
                self.auto_instrument_llm = auto_instrumentation.get("llm", True)
                self.auto_instrument_tool = auto_instrumentation.get("tool", True)
                self.auto_instrument_agent = auto_instrumentation.get("agent", True)
                self.auto_instrument_user_interaction = auto_instrumentation.get(
                    "user_interaction", True
                )
                self.auto_instrument_file_io = auto_instrumentation.get(
                    "file_io", True
                )
                self.auto_instrument_network = auto_instrumentation.get(
                    "network", True
                )
                self.auto_instrument_custom = auto_instrumentation.get("custom", True)
            else:
                # If boolean provided, apply to all components
                self.auto_instrument_llm = bool(auto_instrumentation)
                self.auto_instrument_tool = bool(auto_instrumentation)
                self.auto_instrument_agent = bool(auto_instrumentation)
                self.auto_instrument_user_interaction = bool(auto_instrumentation)
                self.auto_instrument_file_io = bool(auto_instrumentation)
                self.auto_instrument_network = bool(auto_instrumentation)
                self.auto_instrument_custom = bool(auto_instrumentation)

        self.current_agent_id = contextvars.ContextVar("current_agent_id", default=None)
        self.agent_children = contextvars.ContextVar("agent_children", default=[])
        self.component_network_calls = {}  # Store network calls per component
        self.component_user_interaction = {}


    def register_post_processor(self, post_processor_func):
        """
        Pass through the post-processor registration to the BaseTracer
        """
        if not callable(post_processor_func):
            logger.error("post_processor_func must be a callable")
        self.post_processor = post_processor_func

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit"""
        self.stop()

"""
Workflow integration for LlamaIndex tracing.

This module provides specialized tracing for LlamaIndex Workflow objects.
"""

import functools
import inspect
import contextvars
from typing import Any, Callable, Dict, Optional

from llama_index.core.workflow import Workflow
from llama_index.core.instrumentation import get_dispatcher
from ....llamaindex_instrumentation import init_llamaindex_instrumentation, stop_llamaindex_instrumentation

# Context variable to track the current workflow run
current_workflow_run = contextvars.ContextVar("current_workflow_run", default=None)

def trace_workflow(tracer=None):
    """
    Decorator to trace LlamaIndex Workflow objects.
    
    This decorator wraps a Workflow's run method to ensure proper
    instrumentation throughout the entire workflow execution.
    
    Args:
        tracer: Optional tracer instance. If None, will use the current tracer.
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            # Get the active tracer
            active_tracer = tracer
            if active_tracer is None:
                # Try to get the tracer from the object
                active_tracer = getattr(self, "tracer", None)
            
            if active_tracer is None:
                # If no tracer, just run the function
                return await func(self, *args, **kwargs)
            
            # Set up the workflow context
            token = current_workflow_run.set({
                "workflow_id": id(self),
                "workflow_name": self.__class__.__name__,
                "args": args,
                "kwargs": kwargs
            })
            
            # Initialize instrumentation
            handler_refs = init_llamaindex_instrumentation(active_tracer)
            
            try:
                # Run the workflow
                result = await func(self, *args, **kwargs)
                return result
            finally:
                # Clean up
                if handler_refs:
                    stop_llamaindex_instrumentation(active_tracer, handler_refs)
                current_workflow_run.reset(token)
        
        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            # Get the active tracer
            active_tracer = tracer
            if active_tracer is None:
                # Try to get the tracer from the object
                active_tracer = getattr(self, "tracer", None)
            
            if active_tracer is None:
                # If no tracer, just run the function
                return func(self, *args, **kwargs)
            
            # Set up the workflow context
            token = current_workflow_run.set({
                "workflow_id": id(self),
                "workflow_name": self.__class__.__name__,
                "args": args,
                "kwargs": kwargs
            })
            
            # Initialize instrumentation
            handler_refs = init_llamaindex_instrumentation(active_tracer)
            
            try:
                # Run the workflow
                result = func(self, *args, **kwargs)
                return result
            finally:
                # Clean up
                if handler_refs:
                    stop_llamaindex_instrumentation(active_tracer, handler_refs)
                current_workflow_run.reset(token)
        
        # Choose the right wrapper based on if the function is async or not
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def patch_workflow_class():
    """
    Monkey patch the Workflow class from LlamaIndex to automatically trace all runs.
    
    This function replaces the original run method with a traced version.
    """
    original_run = Workflow.run
    original_arun = Workflow.arun if hasattr(Workflow, "arun") else None
    
    # Apply our tracing decorator
    Workflow.run = trace_workflow()(original_run)
    
    if original_arun:
        Workflow.arun = trace_workflow()(original_arun)
    
    return {
        "original_run": original_run,
        "original_arun": original_arun
    }

def unpatch_workflow_class(originals):
    """
    Restore the original Workflow methods.
    
    Args:
        originals: Dictionary with original methods from patch_workflow_class
    """
    if originals:
        if "original_run" in originals:
            Workflow.run = originals["original_run"]
        if "original_arun" in originals and hasattr(Workflow, "arun"):
            Workflow.arun = originals["original_arun"]

"""
LLM integration for LlamaIndex tracing.

This module provides specialized tracing for LlamaIndex LLM models.
"""

import functools
from typing import Any, Callable, Dict, List, Optional, Union
import inspect
import asyncio

from llama_index.core.llms import ChatMessage, CompletionResponse, LLMMetadata
from llama_index.core.llms.base import LLM, CompletionResponse, ChatResponse
from ....llamaindex_instrumentation import init_llamaindex_instrumentation, stop_llamaindex_instrumentation

def trace_llm(tracer=None):
    """
    Decorator to trace LlamaIndex LLM operations.
    
    This decorator wraps LLM methods to ensure proper
    instrumentation throughout the entire LLM process.
    It handles both standard and streaming LLM responses.
    
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
            
            # Check if this is a streaming operation
            func_name = func.__name__
            is_streaming = "stream" in func_name
            
            # Initialize instrumentation
            handler_refs = init_llamaindex_instrumentation(active_tracer)
            
            try:
                # Run the LLM function
                result = await func(self, *args, **kwargs)
                
                # Special handling for streaming results
                if is_streaming:
                    # Return a modified generator that preserves tracing context
                    if asyncio.iscoroutine(result) or inspect.isasyncgen(result):
                        async def wrapped_async_gen():
                            async for token in result:
                                yield token
                            # Ensure cleanup happens after the generator is consumed
                            if handler_refs:
                                stop_llamaindex_instrumentation(active_tracer, handler_refs)
                        return wrapped_async_gen()
                    else:
                        # For synchronous generators
                        def wrapped_gen():
                            for token in result:
                                yield token
                            # Ensure cleanup happens after the generator is consumed
                            if handler_refs:
                                stop_llamaindex_instrumentation(active_tracer, handler_refs)
                        return wrapped_gen()
                return result
            finally:
                # Clean up only for non-streaming results
                # For streaming, cleanup is handled in the wrapped generator
                if not is_streaming and handler_refs:
                    stop_llamaindex_instrumentation(active_tracer, handler_refs)
        
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
            
            # Check if this is a streaming operation
            func_name = func.__name__
            is_streaming = "stream" in func_name
            
            # Initialize instrumentation
            handler_refs = init_llamaindex_instrumentation(active_tracer)
            
            try:
                # Run the LLM function
                result = func(self, *args, **kwargs)
                
                # Special handling for streaming results
                if is_streaming and inspect.isgenerator(result):
                    def wrapped_gen():
                        for token in result:
                            yield token
                        # Ensure cleanup happens after the generator is consumed
                        if handler_refs:
                            stop_llamaindex_instrumentation(active_tracer, handler_refs)
                    return wrapped_gen()
                return result
            finally:
                # Clean up only for non-streaming results
                # For streaming, cleanup is handled in the wrapped generator
                if not is_streaming and handler_refs:
                    stop_llamaindex_instrumentation(active_tracer, handler_refs)
        
        # Choose the right wrapper based on if the function is async or not
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def patch_llm_class():
    """
    Monkey patch the LLM class from LlamaIndex to automatically trace all LLM operations.
    
    This function replaces the original LLM methods with traced versions.
    """
    # Store original methods
    original_complete = LLM.complete
    original_chat = LLM.chat
    original_stream_complete = LLM.stream_complete if hasattr(LLM, "stream_complete") else None
    original_stream_chat = LLM.stream_chat if hasattr(LLM, "stream_chat") else None
    original_acomplete = LLM.acomplete if hasattr(LLM, "acomplete") else None
    original_achat = LLM.achat if hasattr(LLM, "achat") else None
    original_astream_complete = LLM.astream_complete if hasattr(LLM, "astream_complete") else None
    original_astream_chat = LLM.astream_chat if hasattr(LLM, "astream_chat") else None
    
    # Apply our tracing decorator
    LLM.complete = trace_llm()(original_complete)
    LLM.chat = trace_llm()(original_chat)
    
    if original_stream_complete:
        LLM.stream_complete = trace_llm()(original_stream_complete)
    
    if original_stream_chat:
        LLM.stream_chat = trace_llm()(original_stream_chat)
    
    if original_acomplete:
        LLM.acomplete = trace_llm()(original_acomplete)
    
    if original_achat:
        LLM.achat = trace_llm()(original_achat)
    
    if original_astream_complete:
        LLM.astream_complete = trace_llm()(original_astream_complete)
    
    if original_astream_chat:
        LLM.astream_chat = trace_llm()(original_astream_chat)
    
    return {
        "original_complete": original_complete,
        "original_chat": original_chat,
        "original_stream_complete": original_stream_complete,
        "original_stream_chat": original_stream_chat,
        "original_acomplete": original_acomplete,
        "original_achat": original_achat,
        "original_astream_complete": original_astream_complete,
        "original_astream_chat": original_astream_chat
    }

def unpatch_llm_class(originals):
    """
    Restore the original LLM methods.
    
    Args:
        originals: Dictionary with original methods from patch_llm_class
    """
    if originals:
        if "original_complete" in originals:
            LLM.complete = originals["original_complete"]
        if "original_chat" in originals:
            LLM.chat = originals["original_chat"]
        if "original_stream_complete" in originals and hasattr(LLM, "stream_complete"):
            LLM.stream_complete = originals["original_stream_complete"]
        if "original_stream_chat" in originals and hasattr(LLM, "stream_chat"):
            LLM.stream_chat = originals["original_stream_chat"]
        if "original_acomplete" in originals and hasattr(LLM, "acomplete"):
            LLM.acomplete = originals["original_acomplete"]
        if "original_achat" in originals and hasattr(LLM, "achat"):
            LLM.achat = originals["original_achat"]
        if "original_astream_complete" in originals and hasattr(LLM, "astream_complete"):
            LLM.astream_complete = originals["original_astream_complete"]
        if "original_astream_chat" in originals and hasattr(LLM, "astream_chat"):
            LLM.astream_chat = originals["original_astream_chat"]

"""
Embedding integration for LlamaIndex tracing.

This module provides specialized tracing for LlamaIndex embedding models.
"""

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

from llama_index.core.embeddings import BaseEmbedding
from ....llamaindex_instrumentation import init_llamaindex_instrumentation, stop_llamaindex_instrumentation

def trace_embedding(tracer=None):
    """
    Decorator to trace LlamaIndex embedding operations.
    
    This decorator wraps embedding methods to ensure proper
    instrumentation throughout the entire embedding process.
    
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
            
            # Initialize instrumentation
            handler_refs = init_llamaindex_instrumentation(active_tracer)
            
            try:
                # Run the embedding function
                result = await func(self, *args, **kwargs)
                return result
            finally:
                # Clean up
                if handler_refs:
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
            
            # Initialize instrumentation
            handler_refs = init_llamaindex_instrumentation(active_tracer)
            
            try:
                # Run the embedding function
                result = func(self, *args, **kwargs)
                return result
            finally:
                # Clean up
                if handler_refs:
                    stop_llamaindex_instrumentation(active_tracer, handler_refs)
        
        # Choose the right wrapper based on if the function is async or not
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def patch_embedding_class():
    """
    Monkey patch the BaseEmbedding class from LlamaIndex to automatically trace all embedding operations.
    
    This function replaces the original embedding methods with traced versions.
    """
    original_get_text_embedding = BaseEmbedding.get_text_embedding
    original_get_text_embeddings = BaseEmbedding.get_text_embeddings
    original_aget_text_embedding = BaseEmbedding._aget_text_embedding if hasattr(BaseEmbedding, "_aget_text_embedding") else None
    original_aget_text_embeddings = BaseEmbedding._aget_text_embeddings if hasattr(BaseEmbedding, "_aget_text_embeddings") else None
    
    # Apply our tracing decorator
    BaseEmbedding.get_text_embedding = trace_embedding()(original_get_text_embedding)
    BaseEmbedding.get_text_embeddings = trace_embedding()(original_get_text_embeddings)
    
    if original_aget_text_embedding:
        BaseEmbedding._aget_text_embedding = trace_embedding()(original_aget_text_embedding)
    
    if original_aget_text_embeddings:
        BaseEmbedding._aget_text_embeddings = trace_embedding()(original_aget_text_embeddings)
    
    return {
        "original_get_text_embedding": original_get_text_embedding,
        "original_get_text_embeddings": original_get_text_embeddings,
        "original_aget_text_embedding": original_aget_text_embedding,
        "original_aget_text_embeddings": original_aget_text_embeddings
    }

def unpatch_embedding_class(originals):
    """
    Restore the original BaseEmbedding methods.
    
    Args:
        originals: Dictionary with original methods from patch_embedding_class
    """
    if originals:
        if "original_get_text_embedding" in originals:
            BaseEmbedding.get_text_embedding = originals["original_get_text_embedding"]
        if "original_get_text_embeddings" in originals:
            BaseEmbedding.get_text_embeddings = originals["original_get_text_embeddings"]
        if "original_aget_text_embedding" in originals and hasattr(BaseEmbedding, "_aget_text_embedding"):
            BaseEmbedding._aget_text_embedding = originals["original_aget_text_embedding"]
        if "original_aget_text_embeddings" in originals and hasattr(BaseEmbedding, "_aget_text_embeddings"):
            BaseEmbedding._aget_text_embeddings = originals["original_aget_text_embeddings"]

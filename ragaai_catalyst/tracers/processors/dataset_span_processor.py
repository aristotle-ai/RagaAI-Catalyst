"""
Dataset Span Processor - Automatically adds dataset attributes to spans.

This processor automatically sets the 'ragaai.dataset' attribute on every span
based on the dataset_name from the tracer initialization.
"""

import logging
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry import context

logger = logging.getLogger("RagaAICatalyst")

class DatasetSpanProcessor(SpanProcessor):
    """
    A SpanProcessor that automatically adds dataset routing attributes to spans.
    
    This ensures that every span gets the 'ragaai.dataset' attribute set to the 
    dataset_name that was provided when the tracer was initialized.
    """
    
    def __init__(self, dataset_name):
        """
        Initialize the DatasetSpanProcessor.
        
        Args:
            dataset_name (str): The dataset name to set on all spans
        """
        self.dataset_name = dataset_name
        logger.debug(f"DatasetSpanProcessor initialized with dataset: {dataset_name}")
    
    def on_start(self, span, parent_context=None):
        """
        Called when a span starts. Automatically set the dataset attribute.
        
        Args:
            span: The span that is starting
            parent_context: The parent context (optional)
        """
        try:
            if span and span.is_recording():
                # Set the dataset attribute on the span
                span.set_attribute("ragaai.dataset", self.dataset_name)
                
                # Optionally add a marker that this was auto-set
                span.set_attribute("ragaai.auto_dataset", True)
                
                logger.debug(f"Set dataset attribute '{self.dataset_name}' on span: {span.name}")
            
        except Exception as e:
            logger.warning(f"Error setting dataset attribute on span: {e}")
    
    def on_end(self, span):
        """
        Called when a span ends. No action needed for dataset routing.
        
        Args:
            span: The span that is ending
        """
        pass
    
    def shutdown(self):
        """Shutdown the processor."""
        pass
    
    def force_flush(self, timeout_millis=None):
        """Force flush the processor."""
        pass
    
    def update_dataset_name(self, new_dataset_name):
        """
        Update the dataset name for future spans.
        
        Args:
            new_dataset_name (str): New dataset name to use
        """
        old_dataset = self.dataset_name
        self.dataset_name = new_dataset_name
        logger.info(f"Updated dataset from '{old_dataset}' to '{new_dataset_name}'") 
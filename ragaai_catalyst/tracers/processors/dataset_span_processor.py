"""
Dataset Span Processor - Automatically adds dataset attributes to spans.

This processor automatically sets the 'ragaai.dataset' attribute on every span
based on the dataset_name from the tracer initialization or context variables for per-request isolation.
"""

import logging
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry import context

logger = logging.getLogger("RagaAICatalyst")

class DatasetSpanProcessor(SpanProcessor):
    """
    A SpanProcessor that automatically adds dataset routing attributes to spans.
    
    This ensures that every span gets the 'ragaai.dataset' attribute set to the 
    dataset_name that was provided when the tracer was initialized, or from the
    current request context for per-request isolation.
    """
    
    def __init__(self, dataset_name):
        """
        Initialize the DatasetSpanProcessor.
        
        Args:
            dataset_name (str): The default dataset name to set on spans (fallback)
        """
        self.default_dataset_name = dataset_name
        logger.debug(f"DatasetSpanProcessor initialized with default dataset: {dataset_name}")
    
    def on_start(self, span, parent_context=None):
        """
        Called when a span starts. Automatically set the dataset attribute.
        
        Args:
            span: The span that is starting
            parent_context: The parent context (optional)
        """
        try:
            if span and span.is_recording():
                # First try to get dataset name from current context (per-request)
                dataset_name = context.get_value("ragaai.dataset_name")
                
                if dataset_name is None:
                    # Fallback to default dataset name if context is not set
                    dataset_name = self.default_dataset_name
                    logger.debug(f"Using default dataset '{dataset_name}' for span: {span.name}")
                else:
                    logger.debug(f"Using context dataset '{dataset_name}' for span: {span.name}")
                
                # Set the dataset attribute on the span
                span.set_attribute("ragaai.dataset", dataset_name)
                
                # Add context source indicator for debugging
                context_source = "context" if context.get_value("ragaai.dataset_name") else "default"
                span.set_attribute("ragaai.dataset_source", context_source)
                
                logger.debug(f"Set dataset attribute '{dataset_name}' on span: {span.name} (source: {context_source})")
            
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
        Update the default dataset name for future spans.
        Note: This method is kept for backward compatibility but context variables take precedence.
        
        Args:
            new_dataset_name (str): New default dataset name to use
        """
        old_dataset = self.default_dataset_name
        self.default_dataset_name = new_dataset_name
        logger.info(f"Updated default dataset from '{old_dataset}' to '{new_dataset_name}'")
        logger.info("Note: Context variables will take precedence over this default value") 
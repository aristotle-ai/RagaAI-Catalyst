from .experiment import Experiment
from .ragaai_catalyst import RagaAICatalyst
from .utils import response_checker
from .dataset import Dataset
from .prompt_manager import PromptManager
from .evaluation import Evaluation
from .synthetic_data_generation import SyntheticDataGeneration
from .redteaming import RedTeaming
from .guardrails_manager import GuardrailsManager
from .guard_executor import GuardExecutor
from .tracers import Tracer, init_tracing, trace_agent, trace_llm, trace_tool, current_span, trace_custom
from .tracers.distributed import finalize_streaming_trace, is_streaming_active
from .redteaming import RedTeaming

# Expose key components for tracing operations
from ragaai_catalyst.tracers.tracer import Tracer
from ragaai_catalyst.tracers.agentic_tracing.tracers.decorators import trace_agent, trace_tool, trace_llm, trace_custom, trace_file_ops, trace_network
from ragaai_catalyst.tracers.agentic_tracing.tracers.span_manager import current_span
from ragaai_catalyst.client import RagaAICatalyst

# Initialize tracing system
def init_tracing(catalyst=None, tracer=None):
    """
    Initialize the tracing system with the given catalyst client and tracer.
    
    Args:
        catalyst: RagaAICatalyst client instance
        tracer: Tracer instance
    """
    if catalyst is not None:
        import os
        os.environ["RAGAAI_CATALYST_TOKEN"] = catalyst.token
        os.environ["RAGAAI_CATALYST_BASE_URL"] = catalyst.base_url
        
    return tracer


__all__ = [
    "Experiment", 
    "RagaAICatalyst", 
    "Tracer", 
    "PromptManager", 
    "Evaluation",
    "SyntheticDataGeneration",
    "RedTeaming",
    "GuardrailsManager", 
    "GuardExecutor",
    "init_tracing", 
    "trace_agent", 
    "trace_llm",
    "trace_tool",
    "current_span",
    "trace_custom",
    "finalize_streaming_trace",
    "is_streaming_active",
    "trace_file_ops",
    "trace_network"
]

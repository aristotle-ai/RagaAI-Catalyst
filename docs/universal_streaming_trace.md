# Universal Streaming Trace Support

RagaAI Catalyst provides comprehensive support for capturing traces during streaming operations across all tracer types, not just LlamaIndex. This feature is particularly valuable when working with any asynchronous streaming content, such as LLM responses, chat completions, or custom streaming data sources.

## Overview

When working with streaming operations, traditional tracing approaches may struggle to capture the complete lifecycle of events due to the asynchronous nature of streaming. RagaAI Catalyst solves this problem with a unified approach:

1. Automatic streaming detection in all tracer types
2. Consistent trace context maintenance throughout the streaming lifecycle
3. Universal API for finalizing traces after streaming completes
4. Support for concurrent streaming operations with thread-safe handling

This ensures that all events are captured in a single, comprehensive trace without requiring modification of the underlying streaming implementation.

## Core Features

### 1. Universal Support

The streaming trace functionality is now implemented at the base level, making it available to all tracer types:

- Basic Tracer
- LlamaIndex Tracer
- LangChain Tracer
- Agent Tracer
- Custom Tracers

### 2. Thread Safety

All streaming operations are thread-safe, allowing for concurrent streaming traces without data corruption:

- Lock-based access control for shared resources
- Atomic operations for state changes
- Proper isolation between different streaming contexts

### 3. Simple API

Two primary functions make working with streaming traces straightforward:

- `finalize_streaming_trace()`: Complete a streaming trace when all events are finished
- `is_streaming_active()`: Check if there's currently an active streaming operation

## Usage

### Basic Usage Pattern

```python
from ragaai_catalyst import (
    RagaAICatalyst, 
    Tracer, 
    init_tracing, 
    finalize_streaming_trace,
    is_streaming_active
)

# Initialize RagaAI Catalyst
catalyst = RagaAICatalyst(
    token="YOUR_TOKEN",
    base_url="https://api.raga.ai"
)

# Create a tracer (works with any tracer type)
tracer = Tracer(
    project_name="your-project",
    dataset_name="your-dataset"
)

# Initialize tracing
init_tracing(catalyst=catalyst, tracer=tracer)

# Start tracing
tracer.start()

try:
    # Run your streaming operation
    streaming_result = await your_streaming_function("Your input here")
    
    # Check if streaming is active
    if is_streaming_active():
        # Finalize the trace after streaming completes
        finalize_streaming_trace()
    else:
        # Normal stop if no streaming was detected
        tracer.stop()
    
except Exception as e:
    print(f"Error during streaming: {e}")
    tracer.stop()
```

### With Decorator-Based Tracing

The streaming functionality also works seamlessly with decorator-based tracing:

```python
from ragaai_catalyst import (
    trace_agent, 
    trace_tool, 
    finalize_streaming_trace,
    is_streaming_active
)

# Traced streaming function
@trace_agent(name="streaming_agent")
async def stream_content(query: str):
    result = ""
    async for chunk in llm.stream(query):
        print(chunk, end="", flush=True)
        result += chunk
    return result

# Main function
async def main():
    # Function with streaming
    await stream_content("Your query here")
    
    # Finalize after streaming is complete
    if is_streaming_active():
        finalize_streaming_trace()
```

### FastAPI Integration

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

from ragaai_catalyst import (
    RagaAICatalyst, 
    Tracer, 
    init_tracing, 
    finalize_streaming_trace
)

app = FastAPI()

# Initialize tracing
catalyst = RagaAICatalyst(token="YOUR_TOKEN")
tracer = Tracer(project_name="fastapi-streaming", dataset_name="streaming-test")
init_tracing(catalyst=catalyst, tracer=tracer)

@app.post("/run/")
async def run_agent(payload: dict):
    """Endpoint that streams responses."""
    input_query = payload.get("input")
    
    # Start tracing for the request
    tracer.start()
    
    # Return a streaming response
    return StreamingResponse(
        event_generator(input_query), 
        media_type="text/event-stream"
    )

async def event_generator(query):
    """Generate streaming events and finalize trace."""
    try:
        # Stream the response
        async for chunk in your_streaming_function(query):
            yield f"data: {chunk}\n\n"
            
        # Finalize the trace after streaming completes
        finalize_streaming_trace()
        
    except Exception as e:
        print(f"Error in event_generator: {e}")
        yield f"data: error: {str(e)}\n\n"
        tracer.stop()
```

## Technical Details

### How Streaming Is Detected

The system detects streaming operations through:

1. Event pattern analysis: Identifying characteristic patterns of events that indicate streaming
2. Temporal spacing: Analyzing the timing between related events
3. Explicit flags: Through integration with streaming frameworks like LlamaIndex

When streaming is detected, the trace is automatically marked with `is_streaming=True` and `pending_finalization=True`, which delays trace completion until explicitly finalized.

### Trace Lifecycle During Streaming

1. **Initialization**: Tracer is created and started normally
2. **Streaming Detection**: System detects streaming events are occurring
3. **Context Maintenance**: Trace context remains active during streaming
4. **Event Collection**: All events are continuously captured
5. **Finalization**: When `finalize_streaming_trace()` is called, events are processed
6. **Upload**: Complete trace is processed and uploaded

## Best Practices

1. **Always Check Streaming Status**: Use `is_streaming_active()` before finalizing
2. **Use Try-Finally Blocks**: Ensure proper cleanup even if errors occur
3. **Explicit Finalization**: Always call `finalize_streaming_trace()` after streaming
4. **Context Preservation**: Maintain the tracer context throughout streaming

## Troubleshooting

### Missing Events

If events are missing from your trace:

1. Verify `finalize_streaming_trace()` is called after all streaming completes
2. Check that the tracing context is maintained throughout streaming
3. Ensure you're not creating multiple trace contexts

### Trace Finalization Issues

If traces aren't being properly finalized:

1. Confirm streaming actually completes before finalization
2. Check for exceptions during streaming that might prevent finalization
3. Verify you're using the same tracer instance consistently

### Performance Considerations

For optimal performance with streaming traces:

1. Avoid creating too many small spans within a streaming operation
2. Consider batching events when dealing with high-frequency streaming
3. Be mindful of trace size when capturing large streaming operations

## Migration from LlamaIndex-Only Streaming

If you were previously using the LlamaIndex-specific streaming functionality:

```python
# Old approach (LlamaIndex only)
from ragaai_catalyst import RagaAICatalyst, Tracer

tracer = Tracer(
    project_name="your-project",
    dataset_name="your-dataset",
    tracer_type="llamaindex"
)

# New universal approach
from ragaai_catalyst import (
    RagaAICatalyst, 
    Tracer, 
    finalize_streaming_trace,
    is_streaming_active
)

# Works with any tracer type
tracer = Tracer(
    project_name="your-project",
    dataset_name="your-dataset"
)

# Finalize with the universal function
finalize_streaming_trace()
```

For more detailed information on general trace management, refer to the [Trace Management documentation](trace_management.md).

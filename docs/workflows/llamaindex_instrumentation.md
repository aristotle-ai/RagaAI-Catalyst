# LlamaIndex Instrumentation

RagaAI Catalyst provides comprehensive instrumentation for LlamaIndex applications, ensuring proper event handling, trace context management, and integration with existing tracing frameworks. This guide shows how to use the LlamaIndex instrumentation features.

## Automatic Instrumentation

The simplest way to instrument your LlamaIndex application is to use a Tracer context manager. The instrumentation is automatically handled when you use the context manager:

```python
from ragaai_catalyst import RagaAICatalyst, Tracer, init_tracing

# Initialize RagaAI Catalyst
catalyst = RagaAICatalyst(
    access_key="your_access_key",
    secret_key="your_secret_key",
    base_url="https://api.example.com/"
)
tracer = Tracer(
    project_name="your_project",
    dataset_name="your_dataset",
    tracer_type="Agentic"
)
init_tracing(catalyst=catalyst, tracer=tracer)

# Use the tracer as a context manager
with tracer:
    # Your LlamaIndex code here
    # All LlamaIndex events will be automatically captured
    response = query_engine.query("What is the capital of France?")
```

## Workflow Tracing

LlamaIndex's `Workflow` class is automatically instrumented to ensure proper tracing throughout the entire workflow execution:

```python
from llama_index.core.workflow import Workflow, step

class MyWorkflow(Workflow):
    @step
    async def first_step(self, ctx, ev):
        # Logic for the first step
        return next_event
        
    @step
    async def second_step(self, ctx, ev):
        # Logic for the second step
        return final_event

# Initialize the workflow
workflow = MyWorkflow()

# All steps in the workflow will be automatically traced
with tracer:
    result = workflow.run(input="user query")
```

## Custom Component Tracing

You can also explicitly trace specific components using dedicated decorators:

```python
from ragaai_catalyst import trace_agent, trace_tool, trace_custom, current_span

# Trace agent functions
@trace_agent(name="my_agent")
def execute_agent(input_text):
    # Agent logic here
    return result

# Trace tool calls
@trace_tool(name="my_tool")
def execute_tool(tool_input):
    # Tool logic here
    return tool_output

# Trace custom components
@trace_custom(name="data_processor")
def process_data(data):
    # Add custom metrics
    span = current_span()
    span.add_metrics(
        name="accuracy", 
        score=0.95, 
        reasoning="High confidence in processed data"
    )
    return processed_data
```

## Advanced Integration

For more advanced use cases, you can directly work with the LlamaIndex instrumentation handlers:

```python
from ragaai_catalyst import init_llamaindex_instrumentation, stop_llamaindex_instrumentation

# Manually initialize instrumentation
handler_refs = init_llamaindex_instrumentation(tracer)

try:
    # Your code here
    pass
finally:
    # Clean up
    stop_llamaindex_instrumentation(tracer, handler_refs)
```

## Supported Event Types

The LlamaIndex instrumentation captures and traces the following event types:

- LLM events (chat, completion, prediction)
- Agent events (chat steps, tool calls)
- Tool events (execution, results)
- Retrieval events
- Query events
- Embedding events
- Reranking events
- Synthesis events
- Workflow events

## Context Propagation

The instrumentation system maintains trace context across asynchronous boundaries, ensuring that spans are properly related to each other even in complex async workflows:

```python
async def process_async():
    # Context is maintained across async boundaries
    async with tracer:
        # All async operations will maintain proper trace context
        results = await asyncio.gather(
            async_operation_1(),
            async_operation_2()
        )
    return results
```

By using RagaAI Catalyst's LlamaIndex instrumentation, you can gain deep insights into your LlamaIndex application's performance, behavior, and trace data without having to manually instrument every component.

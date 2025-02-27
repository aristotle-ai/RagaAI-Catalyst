"""
Example demonstrating how to use RagaAI Catalyst to trace streaming operations across all tracer types.

This example shows how to use the enhanced tracing system with its support for streaming
operations in any context, not just LlamaIndex.
"""

import os
import asyncio
import time
from typing import AsyncIterable, Dict, List, Any

from ragaai_catalyst import (
    RagaAICatalyst, 
    Tracer, 
    init_tracing, 
    finalize_streaming_trace,
    is_streaming_active,
    trace_llm,
    trace_tool,
    trace_agent
)

# Set your API key if needed
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Initialize RagaAI Catalyst
catalyst = RagaAICatalyst(
    token="YOUR_RAGAAI_CATALYST_TOKEN",
    base_url="https://api.raga.ai"
)

# Create a tracer for general purposes
tracer = Tracer(
    project_name="universal-streaming-example",
    dataset_name="streaming-test",
    metadata={"purpose": "universal streaming demonstration"},
)

# Initialize the tracer
init_tracing(catalyst=catalyst, tracer=tracer)

# Simulating a streaming response
async def simulate_streaming_response(query: str) -> AsyncIterable[str]:
    """Simulate an async streaming response from an LLM or tool."""
    chunks = [
        "I'm generating ",
        "a response ",
        "for your query: ",
        f"'{query}'. ",
        "This simulates ",
        "how streaming ",
        "data works."
    ]
    for chunk in chunks:
        await asyncio.sleep(0.3)  # Simulate processing time
        yield chunk

# Decorated streaming function
@trace_agent(name="streaming_agent")
async def stream_content(query: str):
    """Function that streams results and is traced."""
    print(f"Starting streaming response for: {query}")
    result = ""
    
    async for chunk in simulate_streaming_response(query):
        print(f"Chunk: {chunk}", end="", flush=True)
        result += chunk
        
    print("\nStreaming complete!")
    return result

# Example showing how to use tracing with both simple and streaming functions
@trace_agent(name="main_agent")
async def run_example():
    """Run the example to demonstrate universal streaming tracing."""
    # Simple (non-streaming) traced function
    @trace_tool(name="simple_calculator")
    def calculate(a: int, b: int) -> int:
        """Simple calculation tool."""
        return a + b
    
    print("Running simple calculation...")
    result = calculate(5, 7)
    print(f"Result: {result}")
    
    # Now run a streaming operation
    print("\nStarting streaming operation...")
    streaming_result = await stream_content("Tell me about streaming tracing")
    
    print(f"\nFinal result: {streaming_result}")
    return {"simple_result": result, "streaming_result": streaming_result}

async def main():
    """Main function to run the example."""
    # Start tracing for the entire process
    tracer.start()
    
    try:
        # Run the example that includes both simple and streaming operations
        await run_example()
        
        # Check if we have an active streaming operation
        if is_streaming_active():
            print("\nDetected active streaming operation, finalizing trace...")
            finalize_result = finalize_streaming_trace()
            print(f"Trace finalization result: {finalize_result}")
        else:
            print("\nNo active streaming detected, stopping trace normally...")
            tracer.stop()
        
    except Exception as e:
        print(f"Error during execution: {e}")
        tracer.stop()

if __name__ == "__main__":
    print("Universal Streaming Trace Example")
    print("=================================")
    asyncio.run(main())

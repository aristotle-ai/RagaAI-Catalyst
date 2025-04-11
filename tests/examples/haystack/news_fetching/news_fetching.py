import os
from dotenv import load_dotenv
from typing import Any, Dict, List
from haystack.dataclasses import ChatMessage
from haystack.components.tools import ToolInvoker
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.routers import ConditionalRouter
from haystack.tools import ComponentTool
from haystack.components.websearch import SerperDevWebSearch
from haystack import Pipeline, component
from haystack.core.component.types import Variadic
import argparse

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))


from ragaai_catalyst import RagaAICatalyst, Tracer, init_tracing

# Load environment variables from .env file
load_dotenv()

# Setup Raga AI Catalyst for enhanced monitoring and tracing
catalyst = RagaAICatalyst(
    access_key=os.getenv('RAGAAI_CATALYST_ACCESS_KEY'),
    secret_key=os.getenv('RAGAAI_CATALYST_SECRET_KEY'),
    base_url=os.getenv('RAGAAI_CATALYST_BASE_URL')
)

tracer = Tracer(
    project_name='prompt_metric_dataset',#os.getenv("RAGAAI_PROJECT_NAME"),
    dataset_name='pytest_dataset',#os.getenv("RAGAAI_DATASET_NAME"),
    tracer_type="agentic/haystack",
)

# Initialize tracing to track system performance and activities
init_tracing(catalyst=catalyst, tracer=tracer)

# Component to collect and store messages temporarily
@component()
class MessageCollector:
    def __init__(self):
        self._messages = []

    @component.output_types(messages=List[ChatMessage])
    def run(self, messages: Variadic[List[ChatMessage]]) -> Dict[str, Any]:
        self._messages.extend([msg for inner in messages for msg in inner])
        return {"messages": self._messages}

    def clear(self):
        self._messages = []

# Component tool for web search, using SerperDev
web_tool = ComponentTool(
    component=SerperDevWebSearch(top_k=3)
)

# Routing conditions to handle replies with or without tool calls
routes = [
    {
        "condition": "{{replies[0].tool_calls | length > 0}}",
        "output": "{{replies}}",
        "output_name": "there_are_tool_calls",
        "output_type": List[ChatMessage],
    },
    {
        "condition": "{{replies[0].tool_calls | length == 0}}",
        "output": "{{replies}}",
        "output_name": "final_replies",
        "output_type": List[ChatMessage],
    },
]

# Setup the pipeline for processing user queries
tool_agent = Pipeline()
tool_agent.add_component("message_collector", MessageCollector())
tool_agent.add_component("generator", OpenAIChatGenerator(model="gpt-4o-mini", tools=[web_tool]))
tool_agent.add_component("router", ConditionalRouter(routes, unsafe=True))
tool_agent.add_component("tool_invoker", ToolInvoker(tools=[web_tool]))

# Define connections in the pipeline
tool_agent.connect("generator.replies", "router")
tool_agent.connect("router.there_are_tool_calls", "tool_invoker")
tool_agent.connect("router.there_are_tool_calls", "message_collector")
tool_agent.connect("tool_invoker.tool_messages", "message_collector")
tool_agent.connect("message_collector", "generator.messages")

# Example messages to simulate user interaction
messages = [
    ChatMessage.from_system("Hello! Ask me anything about current news or information."),
    ChatMessage.from_user("What is the latest news on the Mars Rover mission?")
]


def main(info: str):
    print(f"Info: {info}")
    # Run the pipeline with the provided example messages
    result = tool_agent.run({"messages": messages})

    # Print the final reply from the agent
    print(result["router"]["final_replies"][0].text)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test the news_fetching.py script.")
    parser.add_argument("--info", type=str, default="testing-news-fetching", help="The info to use (e.g., testing-news-fetching)")
    args = parser.parse_args()

    main(args.info)

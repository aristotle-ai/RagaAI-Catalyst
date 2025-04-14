import asyncio
import json
import os
 
from fastapi.responses import StreamingResponse
# os.environ["DEBUG"] = "1"
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import BaseTool, ToolOutput
from llama_index.core.workflow import Event, Workflow
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    step
)
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.agent.react.types import BaseReasoningStep, ActionReasoningStep
from llama_index.core.agent.react.output_parser import ReActOutputParser
from llama_index.core.tools import ToolSelection
import uvicorn
from llama_index.llms.azure_openai import AzureOpenAI
from dotenv import load_dotenv
from ragaai_catalyst import RagaAICatalyst
from ragaai_catalyst import Tracer
from pathlib import Path
import re

load_dotenv()

catalyst = RagaAICatalyst(
    access_key=os.getenv('CATALYST_ACCESS_KEY'), 
    secret_key=os.getenv('CATALYST_SECRET_KEY'), 
    base_url=os.getenv('CATALYST_BASE_URL')
)
tracer = Tracer(
    project_name=os.getenv('PROJECT_NAME'),
    dataset_name=os.getenv('DATASET_NAME'),
    tracer_type="agentic/llamaindex",
)

def masking_function(value):
    """
    Returns how to Mask strings values
    """
    value = re.sub(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', '< REDACTED EMAIL ADDRESS>', value)
    return value

tracer.register_masking_function(masking_function)
endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
deployment = os.environ["AZURE_DEPLOYMENT"]
subscription_key = os.environ["AZURE_SUBSCRIPTION_KEY"]
model = "gpt-4o-mini"

FI_LLM = AzureOpenAI(
    azure_endpoint=endpoint,
    model = model,
    api_key=subscription_key,
    api_version="2024-05-01-preview",
    engine=deployment
)
import random
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
import uvicorn
import json
import asyncio
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolSelection, ToolOutput
from llama_index.core.workflow import Event
from typing import Any, List
from llama_index.core.agent.react import ReActChatFormatter, ReActOutputParser
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    ObservationReasoningStep,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools.types import BaseTool
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool

app = FastAPI(title="ReAct Agent API")

# Event classes
class PrepEvent(Event):
    pass

class InputEvent(Event):
    input: list[ChatMessage]

class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]

class FunctionOutputEvent(Event):
    output: ToolOutput

class ProgressEvent(Event):
    msg: str

# ReAct Agent Implementation
class ReActAgent(Workflow):
    def __init__(
        self,
        *args: Any,
        llm: LLM | None = None,
        tools: list[BaseTool] | None = None,
        extra_context: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []
        self.llm = llm or OpenAI()
        self.memory = ChatMemoryBuffer.from_defaults(llm=llm)
        self.formatter = ReActChatFormatter.from_defaults(
            context=extra_context or ""
        )
        self.output_parser = ReActOutputParser()
        self.sources = []

    @step
    async def new_user_msg(self, ctx: Context, ev: StartEvent) -> PrepEvent:
        # clear sources
        self.sources = []
        # get user input
        user_input = ev.input
        user_msg = ChatMessage(role="user", content=user_input)
        self.memory.put(user_msg)
        # clear current reasoning
        await ctx.set("current_reasoning", [])
        return PrepEvent()

    @step
    async def prepare_chat_history(
        self, ctx: Context, ev: PrepEvent
    ) -> InputEvent:
        # get chat history
        chat_history = self.memory.get()
        current_reasoning = await ctx.get("current_reasoning", default=[])
        llm_input = self.formatter.format(
            self.tools, chat_history, current_reasoning=current_reasoning
        )
        return InputEvent(input=llm_input)

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | StopEvent:
        chat_history = ev.input
        response = await self.llm.achat(chat_history)
        try:
            reasoning_step = self.output_parser.parse(response.message.content)
            (await ctx.get("current_reasoning", default=[])).append(
                reasoning_step
            )
            if reasoning_step.is_done:
                self.memory.put(
                    ChatMessage(
                        role="assistant", content=reasoning_step.response
                    )
                )
                return StopEvent(
                    result={
                        "response": reasoning_step.response,
                        "sources": [*self.sources],
                        "reasoning": await ctx.get(
                            "current_reasoning", default=[]
                        ),
                    }
                )
            elif isinstance(reasoning_step, ActionReasoningStep):
                tool_name = reasoning_step.action
                tool_args = reasoning_step.action_input
                ctx.write_event_to_stream(
                    ProgressEvent(
                        msg=reasoning_step.thought
                    )
                )
                return ToolCallEvent(
                    tool_calls=[
                        ToolSelection(
                            tool_id="fake",
                            tool_name=tool_name,
                            tool_kwargs=tool_args,
                        )
                    ]
                )
        except Exception as e:
            (await ctx.get("current_reasoning", default=[])).append(
                ObservationReasoningStep(
                    observation=f"There was an error in parsing my reasoning: {e}"
                )
            )
        # if no tool calls or final response, iterate again
        return PrepEvent()

    @step
    async def handle_tool_calls(
        self, ctx: Context, ev: ToolCallEvent
    ) -> PrepEvent:
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}
        # call tools -- safely!
        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            if not tool:
                (await ctx.get("current_reasoning", default=[])).append(
                    ObservationReasoningStep(
                        observation=f"Tool {tool_call.tool_name} does not exist"
                    )
                )
                continue
            try:
                tool_output = tool(**tool_call.tool_kwargs)
                self.sources.append(tool_output)
                (await ctx.get("current_reasoning", default=[])).append(
                    ObservationReasoningStep(observation=tool_output.content)
                )
            except Exception as e:
                (await ctx.get("current_reasoning", default=[])).append(
                    ObservationReasoningStep(
                        observation=f"Error calling tool {tool.metadata.get_name()}: {e}"
                    )
                )
        # prep the next iteration
        return PrepEvent()
from litellm import completion
# Email generation tools
def generate_email_from_username(username: str, domain: str = "example.com") -> str:
    """
    Generates professional email suggestions based on a username.
    Provides multiple format variations using the given domain.
    
    Args:
        username: The base username to generate emails from
        domain: The domain to use for the email (default: example.com)
    
    Returns:
        A string containing multiple email format suggestions
    """
    prompt = f"""Generate 4 professional email address suggestions for the username "{username}" using the domain "{domain}".
    Follow these rules:
    1. Use common professional email formats
    2. Include at least one format with first initial + last name
    3. Make suggestions realistic and business-appropriate
    4. Present each suggestion on a new line with a brief explanation
    5. Do not include any personal information
    
    Format your response as:
    - email1@domain.com (explanation)
    - email2@domain.com (explanation)
    """

    try:
        response = completion(
            model=model,  # or your preferred model
            messages=[{
                "role": "system",
                "content": "You are a helpful assistant that generates professional email suggestions."
            },
            {
                "role": "user",
                "content": prompt
            }],
            temperature=0.7,
            max_tokens=200
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        # Fallback to basic email generation if LLM call fails
        formats = [
            f"{username}@{domain}",
            f"{username[0]}.{username[1:]}@{domain}",
            f"{username[0]}{username[1:]}@{domain}",
            f"{username}.{random.randint(100,999)}@{domain}"
        ]
        return "Suggested email formats (fallback mode):\n" + "\n".join(f"- {email}" for email in formats)


def generate_similar_emails(email: str) -> str:
    """
    Generates similar email variations using LLM based on an existing email address.
    
    Args:
        email: The original email address to base variations on
    
    Returns:
        A string containing similar but unique email suggestions
    """
    if "@" not in email:
        return "Invalid email format - must contain @ symbol"
    
    local_part, domain = email.split("@", 1)
    
    prompt = f"""Generate 4 professional variations of the email address "{email}".
    Follow these rules:
    1. Keep the domain "{domain}" unchanged
    2. Create variations of the local part "{local_part}"
    3. Use common professional variations like:
       - Adding numbers
       - Using different separators (. or _)
       - Abbreviating parts
       - Rearranging components
    4. Each suggestion should be realistic and business-appropriate
    5. Include a brief explanation for each variation
    
    Format your response as:
    - variation1@{domain} (explanation)
    - variation2@{domain} (explanation)
    """

    try:
        response = completion(
            model=model,
            messages=[{
                "role": "system",
                "content": "You are a helpful assistant that generates professional email address variations while maintaining business appropriateness."
            },
            {
                "role": "user",
                "content": prompt
            }],
            temperature=0.7,
            max_tokens=200
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        # Fallback to basic email variation if LLM call fails
        variations = [
            f"{local_part}{random.randint(10,99)}@{domain}",
            f"{local_part}.alt@{domain}",
            f"{local_part.replace('.', '_')}@{domain}",
            f"{local_part[0]}{local_part[1:].replace('.', '')}@{domain}"
        ]
        return "Similar email variations (fallback mode):\n" + "\n".join(f"- {email}" for email in variations)

# Create tools
tools = [
    FunctionTool.from_defaults(
        generate_email_from_username,
        name="generate_email_from_username",
        description="Generates professional email address suggestions from a username"
    ),
    FunctionTool.from_defaults(
        generate_similar_emails,
        name="generate_similar_emails",
        description="Creates similar but unique email variations based on an existing email address"
    )
]

# Initialize agent
agent = ReActAgent(
    llm=OpenAI(),  # Replace with your actual LLM if needed
    tools=tools,
    timeout=120,
    verbose=True
)

@app.post("/run/")
async def run_agent(payload: dict, background_tasks: BackgroundTasks):
    """Endpoint to run the ReAct agent with user input."""
    input = payload.get("input")  # Extract input from the payload
    handler = agent.run(input=input)
    return StreamingResponse(event_generator(handler), media_type="text/event-stream")

async def event_generator(handler):
    """Stream workflow events"""
    try:
        async for event in handler.stream_events():
            if isinstance(event, ProgressEvent):
                yield f"data: {json.dumps({'type': 'thought', 'msg': event.msg})}\n\n"
                    
        result = await handler
        yield f"data: {json.dumps({'type': 'answer', 'result': {'answer':result['response']}})}\n\n"
    except asyncio.CancelledError:
        print("Streaming cancelled by the client.")
    except Exception as e:
        print(f"Error in event_generator: {e}")
        yield f"data: {json.dumps({'type': 'error', 'msg': str(e)})}\n\n"

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8081)

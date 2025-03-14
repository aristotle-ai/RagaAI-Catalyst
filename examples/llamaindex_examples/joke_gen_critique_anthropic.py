# !pip install llama-index-llms-anthropic
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from llama_index.llms.anthropic import Anthropic
from dotenv import load_dotenv
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

load_dotenv()
from ragaai_catalyst.tracers import Tracer
from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst import trace_llm

catalyst = RagaAICatalyst(
    access_key=os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
    secret_key=os.getenv("RAGAAI_CATALYST_SECRET_KEY"),
    base_url=os.getenv("RAGAAI_CATALYST_BASE_URL"),
)

# Initialize tracer
tracer = Tracer(
    project_name="Llama-index_testing",
    dataset_name="anthropic",
    tracer_type="Agentic",
)

init_tracing(catalyst=catalyst, tracer=tracer)

class JokeEvent(Event):
    joke: str


class JokeFlow(Workflow):
    llm = Anthropic()

    @step
    #@trace_llm("generate joke")
    async def generate_joke(self, ev: StartEvent) -> JokeEvent:
        topic = ev.topic
        prompt = f"Write your best joke about {topic}."
        response = await self.llm.acomplete(prompt)
        return JokeEvent(joke=str(response))

    @step
    #@trace_llm("criticise joke")
    async def critique_joke(self, ev: JokeEvent) -> StopEvent:
        joke = ev.joke
        prompt = f"Give a thorough analysis and critique of the following joke: {joke}"
        response = await self.llm.acomplete(prompt)
        return StopEvent(result=str(response))
    

async def main():
    w = JokeFlow(timeout=60, verbose=False)
    result = await w.run(topic="climate change")
    print(str(result))

if __name__ == "__main__":
    import asyncio
    with tracer:
        asyncio.run(main())
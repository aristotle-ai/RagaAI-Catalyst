import asyncio

from agents import Agent, Runner
from agents.mcp import MCPServer, MCPServerStdio
from dotenv import load_dotenv
import os


load_dotenv()
import openai

from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer

load_dotenv()

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

catalyst = RagaAICatalyst(
    access_key=os.getenv('RAGAAI_CATALYST_ACCESS_KEY'), 
    secret_key=os.getenv('RAGAAI_CATALYST_SECRET_KEY'), 
    base_url=os.getenv('RAGAAI_CATALYST_BASE_URL')
)

tracer = Tracer(
    project_name=os.environ['RAGAAI_PROJECT_NAME'],
    dataset_name=os.environ['RAGAAI_DATASET_NAME'],
    tracer_type="mcp",
)

init_tracing(catalyst=catalyst, tracer=tracer)
# Connect to your Phoenix instance


async def run(mcp_server: MCPServer):
    agent = Agent(
        name="Assistant",
        instructions="Use the tools to answer the users question.",
        mcp_servers=[mcp_server],
    )
    while True:
        message = input("\n\nEnter your question (or 'exit' to quit): ")
        if message.lower() == "exit" or message.lower() == "q":
            break
        print(f"\n\nRunning: {message}")
        result = await Runner.run(starting_agent=agent, input=message)
        print(result.final_output)


async def main():
    async with MCPServerStdio(
        name="Financial Analysis Server",
        params={
            "command": "fastmcp",
            "args": ["run", "./server.py"],
        },
        client_session_timeout_seconds=30,
    ) as server:
        await run(server)
        
if __name__ == "__main__":
    asyncio.run(main())
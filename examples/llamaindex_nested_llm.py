import os
from llama_index.core.agent import ReActAgent
from llama_index.llms.anthropic import Anthropic
from llama_index.core.tools import FunctionTool
import anthropic

from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer
from dotenv import load_dotenv
load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

catalyst = RagaAICatalyst(
    access_key=os.getenv('RAGAAI_CATALYST_ACCESS_KEY'), 
    secret_key=os.getenv('RAGAAI_CATALYST_SECRET_KEY'), 
    base_url=os.getenv('RAGAAI_CATALYST_BASE_URL')
)

# Initialize tracer
tracer = Tracer(
    project_name='testing_v',
    dataset_name='testing_vv',
    tracer_type="agentic/llamaindex",
)

init_tracing(catalyst=catalyst, tracer=tracer)

# Initialize Anthropic LLMs with different specializations
strategist_llm = Anthropic(model="claude-3-opus-20240229")  # Senior strategist
analyst_llm = Anthropic(model="claude-3-sonnet-20240229")   # Market analyst
finance_llm = Anthropic(model="claude-3-haiku-20240307")    # Financial specialist

# 1. Create the Financial Analyst Agent
def financial_analysis(query: str) -> str:
    """Specialized in financial metrics and projections"""
    prompt = f"""As a financial expert, analyze this request:
    {query}
    
    Provide:
    1. Key financial considerations
    2. Risk assessment
    3. Recommended metrics to track
    """
    return str(finance_llm.complete(prompt))

finance_tool = FunctionTool.from_defaults(
    name="financial_analyst",
    description="Use for financial projections, risk assessment, and metric analysis",
    fn=financial_analysis
)

financial_agent = ReActAgent.from_tools(
    tools=[finance_tool],
    llm=finance_llm,
    verbose=True
)

# 2. Create the Market Analyst Agent
def market_analysis(query: str) -> str:
    """Specialized in market trends and competitive landscape"""
    prompt = f"""As a market analyst, evaluate:
    {query}
    
    Include:
    1. Market size potential
    2. Competitive landscape
    3. Customer segmentation
    """
    return str(analyst_llm.complete(prompt))

market_tool = FunctionTool.from_defaults(
    name="market_analyst",
    description="Use for market research, competition analysis, and growth opportunities",
    fn=market_analysis
)

market_agent = ReActAgent.from_tools(
    tools=[market_tool],
    llm=analyst_llm,
    verbose=True
)

# 3. Create the Chief Strategist Agent (main agent)
def consult_finance(query: str) -> str:
    return str(financial_agent.chat(query))

def consult_market(query: str) -> str:
    return str(market_agent.chat(query))

finance_consult_tool = FunctionTool.from_defaults(
    name="consult_finance",
    description="Access financial expertise for projections and risk analysis",
    fn=consult_finance
)

market_consult_tool = FunctionTool.from_defaults(
    name="consult_market",
    description="Access market intelligence and competitive analysis",
    fn=consult_market
)

chief_strategist = ReActAgent.from_tools(
    tools=[finance_consult_tool, market_consult_tool],
    llm=strategist_llm,
    verbose=True,
    system_prompt="""You are a Chief Strategy Officer. 
    Delegate specialized analysis to your team when needed, 
    then synthesize their inputs into executive recommendations."""
)

# 4. Run a strategic analysis
response = chief_strategist.chat(
    "We're considering expanding into the electric vehicle charging station market. "
    "Prepare a strategic analysis covering: "
    "1. Market potential in Southeast Asia "
    "2. Required capital investment "
    "3. Key success factors "
    "4. Recommended implementation timeline"
)

print("=== STRATEGIC RECOMMENDATION ===")
print(response)
import json
import os
from datetime import datetime, timedelta
import requests

import openai
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

# from phoenix.otel import register

load_dotenv()

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


# Configure OpenAI client
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4-turbo"

# Create MCP server
mcp = FastMCP("Financial Analysis Server")


class StockAnalysisRequest(BaseModel):
    ticker: str
    time_period: str = "short-term"  # short-term, medium-term, long-term


@mcp.tool()
# @tracer.tool(name="MCP.analyze_stock") # this OpenInference call adds tracing to this method
def analyze_stock(request: StockAnalysisRequest) -> dict:
    """Analyzes a stock based on its ticker symbol and provides investment recommendations."""

    try:
        # Alpha Vantage API key
        alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not alpha_vantage_key:
            return {
                "error": "ALPHA_VANTAGE_API_KEY not found in environment variables",
                "ticker": request.ticker,
                "time_horizon": request.time_period
            }
        
        # Fetch current stock quote
        quote_url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={request.ticker}&apikey={alpha_vantage_key}"
        quote_response = requests.get(quote_url)
        quote_data = quote_response.json()
        
        if "Global Quote" not in quote_data:
            return {
                "error": f"Invalid ticker symbol or API limit reached for {request.ticker}",
                "ticker": request.ticker,
                "time_horizon": request.time_period,
                "api_response": quote_data
            }
        
        quote = quote_data["Global Quote"]
        
        # Fetch company overview
        overview_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={request.ticker}&apikey={alpha_vantage_key}"
        overview_response = requests.get(overview_url)
        overview_data = overview_response.json()
        
        # Extract key data
        current_price = float(quote.get("05. price", 0))
        change_percent = quote.get("10. change percent", "0%").replace("%", "")
        previous_close = float(quote.get("08. previous close", 0))
        
        # Company overview data
        company_name = overview_data.get("Name", "Unknown")
        market_cap = overview_data.get("MarketCapitalization", "0")
        pe_ratio = overview_data.get("PERatio", "N/A")
        dividend_yield = overview_data.get("DividendYield", "0")
        fifty_two_week_high = overview_data.get("52WeekHigh", "0")
        fifty_two_week_low = overview_data.get("52WeekLow", "0")
        
        # Fetch historical data for performance analysis
        period_map = {
            "short-term": "DAILY",
            "medium-term": "WEEKLY", 
            "long-term": "MONTHLY"
        }
        
        interval = period_map.get(request.time_period, "DAILY")
        hist_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_{interval}&symbol={request.ticker}&apikey={alpha_vantage_key}"
        hist_response = requests.get(hist_url)
        hist_data = hist_response.json()
        
        # Calculate recent performance
        recent_change = float(change_percent) if change_percent != "0" else 0
        
        # Create prompt with real data for LLM analysis
        prompt = f"""
        Analyze the following real stock data for {request.ticker} from Alpha Vantage API and provide investment recommendations:
        
        Company: {company_name}
        Current Price: ${current_price:.2f}
        Market Cap: ${market_cap}
        P/E Ratio: {pe_ratio}
        Dividend Yield: {dividend_yield}
        52-Week High: ${fifty_two_week_high}
        52-Week Low: ${fifty_two_week_low}
        Recent Performance: {recent_change:.2f}%
        Previous Close: ${previous_close:.2f}
        
        Time horizon: {request.time_period}
        
        Based on this real Alpha Vantage API data, provide:
        1. Company overview
        2. Recent financial performance analysis
        3. Risk assessment
        4. Investment recommendation
        
        Format your response as a JSON object with the following structure:
        {{
            "ticker": "{request.ticker}",
            "company_name": "{company_name}",
            "current_price": {current_price:.2f},
            "overview": "Brief company description and business model",
            "financial_performance": "Analysis of recent performance based on the {recent_change:.2f}% change",
            "key_metrics": {{
                "market_cap": "{market_cap}",
                "pe_ratio": "{pe_ratio}",
                "dividend_yield": "{dividend_yield}",
                "52_week_high": "{fifty_two_week_high}",
                "52_week_low": "{fifty_two_week_low}"
            }},
            "risk_assessment": "Analysis of risks based on volatility and market conditions",
            "recommendation": "Buy/Hold/Sell recommendation with explanation",
            "time_horizon": "{request.time_period}"
        }}
        """

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        analysis = json.loads(response.choices[0].message.content)
        
        # Add real-time data to the response
        analysis["real_time_data"] = {
            "data_source": "Alpha Vantage API",
            "last_updated": datetime.now().isoformat(),
            "current_price": current_price,
            "market_cap": market_cap,
            "pe_ratio": pe_ratio,
            "dividend_yield": dividend_yield,
            "52_week_high": fifty_two_week_high,
            "52_week_low": fifty_two_week_low,
            "recent_change_percent": recent_change,
            "previous_close": previous_close
        }
        
        return analysis
        
    except Exception as e:
        return {
            "error": f"Failed to fetch stock data for {request.ticker}: {str(e)}",
            "ticker": request.ticker,
            "time_horizon": request.time_period
        }

# ... define any additional MCP tools you wish

if __name__ == "__main__":
    mcp.run()
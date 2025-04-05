# Haystack News Fetching Example with RagaAI Catalyst

This example demonstrates how to implement a news fetching agent with Haystack and RagaAI Catalyst for tracing and monitoring. The agent can use tools (like web search) to answer user queries more effectively.

## Overview

The example builds an agent that can:
1. Process user queries and determine if tools are needed
2. Execute web searches using the SerperDev API
3. Route responses based on whether tool calls are needed
4. Track the conversation history for context
5. Monitor the entire process using RagaAI Catalyst

## Prerequisites

- OpenAI API key
- SerperDev API key
- RagaAI Catalyst credentials

## Environment Variables

Create a `.env` file with the following variables:

```
CATALYST_ACCESS_KEY=your_access_key
CATALYST_SECRET_KEY=your_secret_key
CATALYST_BASE_URL=your_base_url
PROJECT_NAME=your_project_name
DATASET_NAME=your_dataset_name
OPENAI_API_KEY=your_openai_api_key
SERPERDEV_API_KEY=your_serperdev_api_key
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Components

### MessageCollector
A custom component that maintains conversation history by collecting and storing messages throughout the interaction.

### Pipeline Components
- OpenAIChatGenerator: Processes messages and determines tool usage
- ConditionalRouter: Routes responses based on tool call presence
- ToolInvoker: Executes tool calls (web search in this example)
- SerperDevWebSearch: Performs web searches using the SerperDev API

## Pipeline Flow

1. User query is processed by the chat generator
2. Router checks if tool calls are needed
3. If tools are needed:
   - Tool calls are executed
   - Results are collected and sent back to the generator
4. Final response is generated and returned

## Usage

Run the script:
```bash
python news_fetching.py
```

The example includes a sample query about fetching news on mars.

## Monitoring

The implementation includes RagaAI Catalyst integration for tracing and monitoring your agent's behavior. Access the Catalyst dashboard to:
- Track tool usage patterns
- Monitor response quality
- Analyze conversation flows
- Debug tool call decisions
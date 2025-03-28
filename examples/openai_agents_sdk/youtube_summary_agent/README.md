# YouTube Summary Agent with OpenAI Agents SDK

This example demonstrates how to use the OpenAI Agents SDK with RagaAI Catalyst to create a YouTube video summarizer that can extract and summarize content from YouTube videos.

## Overview

The application uses OpenAI's Agents SDK to:
- Search for YouTube videos based on user queries
- Extract transcripts from YouTube videos
- Generate concise summaries of video content
- Handle different types of user inputs (direct video URLs, channel URLs, or search terms)

The system uses multiple agents to handle different aspects of the workflow, including a clarifier agent for ambiguous queries and a summarizer agent for generating the final summary.

## Requirements

- Python 3.8+
- OpenAI API key
- YouTube Data API key
- RagaAI Catalyst credentials (optional, for tracing)

## Installation

1. Clone the repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy the sample environment file and add your API keys:
   ```bash
   cp sample.env .env
   ```
## Environment Variables
Configure the following environment variables in your .env file:

- OPENAI_API_KEY: Your OpenAI API key
- YOUTUBE_API_KEY: Your YouTube Data API key
- CATALYST_ACCESS_KEY: Your RagaAI Catalyst access key (optional)
- CATALYST_SECRET_KEY: Your RagaAI Catalyst secret key (optional)
- CATALYST_BASE_URL: RagaAI Catalyst base URL (optional)
- PROJECT_NAME: Name for your project in RagaAI Catalyst (optional)
- DATASET_NAME: Name for your dataset in RagaAI Catalyst (optional)

## Usage
Run the example script:
```bash
python youtube_summary_agent.py
```

The script will prompt you to enter a query, which can be:

- A direct YouTube video URL (e.g., https://www.youtube.com/watch?v=...)
- A YouTube channel URL followed by a search term (e.g., https://www.youtube.com/@channel - search term)
- A general search term (e.g., machine learning tutorial)
The script will then:

1. Process your query to identify the target video
2. Retrieve the video transcript
3. Generate a concise summary of the video content
4. Display the summary and the video link


## Features
- **Flexible Input Handling**: Accepts different types of user queries and intelligently processes them
- **Channel-Specific Searches**: Can search within a specific YouTube channel for relevant content
- **Clarification Agent**: Asks follow-up questions when user input is ambiguous
- **Transcript Extraction**: Automatically retrieves and processes video transcripts
- **AI-Powered Summarization**: Uses OpenAI's models to generate concise, readable summaries

## Integration with RagaAI Catalyst
This example integrates with RagaAI Catalyst for tracing and monitoring agent interactions. The integration helps with:

- Tracking agent performance
- Debugging complex agent workflows
- Collecting data for future improvements

## Customization
You can modify the agent instructions in the script to change the style or format of the summaries generated. The summarizer agent can be customized to produce different types of content, such as bullet points, longer analyses, or content focused on specific aspects of the videos.
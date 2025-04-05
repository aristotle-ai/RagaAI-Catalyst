# Most Upvoted Paper Summarizer

This script fetches, downloads, and summarizes the most upvoted paper from Hugging Face daily papers. It uses SmoLAgents to create a pipeline that:

1. Fetches the top paper from Hugging Face
2. Gets its arXiv ID
3. Downloads the paper
4. Reads and summarizes its content

## Features

- Automated paper discovery from Hugging Face's daily papers
- ArXiv integration for paper downloads
- PDF processing with first 3 pages analysis
- LLM-powered summarization using Qwen2.5-Coder-32B
- Modular tool-based architecture using SmoLAgents

## Components

- `get_hugging_face_top_daily_paper()`: Scrapes and retrieves the most upvoted paper from HuggingFace
- `get_paper_id_by_title()`: Finds the corresponding arXiv ID for a paper title
- `download_paper_by_id()`: Downloads the paper PDF from arXiv
- `read_pdf_file()`: Processes the PDF and extracts text from the first three pages

## Requirements

- SmoLAgents
- Hugging Face API token
- Dependencies:
  - arxiv
  - requests
  - beautifulsoup4
  - huggingface_hub
  - pypdf

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Set up your Hugging Face API token:
   - Replace 'HF_API_TOKEN' in the code with your actual token
   - Or set it as an environment variable

## Usage

```python
from most_upvoted_paper import main

# Run the paper summarization pipeline
main()
```

## Output

The script will:
1. Print the total number of pages in the downloaded paper
2. Process the first three pages
3. Generate a summary using the Qwen2.5-Coder model

## Note

This is an example implementation using the SmoLAgents framework. The script demonstrates how to create a complex pipeline by combining multiple tools and LLM capabilities.
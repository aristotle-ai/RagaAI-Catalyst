import os
import re
import json
import subprocess
import logging
from typing import Dict, Optional, List
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(command, cwd: Optional[str] = None):
    cwd = cwd or os.getcwd()
    logger.info(f"Running command: {command} in cwd: {cwd}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Command run successfully")
        output = result.stdout + '\n' + result.stderr
        return output
    except Exception as e:
        logger.error(f"Command failed: {e}")
        raise


def extract_information(logs: str) -> str:
    print("Extracting information from logs")
    
    # Define the patterns
    patterns = [
        re.compile(r"Trace saved to (.*)$"), 
        re.compile(r"Trace file saved at (.*)$"),
        # re.compile(r"Uploading trace metrics for (.*)$"),
        # re.compile(r"Uploading agentic traces for (.*)$"),
        re.compile(r"Submitting new upload task for file: (.*)$")
    ]
    
    # Split the text into lines to process them individually
    lines = logs.splitlines()
    locations = []

    # Search each line for the patterns
    for pattern in patterns: 
        for line in lines:
            match = pattern.search(line)
            if match:
                # The captured group (.*) will contain the file path
                locations.append(match.group(1).strip())
        if len(locations) > 0:
            break
    
    return locations

def load_trace_data(locations: List[str]) -> Dict:
    final_data = {}
    for location in locations:
        try:
            with open(location, 'r') as f:
                data = json.load(f)
                if len(str(data)) > len(str(final_data)):
                    final_data = data
        except Exception as e:
            continue

    if final_data == {}:
        raise ValueError("No trace data found")
    return final_data



import json
import os
import requests
import logging
from importlib import resources
from dataclasses import asdict

logger = logging.getLogger(__name__)

def convert_usage_to_dict(usage):
    # Initialize the token_usage dictionary with default values
    token_usage = {
        "input": 0,
        "completion": 0,
        "reasoning": 0,  # Default reasoning tokens to 0 unless specified
    }

    if usage:
        if isinstance(usage, dict):
            # Access usage data as dictionary keys
            token_usage["input"] = usage.get("prompt_tokens", 0)
            token_usage["completion"] = usage.get("completion_tokens", 0)
            # If reasoning tokens are provided, adjust accordingly
            token_usage["reasoning"] = usage.get("reasoning_tokens", 0)
        else:
            # Handle the case where usage is not a dictionary
            # This could be an object with attributes, or something else
            try:
                token_usage["input"] = getattr(usage, "prompt_tokens", 0)
                token_usage["completion"] = getattr(usage, "completion_tokens", 0)
                token_usage["reasoning"] = getattr(usage, "reasoning_tokens", 0)
            except AttributeError:
                # If attributes are not found, log or handle the error as needed
                print(f"Warning: Unexpected usage type: {type(usage)}")

    return token_usage


def calculate_cost(
    token_usage,
    input_cost_per_token=0.0,
    output_cost_per_token=0.0,
    reasoning_cost_per_token=0.0,
):
    """
    Calculate the cost of token usage based on specified cost rates.
    
    This function computes the cost for each token category—input, completion, and reasoning—by multiplying the count of tokens by their respective cost per token. The counts are extracted from the provided `token_usage` dictionary using the keys "prompt_tokens", "completion_tokens", and "reasoning_tokens". The total cost is the sum of the individual costs.
    
    Parameters:
        token_usage (dict): A dictionary containing token counts. Expected keys include:
            - "prompt_tokens" (int): The number of input tokens.
            - "completion_tokens" (int): The number of output tokens.
            - "reasoning_tokens" (int): The number of reasoning tokens.
        input_cost_per_token (float, optional): Cost per input token. Default is 0.0.
        output_cost_per_token (float, optional): Cost per output token. Default is 0.0.
        reasoning_cost_per_token (float, optional): Cost per reasoning token. Default is 0.0.
    
    Returns:
        dict: A dictionary with the following cost details:
            - "input": The cost computed for input tokens.
            - "completion": The cost computed for completion tokens.
            - "reasoning": The cost computed for reasoning tokens.
            - "total": The total cost, which is the sum of input, completion, and reasoning costs.
    
    Example:
        >>> usage = {"prompt_tokens": 10, "completion_tokens": 20, "reasoning_tokens": 5}
        >>> calculate_cost(usage, input_cost_per_token=0.01, output_cost_per_token=0.02, reasoning_cost_per_token=0.03)
        {'input': 0.1, 'completion': 0.4, 'reasoning': 0.15, 'total': 0.65}
    """
    input_tokens = token_usage.get("prompt_tokens", 0)
    output_tokens = token_usage.get("completion_tokens", 0)
    reasoning_tokens = token_usage.get("reasoning_tokens", 0)

    input_cost = input_tokens * input_cost_per_token
    output_cost = output_tokens * output_cost_per_token
    reasoning_cost = reasoning_tokens * reasoning_cost_per_token

    total_cost = input_cost + output_cost + reasoning_cost

    return {
        "input": input_cost,
        "completion": output_cost,
        "reasoning": reasoning_cost,
        "total": total_cost,
    }

def log_event(event_data, log_file_path):
    """
    Logs event data to a specified file in JSON format.
    
    This function converts a dataclass instance into a dictionary using `asdict`,
    serializes it into a JSON-formatted string with `json.dumps`, and appends the result
    to the file indicated by `log_file_path`. Each log entry is written on a new line.
    
    Parameters:
        event_data (dataclass): A dataclass instance containing the event data to log.
        log_file_path (str): The path to the log file where the event data will be appended.
    
    Raises:
        IOError: If an error occurs while opening or writing to the log file.
    """
    event_data = asdict(event_data)
    with open(log_file_path, "a") as f:
        f.write(json.dumps(event_data) + "\n")

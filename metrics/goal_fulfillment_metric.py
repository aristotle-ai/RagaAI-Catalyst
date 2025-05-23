import litellm
import json
import ast
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

def get_model_response(prompt, model="gpt-4o-mini"):
    evaluation = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    result = evaluation.choices[0].message.content
    return result


def extract_data_from_json(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    spans = json_data.get("data", [])[0].get("spans", [])

    all_calls = []
    for span in spans:
        function_name = span.get("name", "")
        span_type = span.get("attributes", {}).get("openinference.span.kind", "")
        input_val = span.get("attributes", {}).get("input.value", "")
        output_val = span.get("attributes", {}).get("output.value", "")
        start_time = span.get("start_time", "")
        end_time = span.get("end_time", "")
        
        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
        all_calls.append({
            "function_name": function_name,
            "span_type": span_type,
            "input": input_val,
            "output": output_val,
            "start_time": start_dt,
            "end_time": end_dt,
        })

    conversations = []
    all_calls_sorted = sorted(all_calls, key=lambda x: x["start_time"])
    for call in all_calls_sorted:
        conversation_entry = {
            "function_name": call["function_name"],
            "input": call["input"],
            "response": call["output"],
        }
        conversations.append(conversation_entry)

   
    first_input = sorted(all_calls, key=lambda x: x["start_time"])[0].get("input", "")
    last_output = sorted(all_calls, key=lambda x: x["end_time"])[-1].get("output", "")

    return first_input, last_output, conversations


def extract_user_goal(conversation):
    prompt = f"""
    You are an AI system designed to analyze conversations between users and agents to identify the primary user intent. Your task is to process the conversation history and provide a clear, concise statement of the user's main goal or purpose.

    ## Input
    You will receive a conversation log containing:
    1. Function names
    2. User inputs or system prompts
    3. LLM or system responses

    ## Output
    Provide a clear description of the main goal driving the user's interaction, expressed in 3-4 sentences.

    ## Analysis Guidelines

    1. Clarity is Key: Provide a clear, unambiguous statement of intent. Avoid vague or overly broad interpretations.
    2. Context Matters: Consider the full context of the conversation, including any background information provided by the user.
    3. Evolution of Intent: If the user's intent changes during the conversation, identify the most recent or prominent intent.
    4. Objectivity: Base your assessment solely on the conversation content, avoiding unsupported assumptions or inferences.
    5. Conciseness: Express the intent in 3-4 well-formulated sentences.
    6. User-Centric: Focus on what the user wants to achieve, not on the system's actions or responses.
    7. Action-Oriented: When possible, frame the intent in terms of an action or goal the user wants to accomplish.
    8. Completeness: Use the 3-4 sentences to capture any nuances or complexities in the user's intent.

    ## Output Format
    Your output should be 3-4 sentences

    ## Example
    Input Conversation:
    [
        {{
            "function_name": "get_travel_recommendations",
            "input": "I'm planning a trip to Japan next month. Can you suggest some must-visit places in Tokyo?",
            "response": "Certainly! Tokyo has many exciting attractions. Some must-visit places include the historic Senso-ji Temple, the bustling Shibuya Crossing, and the serene Meiji Shrine. The Tokyo Skytree offers panoramic views of the city, while the Tsukiji Outer Market is perfect for food lovers."
        }},
        {{
            "function_name": "get_cultural_experiences",
            "input": "Those sound great! I'm particularly interested in experiencing traditional Japanese culture. Any specific recommendations for that?",
            "response": "For traditional Japanese culture in Tokyo, I'd highly recommend visiting..."
        }}
    ]

    Output:
    The user intends to plan a culturally enriching trip to Tokyo, Japan. They are seeking recommendations for must-visit places, with a particular emphasis on experiences that showcase traditional Japanese culture. The user's intent has evolved from a general interest in Tokyo's attractions to a more focused desire for authentic cultural experiences, indicating a preference for immersive and historically significant sites over modern or purely touristic destinations.

    Analyse the given conversation:
    {conversation}

    Remember, your goal is to provide a clear, high-quality identification of the user's primary intent in 3-4 sentences. Prioritize accuracy and clarity in your analysis while capturing the full scope of the user's goals.
    """

    return get_model_response(prompt)


def get_goal_fulfillment_result(input_prompt, user_goal, last_output):
    prompt = f"""
You are an AI system designed to analyze the effectiveness of a response in fulfilling a user's goal. Your task is to evaluate the system response based on both the original user prompt and the user's goal, and then provide a concise, reasoned judgment.

## Input
You will receive:
1. The original user prompt (i.e., what the user typed)
2. A user goal statement (i.e., what the user ultimately wants to achieve)
3. A system-generated response

## Output
Evaluate how well the system response fulfills the user's goal **given the original prompt**. Provide:
- A brief explanation ("reason") of your evaluation
- A fulfillment score from 0 to 1, where:
  - 0 means the response does not fulfill the goal at all
  - 1 means it completely fulfills the goal

## Example
User Prompt: "I'm planning a trip to Japan next month. Can you suggest some must-visit places in Tokyo?"
User Goal: "Get useful travel suggestions for Tokyo."
System Response: "Certainly! Tokyo has many exciting attractions. Some must-visit places include the historic Senso-ji Temple, the bustling Shibuya Crossing, and the serene Meiji Shrine."

Output:
{{
    'score': 1,
    'reason': "The system's response effectively fulfills the user's goal by providing specific, relevant travel suggestions for Tokyo."
}}

## NOTE: Output format must be:
{{
    'score': <score>,
    'reason': "<brief explanation>"
}}

## INPUT
User Prompt: {input_prompt}
User Goal: {user_goal}
System Response: {last_output}
"""
    return get_model_response(prompt)



def execute_goal_fulfillment_metric(trace_path, input_prompt=None, user_goal=None):
   
    # Extract first_input, last_response and conversations from the trace
    first_input, last_output, conversations = extract_data_from_json(trace_path)

    # Take the first input as the default input prompt if not provided
    if not input_prompt:
        input_prompt = first_input

    # Extract user goal from conversations if not provided
    if not user_goal:
        user_goal = extract_user_goal(conversations)

    # Calculate goal fulfillment rate
    result = get_goal_fulfillment_result(
        input_prompt,
        user_goal,
        last_output
    )

    try:
        result = json.loads(result)
    except json.JSONDecodeError:
        result = ast.literal_eval(result)

    return {
        "score": result['score'],
        "reason": result['reason'],
    }


if __name__ == "__main__":
    json_path = '/Users/ragaai_user/Desktop/tracing/rag_agent_traces.json'
    input_prompt = None
    user_goal = "book a flight to Japan"
    result = execute_goal_fulfillment_metric(trace_path=json_path, input_prompt=input_prompt, user_goal=user_goal)
    print(result)
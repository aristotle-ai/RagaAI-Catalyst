import json

def extract_final_trace(data):
        
        trace_aggregate = {}
        for entry in data:
            trace_id = entry["trace_id"]
            if trace_id not in trace_aggregate:
                # Initialize a new entry if trace_id is encountered for the first time
                trace_aggregate[trace_id] = {
                    "trace_id": trace_id,
                    "session_id": entry["session_id"],
                    "pipeline": entry["pipeline"],
                    "metadata": entry["metadata"],
                    "prompt_length": 0,
                    "data": {},
                }

            # Process traces and update the existing or new dictionary
            for trace in entry["traces"]:
                try:
                    result = trace_aggregate[trace_id]
                    if "expected_response" in trace:
                        result["data"]["expected_response"] = trace["expected_response"]
                        continue
                    if trace["name"] == "retrieve_documents.langchain.workflow":
                        prompt = json.loads(trace["attributes"]["traceloop.entity.input"])[
                            "kwargs"
                        ]["input"]
                        result["data"]["prompt"] = prompt
                    
                    if trace["name"] == "PromptTemplate.langchain.task":
                        context = json.loads(trace["attributes"]["traceloop.entity.input"])[
                            "kwargs"
                        ]["context"]
                        system_prompt = json.loads(
                            trace["attributes"]["traceloop.entity.output"]
                        )["kwargs"]["text"]
                        result["data"]["context"] = context
                        result["data"]["system_prompt"] = system_prompt
                    if (
                            trace["name"] == "ChatOpenAI.langchain.task"
                            or trace["name"] == "ChatGroq.langchain.task"
                    ):
                        response = trace["attributes"]["gen_ai.completion.0.content"]
                        result["data"]["response"] = response
                except Exception as e:
                    result["Error"] = str(e)
        for key,value in trace_aggregate.items() :
             return value
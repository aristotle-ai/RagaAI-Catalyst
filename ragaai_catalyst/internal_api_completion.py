import requests
import json
import os
import subprocess
import logging
import tempfile
import traceback
import pandas as pd
import time
from logging.handlers import RotatingFileHandler

log_dir = os.path.join(tempfile.gettempdir(), "ragaai_logs")
print(log_dir)
os.makedirs(log_dir, exist_ok=True)

max_file_size = 5 * 1024 * 1024  # 5 MB
backup_count = 1  # Number of backup files to keep

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            os.path.join(log_dir, "internal_api_completion.log"),
            maxBytes=max_file_size,
            backupCount=backup_count
        )
    ]
)
logger = logging.getLogger("internal_api_completion")

def api_completion(messages, model_config, kwargs):
    attempts = 0
    while attempts < 3:

        user_id = kwargs.get('user_id', '1')
        internal_llm_proxy = kwargs.get('internal_llm_proxy', -1)
            
            
        job_id = model_config.get('job_id',-1)
        converted_message = convert_input(messages,model_config, user_id)
        payload = json.dumps(converted_message)
        headers = {
            'Content-Type': 'application/json',
            # 'Wd-PCA-Feature-Key':f'your_feature_key, $(whoami)'
        }
        try:
            start_time = time.time()
            response = requests.request("POST", internal_llm_proxy, headers=headers, data=payload)
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"API Call: [POST] {internal_llm_proxy} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
            response.raise_for_status()
            if model_config.get('log_level','')=='debug':
                logger.info(f'Model response Job ID {job_id} {response.text}')
            #change raise to pass
            if response.status_code!=200:
                # logger.error(f'Error in model response Job ID {job_id}:',str(response.text))
                print(str(response.text))
                pass
            
            if response.status_code==200:
                response = response.json()   
                #change raise to pass             
                if "error" in response:
                    print(response["error"]["message"])
                    pass
                else:
                    result=  response["choices"][0]["message"]["content"]
                    response1 = result.replace('\n', '').replace('```json','').replace('```', '').strip()
                    try:
                        json_data = json.loads(response1)
                        df = pd.DataFrame(json_data)
                        return(df)
                    except json.JSONDecodeError:
                        attempts += 1  # Increment attempts if JSON parsing fails
                        #change raise to pass
                        if attempts == 3:
                            print("Failed to generate a valid response after multiple attempts.")
                            pass
        #change raise to pass
        except Exception as e:
            print(f"{e}")
            pass


def get_username():
    result = subprocess.run(['whoami'], capture_output=True, text=True)
    result = result.stdout
    return result


def convert_input(messages, model_config, user_id):
    doc_input = {
      "model": model_config.get('model'),
      **model_config,
      "messages": messages,
      "user_id": user_id
    }
    return doc_input


if __name__=='__main__':
    messages = [
        {
            "role": "system",
            "content": "you are a poet well versed in shakespeare literature"
        },
        {
          "role": "user",
          "content": "write a poem on pirates and penguins"
        }
      ]
    kwargs = {"internal_llm_proxy": "http://13.200.11.66:4000/chat/completions", "user_id": 1}
    model_config = {"model": "workday_gateway", "provider":"openai", "max_tokens": 10}
    answer = api_completion(messages, model_config, kwargs)
    print(answer)
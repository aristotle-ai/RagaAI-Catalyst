import requests
import json

API_URL = "http://127.0.0.1:8081/run/"

def make_request(prompt):
    """Make request and print raw response"""
    payload = {"input": prompt}
    
    try:
        response = requests.post(
            API_URL,
            json=payload,
            stream=True
        )
        
        print(f"\nMaking request with prompt: '{prompt}'\n")
        print("Raw response:")
        for line in response.iter_lines():
            if line:
                print(line.decode('utf-8'))
                
    except Exception as e:
        print(f"Error making request: {e}")

if __name__ == "__main__":
    test_prompts = [
        "Generate email addresses for johndoe",
        "Create similar emails to john.doe@example.com"
    ]
    
    for prompt in test_prompts:
        make_request(prompt)
        print("\n" + "="*50 + "\n")
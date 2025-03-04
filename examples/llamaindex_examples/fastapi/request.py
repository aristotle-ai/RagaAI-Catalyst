import requests


url = "http://localhost:8000/run-agent/"  

payload = {
    "input": "Hello, how can I get help with my task? Add 4000 and 23231?"  
}

response = requests.post(url, json=payload)

print(response.json())
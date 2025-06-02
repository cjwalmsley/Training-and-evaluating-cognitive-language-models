import requests
import ollama

url_base_string = "http://localhost:11434/api/"

def prompt(prompt_string, model_string, ):
    url = "http://localhost:11434/api/"

    data = {
        "model": model_string,
        "messages": [
            {
                "role": "user",
                "content": prompt_string
            }
        ],
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()["message"]["content"]

def generated_text_from_prompt(the_prompt, model_string):
    generated_text = ollama.generate(model=model_string, prompt=the_prompt)
    return generated_text.response
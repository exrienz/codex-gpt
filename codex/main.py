#!/usr/bin/env python3
import argparse
import json
import sys
import time
import os
import requests
import threading
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type

API_URL = "https://gpt.code-x.my/api/generate"
DEFAULT_MODEL = "qwen2.5:3b"
MAX_TOKENS = 2048

class GPTAPIError(Exception):
    pass

def get_auth_token():
    token = os.getenv("OLLAMA_API_KEY")
    if not token:
        raise EnvironmentError("Please set the OLLAMA_API_KEY environment variable.")
    return token

def enforce_token_limit(prompt, max_tokens):
    estimated_tokens = int(len(prompt.split()) * 1.33)
    if estimated_tokens > max_tokens:
        raise ValueError(f"Prompt too long: estimated {estimated_tokens} tokens > {max_tokens} limit.")
    return prompt

def start_loading_spinner(stop_event):
    spinner = [
        "[ ⠋ ]", "[ ⠙ ]", "[ ⠹ ]", "[ ⠸ ]",
        "[ ⠼ ]", "[ ⠴ ]", "[ ⠦ ]", "[ ⠧ ]",
        "[ ⠇ ]", "[ ⠏ ]"
    ]
    idx = 0
    while not stop_event.is_set():
        print(f"\r{spinner[idx % len(spinner)]} Thinking...", end="", flush=True)
        idx += 1
        time.sleep(0.1)
    print("\r" + " " * 40 + "\r", end="")

@retry(
    wait=wait_fixed(2),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((requests.exceptions.RequestException, GPTAPIError))
)
def call_gpt_api(model, prompt, stream=True):
    prompt = enforce_token_limit(prompt, MAX_TOKENS)
    auth_token = get_auth_token()
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "prompt": prompt
    }

    response = requests.post(API_URL, headers=headers, data=json.dumps(data), stream=True)
    if response.status_code != 200:
        raise GPTAPIError(f"HTTP {response.status_code}: {response.text}")

    if stream:
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    chunk = json.loads(line)
                    print(chunk.get("response", ""), end="", flush=True)
                except json.JSONDecodeError:
                    continue
        print()
    else:
        stop_event = threading.Event()
        spinner_thread = threading.Thread(target=start_loading_spinner, args=(stop_event,))
        spinner_thread.start()

        response_text = ""
        try:
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        chunk = json.loads(line)
                        response_text += chunk.get("response", "")
                    except json.JSONDecodeError:
                        continue
        finally:
            stop_event.set()
            spinner_thread.join()

        print(response_text or "No response")

def main():
    parser = argparse.ArgumentParser(description="CLI wrapper for Ollama GPT API")
    parser.add_argument("prompt", nargs=1, help="Instruction or question for the model (enclose in quotes)")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--complete", action="store_true", help="Wait for full completion instead of streaming")

    args = parser.parse_args()

    try:
        stdin_data = sys.stdin.read().strip() if not sys.stdin.isatty() else ""
        prompt = f"{stdin_data.strip()}\n\n{args.prompt[0].strip()}" if stdin_data else args.prompt[0].strip()
        call_gpt_api(args.model, prompt, stream=not args.complete)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
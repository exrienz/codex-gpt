import os
import sys
import json
import argparse
import threading
import requests
from tenacity import retry, stop_after_attempt, wait_fixed
from signal import signal, SIGINT

# Constants
MAX_TOKENS = 2048
API_URL = "https://gpt.code-x.my/api/generate"


def estimate_tokens(text: str) -> int:
    """Rough token estimate: 1.33 tokens per word."""
    return int(len(text.split()) * 1.33)


def send_api_request(prompt: str, model: str, stream: bool):
    """Send a request to the API with retry logic."""
    headers = {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}
    payload = {"prompt": prompt, "model": model, "stream": stream}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _call():
        resp = requests.post(API_URL, json=payload, headers=headers, stream=stream)
        resp.raise_for_status()
        return resp

    return _call()


def spinner(stop_event: threading.Event):
    """Simple CLI spinner until stop_event is set."""
    while not stop_event.is_set():
        for char in '|/-\\':
            sys.stdout.write(f"\r{char} Thinking...")
            sys.stdout.flush()
            if stop_event.wait(0.1):
                break
    sys.stdout.write("\r ")  # clear spinner


def interactive_chat(model: str):
    """Launches an interactive chat session. Press Ctrl+C to exit."""
    print("Entering interactive chat mode. Press Ctrl+C to exit.")
    try:
        while True:
            print()  # blank line before prompt
            prompt = input("You: ")
            if not prompt.strip():
                continue
            if estimate_tokens(prompt) > MAX_TOKENS:
                print(f"Error: Prompt exceeds {MAX_TOKENS} token limit.")
                continue
            resp = send_api_request(prompt, model, stream=True)
            for line in resp.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        print(chunk.get("response", ""), end="")
                    except json.JSONDecodeError:
                        continue
            print()
    except KeyboardInterrupt:
        print("\nExiting chat mode.")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(prog='codex')
    parser.add_argument('prompt', nargs='*', help='Prompt text for completion')
    parser.add_argument('-m', '--model', default='qwen2.5:latest', help='Model identifier')
    parser.add_argument('--complete', action='store_true', help='Wait for full response before printing')
    parser.add_argument('--chat', action='store_true', help='Enter interactive chat mode')
    args = parser.parse_args()

    api_key = os.getenv('OLLAMA_API_KEY')
    if not api_key:
        print("Error: OLLAMA_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    if args.chat:
        interactive_chat(args.model)
        return

    prompt_text = " ".join(args.prompt) or sys.stdin.read().strip()
    if not prompt_text:
        print("Error: No prompt provided.", file=sys.stderr)
        sys.exit(1)

    if estimate_tokens(prompt_text) > MAX_TOKENS:
        print(f"Error: Prompt exceeds {MAX_TOKENS} token limit.", file=sys.stderr)
        sys.exit(1)

    stream = not args.complete
    stop_event = threading.Event()

    try:
        resp = send_api_request(prompt_text, args.model, stream=stream)

        if stream:
            for line in resp.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        print(chunk.get("response", ""), end="")
                    except json.JSONDecodeError:
                        continue
        else:
            spinner_thread = threading.Thread(target=spinner, args=(stop_event,))
            spinner_thread.start()

            content = ""
            for line in resp.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        content += chunk.get("response", "")
                    except json.JSONDecodeError:
                        continue
            stop_event.set()
            spinner_thread.join()
            print(content)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

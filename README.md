# Codex-gpt

Codex is a lightweight CLI tool that wraps around the Ollama GPT API to generate text using models hosted at `https://gpt.code-x.my`.

## 🚀 Features

- ✅ Streamed and non-streamed output with smooth UX
- ✅ Retry and error handling
- ✅ Token limit enforcement
- ✅ Loading animation for non-streamed responses
- ✅ Environment variable support for API key

## 📦 Installation

```bash
git clone https://github.com/exrienz/codex-gpt.git
cd codex-gpt
pip install .
```

## 🔑 One-time Setup

Before using `codex`, export your API key:

```bash
export OLLAMA_API_KEY="sk-ollama-xxxxxxxxxxxxxxxxxxxxxxxxxx"
```

For permanent setup:
```bash
echo 'export OLLAMA_API_KEY="sk-ollama-xxxxxxxxxxxxxxxxxxxxxxxxxx"' >> ~/.bashrc
source ~/.bashrc
```

## 🧠 Usage

### Prompt via argument
```bash
codex -p "Why is the sky blue?" --stream
```

### Prompt from file
```bash
codex -f prompt.txt
```

## 🛠️ Development

To modify and reinstall:
```bash
pip install --force-reinstall .
```

## 📜 License

MIT License

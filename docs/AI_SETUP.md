# ptCenter v2 — AI Setup Guide

ptCenter supports four AI providers simultaneously. This guide covers how to set up each one, which to choose, and how to troubleshoot common issues.

---

## Quick Start

For most users: Gemini free tier is ready in under 2 minutes.

```bash
# 1 — Get a free Gemini API key:
#     https://aistudio.google.com/app/apikey

# 2 — Add it to your .env
echo "GEMINI_API_KEY=your_key_here" >> .env
echo "ACTIVE_AI_MODEL=gemini" >> .env

# 3 — Launch ptCenter — AI status is shown in the banner
ptcenter
```

---

## Provider Comparison

| Provider | Model | Cost | Internet | Privacy | Best Use Case |
|---|---|---|---|---|---|
| **Google Gemini** | `gemini-2.0-flash` | **Free** — 15 req/min, 1M tok/day | Required | Data sent to Google | Default for everything |
| **Ollama** | Any local model | **Free** — no limits | Not needed | Fully local | Air-gapped labs, privacy-sensitive |
| **OpenAI** | `gpt-4o` (configurable) | Paid | Required | Data sent to OpenAI | Highest quality analysis |
| **Anthropic Claude** | `claude-3-5-haiku-latest` | Paid | Required | Data sent to Anthropic | Fast + accurate analysis |

> All four can be configured at once. You switch between them at runtime from the Settings menu without restarting.

---

## Google Gemini (Recommended)

**Why Gemini first?** The free tier (no credit card required) gives 15 requests per minute and 1 million tokens per day — more than enough for intensive CTF sessions and full engagements.

### Setup

1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with a Google account
3. Click **Create API key** → copy the key
4. Add it to `.env`:

```ini
GEMINI_API_KEY=AIza...your_key_here
ACTIVE_AI_MODEL=gemini
```

### Model

ptCenter uses `gemini-2.0-flash` — Google's fast, efficient model with a 1M token context window. It handles the largest scan outputs without needing to chunk them.

### Free Tier Limits

| Limit | Value |
|---|---|
| Requests per minute | 15 |
| Tokens per day | 1,000,000 |
| Input token limit | 1,048,576 per request |
| Credit card required | No |

If you hit rate limits during intensive agentic recon sessions, ptCenter will display the API error. Wait 60 seconds and retry, or switch to Ollama as a fallback.

---

## Ollama — Local / Offline

Ollama runs AI models entirely on your machine. No API key, no internet, no data leaving your system. Ideal for air-gapped lab environments or when you want to keep scan data private.

### Install Ollama

```bash
# Official installer
curl -fsSL https://ollama.com/install.sh | sh

# Arch/CachyOS (AUR)
paru -S ollama
# Then enable the service:
sudo systemctl enable --now ollama
```

### Pull a Model

```bash
# Fast, capable, runs well on 8GB+ RAM
ollama pull qwen2.5:3b

# Better quality, needs 16GB+ RAM
ollama pull llama3

# Pentest-focused model (more up-to-date - slow)
ollama pull supergoatscriptguy/mythos-sec:8b

# Smaller alternative for limited hardware
ollama pull xploiter/pentester
```

### Configure ptCenter

```ini
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
ACTIVE_AI_MODEL=ollama
```

### Hardware Recommendations

| Hardware       | Recommended Model                 |
| -------------- | --------------------------------- |
| 8GB RAM (iGPU) | `qwen2.5:3b`                      |
| 16GB RAM       | `qwen2.5:7b` or `llama3`          |
| 32GB+ RAM      | `mixtral:8x7b` or `llama3:70b-q4` |

### Verify Ollama is Running

```bash
# Check the service
curl http://localhost:11434/api/tags

# List pulled models
ollama list

# Test a model directly
ollama run qwen2.5:3b "What is a SQL injection?"
```

---

## OpenAI GPT

### Setup

1. Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Create a new API key
3. Add credit to your account (minimum $5)
4. Add to `.env`:

```ini
OPENAI_API_KEY=sk-...your_key_here
OPENAI_MODEL=gpt-4o
ACTIVE_AI_MODEL=openai
```

### Available Models

| Model | Speed | Quality | Cost |
|---|---|---|---|
| `gpt-4o` | Fast | Highest | ~$5 / 1M tokens |
| `gpt-4o-mini` | Very fast | High | ~$0.15 / 1M tokens |
| `gpt-3.5-turbo` | Fast | Good | ~$0.50 / 1M tokens |

Set `OPENAI_MODEL` to any model string from the table above.

---

## Anthropic Claude

### Setup

1. Go to [https://console.anthropic.com/](https://console.anthropic.com/)
2. Create an account and add credits
3. Create an API key
4. Add to `.env`:

```ini
ANTHROPIC_API_KEY=sk-ant-...your_key_here
CLAUDE_MODEL=claude-3-5-haiku-latest
ACTIVE_AI_MODEL=claude
```

### Available Models

| Model | Speed | Quality | Cost |
|---|---|---|---|
| `claude-3-5-haiku-latest` | Fastest | Good | Lowest |
| `claude-3-5-sonnet-latest` | Fast | High | Medium |
| `claude-3-opus-latest` | Slower | Highest | Highest |

---

## Switching Models at Runtime

You don't need to restart ptCenter to change the AI model. From the main menu:

```
[12] Settings → [1] Select AI Model
```

All currently loaded models are listed. Select by number. The switch takes effect immediately for the next AI call.

The **active model** is indicated with a `◄ ACTIVE` marker in the list. Only models with valid, working credentials are shown.

---

## How AI Features Work

### Scan Analysis

Every module that produces scan output can optionally send it to the AI. The AI is prompted with a structured template:

```
Analyse this <scan_type> scan result and provide:
1. Executive Summary (2–3 sentences)
2. Identified vulnerabilities or security issues
3. Risk Assessment (Critical/High/Medium/Low for each finding)
4. Recommended next steps and mitigation strategies
5. Additional reconnaissance suggestions

Scan Results:
<output>
```

**Auto AI analysis** is on by default. Toggle it at `Settings → [2] Toggle Auto AI Analysis`.

### Chunked Analysis

Scan outputs larger than ~7,500 characters are automatically split into chunks. Each chunk is summarized individually, then a final synthesis prompt combines all summaries:

```
Chunk 1 (7,500 chars) → summary 1
Chunk 2 (7,500 chars) → summary 2
Chunk N (...)         → summary N
                            ↓
               Final unified analysis
```

This means ptCenter can analyze the output of a `nmap -p- -A` scan on a /24 range without truncation.

### Agentic Recon Loop

The AI receives scan output and must respond in strict JSON:

```json
{
  "tool": "nikto",
  "flags": "-host http://10.10.10.1 -p 8080",
  "target": "10.10.10.1",
  "reason": "HTTP service on 8080 detected by nmap — scanning for web vulnerabilities"
}
```

If the response is not valid JSON (accidental markdown fences, preamble, etc.), ptCenter strips common artifacts and retries the parse. If it still fails, the loop terminates gracefully.

### AI Security Chat

The chat REPL maintains conversation history:

```python
conversation_history = [
    {"role": "user",      "content": "What is a SSRF?"},
    {"role": "assistant", "content": "..."},
    {"role": "user",      "content": "Give me a bypass for 127.0.0.1"},
]
```

History is prepended to each new prompt as a formatted transcript. It is capped at 40 entries (20 turns) — older turns are dropped from the front to stay within token limits.

---

## Troubleshooting

### "No AI model configured"

The banner and settings show `✗ Disabled`. This means no API key was found in `.env` and Ollama is not reachable.

**Fix:**
```bash
# Check your .env exists and has a key
cat .env | grep API_KEY

# Check Ollama if using local model
curl http://localhost:11434/api/tags
systemctl status ollama
```

### Gemini 429 Rate Limit

```
Error 429: Resource exhausted — quota exceeded
```

You've hit the 15 req/min limit. Wait 60 seconds. For heavy agentic sessions, consider Ollama as a secondary model or upgrade to a paid Gemini plan.

### Ollama "connection refused"

```
✗ Ollama unavailable — is it running?
```

```bash
# Start Ollama
ollama serve

# Or enable systemd service
sudo systemctl start ollama

# Verify
curl http://localhost:11434/api/tags
```

### Ollama model not found

```bash
# List what you have
ollama list

# Pull your configured model
ollama pull qwen2.5:3b
```

The `OLLAMA_MODEL` in `.env` must exactly match a model name in `ollama list`.

### AI response is empty

Some models refuse certain security-related prompts. Try:
- Switching to a different model via Settings
- Using Ollama with a pentest-specific model (`mythos-sec:8b`)
- The `clear` command in AI chat to reset history

### Analysis cuts off mid-sentence

The model hit its `max_tokens` limit. ptCenter uses `max_tokens=2048` by default. For very dense scan outputs, chunked analysis is triggered automatically — if you're seeing truncated results on chunks, the individual chunk is still too large. This is rare.

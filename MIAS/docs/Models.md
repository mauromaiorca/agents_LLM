# Models & Providers

You can run with **OpenAI**, other providers like **DeepSeek**, or a **local LLM via Ollama**.
Both scripts load `.env` automatically; configure your provider there and tweak the model factory as needed.

> The default code uses `ChatOpenAI` from `langchain-openai`. Below are patterns to switch provider.

---

## 1) OpenAI (default)

`.env`:
```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

The code uses:
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=OPENAI_API_KEY)
```

---

## 2) DeepSeek (API)

DeepSeek exposes an OpenAI-compatible API in many SDKs. With LangChain, you can either:
- Use `ChatOpenAI` pointing to a custom base URL (if the SDK supports it), or
- Use the generic `ChatOpenAI` compatible wrapper provided by the ecosystem.

Example pattern (pseudo-code):
```python
import os
from langchain_openai import ChatOpenAI

# in .env
# DEEPSEEK_API_KEY=...
# DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
# DEEPSEEK_MODEL=deepseek-chat

api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

llm = ChatOpenAI(
    api_key=api_key,
    model=model,
    base_url=base_url,
    temperature=0,
)
```
> Check your account limits and model names in the provider’s docs. If your SDK doesn’t support `base_url`, use a community integration for DeepSeek with LangChain (names sometimes vary).

---

## 3) Local LLM via Ollama

Install Ollama and pull a model:
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3
```

Use LangChain’s Ollama chat model:
```python
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="llama3", temperature=0)
```

**Wiring into the pipeline:** replace `make_llm()` in `pdf_relevance_pipeline.py` with the provider of your choice.
Be mindful that smaller local models may need reduced context (e.g., shorter snippets).

---

## Embeddings

Default embeddings: `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface` (or community fallback).
For fully offline setups, you can keep this model local; it downloads on first run and caches.

To change embeddings:
```python
from langchain_huggingface import HuggingFaceEmbeddings
embed = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
```

Larger models improve retrieval at higher memory/latency cost.

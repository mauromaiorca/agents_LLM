# Installation

This project is pure Python and runs on **macOS** and **Windows**.
It relies on PyMuPDF for text extraction, with optional OCR and table extraction.

## 0) Requirements

- **Python 3.11+**
- Disk space ~200 MB (optional OCR/table tools add more)
- An LLM provider (OpenAI/DeepSeek) **or** a local LLM via Ollama

> You can install into the **system** environment or a **virtualenv**. Virtualenv is recommended.

---

## 1) macOS

### 1.1 Python
- If you use `pyenv`:
  ```bash
  brew install pyenv
  pyenv install 3.11.9
  pyenv global 3.11.9
  ```
- Verify:
  ```bash
  python --version
  ```

### 1.2 System tools (optional but recommended)
- **Tesseract** (OCR), **Poppler** (PDF images), **Ghostscript** (some PDFs), **Java** (for Camelot lattice engine):
  ```bash
  brew install tesseract poppler ghostscript
  ```

### 1.3 Python libraries
```bash
pip install --upgrade pip
pip install langchain langchain-openai langchain-community chromadb pymupdf pypdf python-dotenv
# optional extras:
pip install pdf2image pytesseract camelot-py pdfplumber
# (optional, for cleaner embedding warning)
pip install -U langchain-huggingface
```

> On Apple Silicon, if `pytesseract` complains, ensure Homebrew’s Tesseract is on PATH (`which tesseract`).

---
---

## 2) Windows

### 2.1 Python
- Install from <https://www.python.org/downloads/> (choose 3.11+ and check “Add Python to PATH”).

### 2.2 System tools (optional)
- **Tesseract**: install from <https://github.com/UB-Mannheim/tesseract/wiki>.
- **Poppler**: Windows builds at <https://github.com/oschwartz10612/poppler-windows/releases/>; add `bin` to PATH.
- **Ghostscript**: <https://ghostscript.com/releases/index.html> (optional).

### 2.3 Python libraries
```powershell
py -m pip install --upgrade pip
py -m pip install langchain langchain-openai langchain-community chromadb pymupdf pypdf python-dotenv
py -m pip install pdf2image pytesseract camelot-py pdfplumber
py -m pip install -U langchain-huggingface
```

If `pdf2image` needs Poppler, set the environment variable (example):
```powershell
setx POPPLER_PATH "C:\tools\poppler-24.02.0\Library\bin"
```

---

## 3) Environment (.env)

Create a `.env` file in the repo root (or the working directory you run from):

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
# Alternative providers are documented in docs/Models.md
```

Both scripts load `.env` automatically.

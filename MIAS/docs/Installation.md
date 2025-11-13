# Installation

This project is pure Python and runs on **macOS** and **Windows**.  
It relies on **PyMuPDF** for text extraction and **sentence-transformers** for embeddings, with optional OCR and table extraction.

The recommended way to install Python dependencies is via the provided **`requirements.txt`** file.

---

## 0) Requirements

- **Python 3.11+**
- Disk space ~200–500 MB (models + OCR/table tools)
- An **OpenAI API key** (current code uses the official `openai` Python client)

> You can install into the **system** Python or, preferably, into a **virtual environment**.

---

## 1) macOS

### 1.1 Python

If you use `pyenv` (recommended):

```
brew install pyenv
pyenv install 3.11.9
pyenv global 3.11.9
```

Verify:

```
python --version
which python
```

You should see:

```
Python 3.11.9
/Users/<you>/.pyenv/versions/3.11.9/bin/python
```

---

### 1.2 Create and activate a virtual environment

```
python -m venv .venv
source .venv/bin/activate
```

---

### 1.3 System tools (optional, for OCR/tables)

```
brew install tesseract poppler ghostscript
```

---

### 1.4 Python libraries

```
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2) Windows

### 2.1 Python

Download from:

https://www.python.org/downloads/

Ensure “Add Python to PATH” is checked.

---

### 2.2 Virtual environment

```
py -m venv .venv
.\.venv\Scriptsctivate
```

---

### 2.3 Optional system tools

- Tesseract  
- Poppler  
- Ghostscript  

For Poppler:

```
setx POPPLER_PATH "C:\tools\poppler-24.02.0\Library\bin"
```

---

### 2.4 Python libraries

```
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

---

## 3) Environment (.env)

Create `.env`:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

---

## 4) Run the pipeline

```
./run_pdf_relevance_pipeline.py   --pdf-dir /path/to/pdfs   --topic "topic"   --out-csv output.csv   --threshold 60   --max-pages 20
```

On Windows:

```
.
un_pdf_relevance_pipeline.py --pdf-dir "C:\path\to\pdfs" --topic "topic" --out-csv "C:\out.csv"
```

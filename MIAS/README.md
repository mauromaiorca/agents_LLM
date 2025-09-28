# Marine Invasive Alien Species (MIAS) PDF Pipeline

A small toolkit to **score relevance** and **extract structured fields** from scientific PDFs
(e.g., *marine invasive alien species* in the Mediterranean) using an LLM + lightweight RAG.
It supports **OpenAI**, **other API providers** (e.g., DeepSeek), and **local LLMs** via **Ollama**.

- 📦 Cross‑platform (macOS & Windows)
- 🧪 Reads PDFs (PyMuPDF) with optional OCR and tables
- 🧠 Two‑pass extraction: global parse + field‑by‑field RAG
- 🗂  Output CSV in `<PDF_DIR>/analysis/`
- 🔧 Easily customise fields and prompts

> **Code files (place alongside this repo):**
> - `code/pdf_relevance_pipeline.py`
> - `code/run_pdf_relevance_pipeline.py`

## Quick Start

1. **Install** (see [`docs/Installation.md`](docs/Installation.md)).
2. Put PDFs in a folder, e.g. `~/papers/med_ias_2025`.
3. Run:
   ```bash
   ./code/run_pdf_relevance_pipeline.py --pdf-dir ~/papers/med_ias_2025 --threshold 60 --max-pages 20
   ```
4. Open the CSV at: `~/papers/med_ias_2025/analysis/extractions_from_pdfs.csv`.

## Configure & Customise

- Adjust fields and prompts in [`docs/Config.md`](docs/Config.md).
- Switch models/providers (OpenAI, DeepSeek, local) in [`docs/Models.md`](docs/Models.md).

## Detailed Guides

- Usage walkthrough: [`docs/Usage.md`](docs/Usage.md)
- Advanced options (OCR, tables, speed/cost): [`docs/Advanced.md`](docs/Advanced.md)
- Troubleshooting: [`docs/Troubleshooting.md`](docs/Troubleshooting.md)

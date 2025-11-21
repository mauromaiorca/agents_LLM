# PDF Summary Agent for MIAS Papers

A small toolkit to **score relevance** and **extract structured fields** from scientific PDFs
(e.g., *marine invasive alien species* in the Mediterranean) using an LLM + lightweight RAG.
It supports **OpenAI**, **other API providers** (e.g., DeepSeek), and **local LLMs** via **Ollama**.

> **Code files (place alongside this repo):**
> - `code/pdf_relevance_pipeline.py`
> - `code/run_pdf_relevance_pipeline.py`

## Get Ready
You need to have ChatGPT API, unfortunately this might not be free.
Instructions here: [https://platform.openai.com/](https://platform.openai.com/docs/quickstart)
   
## Quick Start

In the directory where you got the code, and you are going to use it, create a file named .env that will look like:
   ```bash
OPENAI_API_KEY=YOUR-API-KEY
OPENAI_MODEL=gpt-4o-mini
   ```
and replace "YOUR-API-KEY" with your api key retrieved from OpenAI. Also, it should be possible use other LLM, but I didn't test on that. Further, here the gpt-4o-mini is used because it's cheap, see pricing here: [openAI pricing](https://platform.openai.com/docs/pricing), you can also go for a cheaper gpt-5-nano, or slightly more expensive gpt-5-mini.

To install the actual software you can see detailed information here: [`docs/Installation.md`](docs/Installation.md).

Once your code is installed, your API key settled, select a directory with the pdf you want to check (e.g. /home/mauro/test/testLLM/agents_LLM/MIAS/documents)

run the program:
   ```bash
   ./code/run_pdf_relevance_pipeline.py --pdf-dir /home/mauro/test/testLLM/agents_LLM/MIAS/documents --threshold 60 --max-pages 20
   ```


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

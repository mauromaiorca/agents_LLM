# Troubleshooting

## OpenAIError: api_key must be set
- Ensure `.env` has `OPENAI_API_KEY=...`
- The scripts call `load_dotenv(find_dotenv(usecwd=True))`. Run from the repo root or give an absolute path to `.env`.

## ModuleNotFoundError (splitter / embeddings)
- Use `from langchain.text_splitter import RecursiveCharacterTextSplitter` (already in the code).
- If you see deprecation on `HuggingFaceEmbeddings`, optionally `pip install -U langchain-huggingface` and switch imports.

## OCR errors (Windows)
- Verify Tesseract is installed and on PATH.
- For `pdf2image`, install Poppler and set `POPPLER_PATH` env var.

## Camelot table extraction
- Camelot may need Ghostscript and sometimes Java.
- If tables fail, the pipeline still runs (tables are optional).

## Few or zero relevant PDFs
- Increase `--max-pages` to scan more content.
- Lower `--threshold` to allow borderline PDFs through to extraction.
- Refine the default `DEFAULT_TOPIC` in the runner to be less strict.

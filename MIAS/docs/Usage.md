# Usage

## Basic Run

Put your PDFs in a folder, e.g.:
```
~/papers/med_ias_2025/
```

Run the pipeline:
```bash
./code/run_pdf_relevance_pipeline.py   --pdf-dir ~/papers/med_ias_2025   --threshold 60   --max-pages 20
```

Outputs:
- `~/papers/med_ias_2025/analysis/extractions_from_pdfs.csv`

Key columns include:
- `relevance` (0/1), `relevance_percentage` (0–100)
- `relevance_rationale` (why it was deemed relevant)
- Structured fields (author, year, country, metrics, estimate, …)
- `__evidence` columns per field extracted via RAG

## Parameters

- `--threshold` (default 60): PDFs with relevance ≥ threshold undergo full field extraction.
- `--max-pages` (default 12): max pages to read per PDF (set `0` to read all pages).
- `--topic`: optional override; otherwise the default topic compiled into the runner is used.
- `--out-csv`: optional custom output path.

## Interpreting Results

- If a field is unknown or not present, it is `N/A`.
- Evidence columns show short quotes supporting the extracted value.
- You should review the CSV and spot-check against the PDFs—LLMs can err on ambiguous text.

## Speed & Cost Tips

- Lower `--max-pages` for faster runs; raise only when critical details are at the end.
- Bump `--threshold` to avoid deep extraction on marginally relevant PDFs.
- Prefer a local LLM (see `docs/Models.md`) when privacy/cost is a concern.

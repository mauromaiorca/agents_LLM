# Advanced Topics

## How relevance works
- We build a temporary in‑memory vector index over the PDF text (chunked).
- We retrieve top‑k snippets against your **topic**.
- The LLM returns: `relevance` 0/1, `relevance_percentage` (0–100), and a short `rationale`.

Only PDFs above `--threshold` go to full extraction.

## Filling N/A with RAG
- After a global parse, each field with `N/A` triggers a focused query over the retrieved snippets, asking precise, constrained questions (see `FIELD_QUERIES`).
- Evidence quotes are kept next to each filled field.

## Performance tips
- Use smaller `--max-pages` for quick screening; re‑run on shortlisted PDFs with a higher limit.
- Use a local LLM for private docs to avoid network latency and cost; consider high‑quality local models for better accuracy.

## Extending the schema
1. Add your field name to `SCHEMA_FIELDS` (for CSV columns).
2. Add a corresponding entry (instructions) in `FIELD_QUERIES`.
3. (Optional) Normalise the field in the post‑processing section (`_norm_binary`, `_only_number`, etc.).

# Configuration & Customisation

## Where to change fields & prompts

All field prompts are defined in `code/pdf_relevance_pipeline.py` in the `FIELD_QUERIES` dictionary.
Each entry has **clear, constrained instructions** to force precise extraction.

Example (snippet):
```python
FIELD_QUERIES = {
    "country": (
        "Task: identify the country/countries of the study site(s).\n"
        "Rules:\n"
        "- Extract only explicit site countries; do NOT infer from author affiliations.\n"
        "..."
    ),
    "spatial coordinates": "...",
    ...
}
```

You can:
- **Add** new fields: add an entry in `FIELD_QUERIES` and also ensure the field name exists in `SCHEMA_FIELDS`.
- **Remove** fields: remove from both `FIELD_QUERIES` and `SCHEMA_FIELDS`.
- **Refine** instructions: be explicit, forbid inference, specify units and allowable tokens.

## Relevance topic

The default topic is set in `code/run_pdf_relevance_pipeline.py` (`DEFAULT_TOPIC`).
Override at runtime with `--topic "..."` or change the constant for persistent defaults.

## Output location

By default, the runner writes to `<PDF_DIR>/analysis/extractions_from_pdfs.csv`.
Pass `--out-csv` to override.

## Evidence columns

For each field in `FIELD_QUERIES`, the pipeline creates an extra `{field}__evidence` column with a supporting quote.

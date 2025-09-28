#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Runner per pdf_relevance_pipeline.py
- Topic di default (non serve passarlo ogni volta)
- Output automatico in <PDF_DIR>/analysis/extractions_from_pdfs.csv
- Caricamento .env
"""

import argparse
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))  # assicura che .env sia caricata

from pdf_relevance_pipeline import process_directory

# Topic di default (modificalo qui una sola volta)
DEFAULT_TOPIC = (
    "Marine invasive alien species in the Mediterranean; "
    "economic valuation (WTP/WTA/TCM/hedonic/prices/costs); "
    "study characteristics; 2025"
)

def main():
    ap = argparse.ArgumentParser(description="Run PDF relevance+extraction pipeline")
    ap.add_argument("--pdf-dir", required=True, help="Directory con i PDF da analizzare")
    ap.add_argument("--out-csv", help="CSV di output; default: <pdf-dir>/analysis/extractions_from_pdfs.csv")
    ap.add_argument("--topic", help="Descrizione del TOPIC; default preimpostato")
    ap.add_argument("--threshold", type=int, default=60, help="Soglia % per considerare un PDF 'rilevante'")
    ap.add_argument("--max-pages", type=int, default=12, help="Numero massimo di pagine da leggere per PDF (0=tutte)")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir).expanduser().resolve()
    topic = args.topic if args.topic else DEFAULT_TOPIC
    out_csv = Path(args.out_csv) if args.out_csv else (pdf_dir / "analysis" / "extractions_from_pdfs.csv")

    max_pages = None if args.max_pages <= 0 else args.max_pages
    n_all, n_rel = process_directory(
        pdf_dir,
        topic,
        out_csv,
        relevance_threshold=args.threshold,
        max_pages=max_pages
    )
    print(f"\n[RISULTATO]\n- PDF totali: {n_all}\n- Rilevanti ≥ soglia: {n_rel}\n- Output: {out_csv}")

if __name__ == "__main__":
    main()
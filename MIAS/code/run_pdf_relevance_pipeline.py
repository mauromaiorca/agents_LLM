#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI wrapper for the PDF relevance + extraction pipeline.

This script is intentionally thin: it wires together
  - a YAML profile (e.g. marine_valuation.yml)
  - a PDF directory
  - output CSV path
  - relevance threshold and page limit

All prompts, schema fields, and per-field questions live in the YAML profile.
"""

import argparse
import os
from pathlib import Path

from pdf_relevance_pipeline import load_profile, process_directory, text_convert_directory, grobid_extract_all

DEFAULT_CONFIG = "marine_valuation.yml"


def main() -> None:
    ap = argparse.ArgumentParser(description="PDF relevance & extraction (profile-based)")
    ap.add_argument(
        "--pdf-dir",
        required=True,
        help="Directory containing PDF files",
    )
    ap.add_argument(
        "--out-csv",
        help="Output CSV path (default: <PDF_DIR>/analysis/analysis_<dir>.csv)",
    )
    ap.add_argument(
        "--threshold",
        type=int,
        default=60,
        help="Relevance percentage threshold to consider a PDF 'relevant' (default: 60)",
    )
    ap.add_argument(
        "--max-pages",
        type=int,
        default=20,
        help="Maximum pages to read per PDF (0 = all pages, default: 20)",
    )
    ap.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"YAML profile file (default: {DEFAULT_CONFIG})",
    )
    ap.add_argument(
        "--topic",
        default=None,
        help="Override TOPIC for relevance scoring (default: topic from YAML profile)",
    )
    ap.add_argument(
        "--actions-csv",
        default=None,
        help="Optional CSV with per-file actions/instructions (from refinement agent).",
    )
    ap.add_argument(
        "--text-convert",
        default=None,
        metavar="DEBUG_DIR",
        help=(
            "If set, the script does NOT run the LLM pipeline, but only converts PDFs "
            "to text and saves .txt files into DEBUG_DIR."
        ),
    )
    ap.add_argument(
        "--docs-type",
        choices=["generic", "paper"],
        default="generic",
        help=(
            "Type of documents to process: 'generic' uses the built-in PDF text extractor; "
            "'paper' uses a GROBID server to parse scholarly articles."
        ),
    )
    ap.add_argument(
        "--grobid-url",
        default=None,
        help=(
            "Base URL of the GROBID server (only used if --docs-type paper). "
            "If not set, falls back to the GROBID_URL environment variable or http://localhost:8070."
        ),
    )


    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.is_dir():
        raise SystemExit(f"[ERROR] PDF directory does not exist or is not a directory: {pdf_dir}")

    config_path = Path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"[ERROR] Config file not found: {config_path}")

    profile = load_profile(config_path)

    # Topic for relevance scoring: CLI override > profile topic > generic fallback
    topic = args.topic or profile.get("topic") or "General PDF relevance topic"

    if args.out_csv:
        out_csv = Path(args.out_csv)
    else:
        out_csv = pdf_dir / "analysis" / f"analysis_{pdf_dir.name}.csv"

    max_pages = None if args.max_pages <= 0 else args.max_pages

    # Optional per-file actions CSV
    actions_csv = None
    if args.actions_csv:
        actions_csv_path = Path(args.actions_csv)
        if actions_csv_path.is_file():
            actions_csv = actions_csv_path
        else:
            print(f"[WARN] actions-csv file not found, ignoring: {actions_csv_path}")

    # Modalità solo conversione testo per debug
    if args.text_convert is not None:
        debug_dir = Path(args.text_convert)
        
        grobid_url = None
        if args.docs_type == "paper":
            grobid_url = args.grobid_url or os.getenv("GROBID_URL") or "http://localhost:8070"
            print(f"[INFO] text-convert + docs-type=paper, usando GROBID su: {grobid_url}")
            
        text_convert_directory(
            pdf_dir=pdf_dir,
            out_dir=debug_dir,
            docs_type=args.docs_type,
            grobid_url=grobid_url,
            max_pages=max_pages,
            use_ocr_fallback=True,
            with_tables=True,
        )
        print(f"[DONE] Text conversion only. TXT files written to: {debug_dir}")
        return

    # Configurazione GROBID solo se docs_type == "paper"
    grobid_url = None
    if args.docs_type == "paper":
        grobid_url = args.grobid_url or os.getenv("GROBID_URL") or "http://localhost:8070"
        print(f"[INFO] docs-type=paper, using GROBID at: {grobid_url}")

    n_all, n_rel = process_directory(
        pdf_dir=pdf_dir,
        topic=topic,
        out_csv=out_csv,
        profile=profile,
        relevance_threshold=args.threshold,
        max_pages=max_pages,
        actions_csv=actions_csv,
    )

    print(f"[PROFILE] {profile.get('name', 'unnamed')} | TOPIC: {topic}")
    print(f"[DONE] Processed: {n_all}, relevant ≥ {args.threshold}%: {n_rel} -> {out_csv}")



if __name__ == "__main__":
    main()

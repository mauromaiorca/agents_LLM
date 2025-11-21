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
from pathlib import Path

from pdf_relevance_pipeline import load_profile, process_directory

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

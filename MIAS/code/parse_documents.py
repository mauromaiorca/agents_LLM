#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
parse_documents.py

Pipeline per:
  1) Parsare PDF con GROBID in JSON (docs-type=paper)
  2) Arricchire i JSON con annotazioni LLM:
       - livello documento (llm_enrichment)
       - livello sezione/paragrafo (section_enrichment)

Riusa le utility implementate in pdf_relevance_pipeline.py
senza modificarle.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from langchain.schema import SystemMessage, HumanMessage  # type: ignore

from pdf_relevance_pipeline import (
    grobid_extract_all,
    clean_metadata,
    load_profile,
    score_relevance,
    extract_global,
    fill_na_with_rag,
    make_llm,
)

# Limite massimo di caratteri per sezione passati al modello
MAX_SECTION_CHARS = 4000  # puoi ridurre/aumentare a piacere

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg, flush=True)


def run_grobid_papers(
    pdf_dir: Path,
    out_json_dir: Path,
    grobid_url: str,
    max_pages: Optional[int] = None,
) -> None:
    """
    Parsare tutti i PDF in pdf_dir con GROBID e scrivere un JSON per paper in out_json_dir.
    """
    out_json_dir.mkdir(parents=True, exist_ok=True)
    pdf_list = sorted(pdf_dir.glob("*.pdf"))

    _log(f"[INFO] GROBID URL: {grobid_url}")
    _log(f"[INFO] Parsing di {len(pdf_list)} PDF da {pdf_dir.name} verso {out_json_dir.name}")

    if not pdf_list:
        _log("[WARN] Nessun PDF trovato, nulla da fare.")
        return

    for i, pdf_path in enumerate(pdf_list, start=1):
        _log(f"[GROBID] [{i}/{len(pdf_list)}] {pdf_path.name} ...")
        try:
            meta = grobid_extract_all(
                pdf_path=pdf_path,
                grobid_url=grobid_url,
                max_pages=max_pages,
                with_tables=True,
            )
        except Exception as e:
            _log(f"[GROBID][ERRORE] {pdf_path.name}: {e}")
            meta = None

        if not meta:
            meta = {
                "pdf_filename": pdf_path.name,
                "error": "grobid_extract_all returned empty or failed",
            }
        else:
            meta = clean_metadata(meta)

        json_path = out_json_dir / (pdf_path.stem + ".json")
        json_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _log(f"[GROBID] Salvato JSON: {json_path}")


def _build_full_text(meta: Dict[str, Any]) -> str:
    """
    Costruisce un testo unico a partire dal JSON di un articolo.
    Preferisce 'fulltext' se presente, altrimenti abstract + sections.
    """
    fulltext = (meta.get("fulltext") or "").strip()
    if fulltext:
        return fulltext

    parts: List[str] = []
    abstract = (meta.get("abstract") or "").strip()
    if abstract:
        parts.append(abstract)

    for sec in meta.get("sections") or []:
        txt = (sec.get("text") or "").strip()
        if txt:
            parts.append(txt)

    return "\n\n".join(parts)


def _safe_json_from_llm(content: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prova a parsare JSON da una risposta LLM; se fallisce restituisce fallback.
    """
    content = (content or "").strip()
    if not content:
        return dict(fallback)

    try:
        m = re.search(r"\{.*\}", content, flags=re.S)
        data = json.loads(m.group(0) if m else content)
        if isinstance(data, dict):
            return data
        return dict(fallback)
    except Exception:
        return dict(fallback)


def enrich_json_documents(
    json_dir: Path,
    enrich_yaml: Path,
    relevance_threshold: int = 50,
) -> None:
    """
    Arricchisce tutti i JSON in json_dir usando il profilo in enrich_yaml.

    Aggiunge a ciascun JSON:
      - 'llm_enrichment'      : enrichment a livello di articolo (un dict)
      - 'section_enrichment'  : lista di enrichment per sezione/paragrafo
    """
    if not json_dir.is_dir():
        raise SystemExit(f"[ERROR] JSON directory non trovata: {json_dir}")

    if not enrich_yaml.is_file():
        raise SystemExit(f"[ERROR] File YAML per enrichment non trovato: {enrich_yaml}")

    # YAML grezzo (ci serve per section_enrichment)
    raw_cfg = yaml.safe_load(enrich_yaml.read_text(encoding="utf-8"))
    if not isinstance(raw_cfg, dict):
        raise SystemExit("[ERROR] Il file YAML di enrichment deve essere un mapping.")

    # Profilo normalizzato per pdf_relevance_pipeline
    profile = load_profile(enrich_yaml)
    topic = profile.get("topic") or raw_cfg.get("topic") or "Generic scientific article"

    section_enrich_cfg = raw_cfg.get("section_enrichment") or {}
    section_enrich_prompt = (section_enrich_cfg.get("system_prompt") or "").strip()
    have_section_enrichment = bool(section_enrich_prompt)

    _log(f"[ENRICH] Profilo: {profile.get('name')} | TOPIC: {topic}")

    json_files = sorted(json_dir.glob("*.json"))
    _log(f"[ENRICH] Numero di documenti da arricchire: {len(json_files)}")

    llm_for_sections = make_llm() if have_section_enrichment else None

    for idx, json_path in enumerate(json_files, start=1):
        _log(f"[ENRICH] [{idx}/{len(json_files)}] {json_path.name}")

        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            _log(f"[ENRICH][WARN] Impossibile leggere JSON {json_path}: {e}")
            continue

        pdf_filename = meta.get("pdf_filename") or (json_path.stem + ".pdf")
        title = (meta.get("title") or "").strip()
        full_text = _build_full_text(meta)

        if not full_text.strip():
            _log(f"[ENRICH][WARN] Testo vuoto per {pdf_filename}, salto arricchimento.")
            continue

        # --- Relevance ---
        rel = score_relevance(topic=topic, text=full_text, profile=profile)
        rel_pct = int(rel.get("relevance_percentage", 0))
        relevance_flag = 1 if rel_pct >= relevance_threshold else 0
        rel_rationale = (rel.get("rationale") or "").strip()

        # --- Estrazione globale + RAG refinement ---
        global_record = extract_global(title=title, text=full_text, profile=profile)
        if not isinstance(global_record, dict):
            global_record = {}
        global_record = fill_na_with_rag(global_record, full_text, profile)

        # Document-level enrichment
        llm_enrichment: Dict[str, Any] = {
            "pdf_filename": pdf_filename,
            "relevance_percentage": rel_pct,
            "relevance": relevance_flag,
            "relevance_rationale": rel_rationale,
        }
        for k, v in global_record.items():
            if k not in llm_enrichment:
                llm_enrichment[k] = v

        meta["llm_enrichment"] = llm_enrichment

        # --- Section / paragraph-level enrichment ---
        # --- Section / paragraph-level enrichment ---
        if have_section_enrichment and llm_for_sections is not None:
            sections = meta.get("sections") or []
            sec_records: List[Dict[str, Any]] = []

            for s_idx, sec in enumerate(sections):
                sec_text = (sec.get("text") or "").strip()
                sec_title = (sec.get("title") or "").strip()
                if not sec_text:
                    continue

                # taglio la sezione per limitare i token
                truncated_text = sec_text[:MAX_SECTION_CHARS]

                # per velocità: nessun abstract, nessun pdf_filename, solo sezione
                user_content = (
                    f"SECTION_INDEX: {s_idx}\n"
                    f"SECTION_TITLE: {sec_title or 'N/A'}\n\n"
                    f"SECTION_TEXT:\n{truncated_text}"
                )

                fallback_section: Dict[str, Any] = {
                    "section_role": "N/A",
                    "section_main_research_question": "N/A",
                    "section_secondary_questions": [],
                    "section_hypotheses_or_assumptions": "N/A",
                    "section_key_ideas": "N/A",
                    "section_methods_summary": "N/A",
                    "section_results_summary": "N/A",
                    "section_limitations_or_caveats": "N/A",
                    "section_general_keywords": "N/A",
                    "section_methods_keywords": "N/A",
                    "section_results_keywords": "N/A",
                    "section_qa_local": [],
                }

                try:
                    msg = llm_for_sections(
                        [
                            SystemMessage(content=section_enrich_prompt),
                            HumanMessage(content=user_content),
                        ]
                    )
                    content = getattr(msg, "content", "").strip()
                    sec_data = _safe_json_from_llm(content, fallback_section)
                except Exception as e:
                    _log(f"[ENRICH][WARN] Section-level enrichment failed for {pdf_filename} sec {s_idx}: {e}")
                    sec_data = dict(fallback_section)

                def _ensure_list(x: Any) -> List[Any]:
                    if isinstance(x, list):
                        return x
                    if x in (None, "", "N/A"):
                        return []
                    return [x]

                sec_data["section_secondary_questions"] = _ensure_list(
                    sec_data.get("section_secondary_questions")
                )
                sec_data["section_qa_local"] = _ensure_list(sec_data.get("section_qa_local"))

                rec: Dict[str, Any] = {
                    "section_index": s_idx,
                    "section_id": sec.get("id") or "",
                    "section_type": sec.get("type") or "",
                    "section_title": sec_title,
                    "section_n": sec.get("n") or "",
                }
                for k, v in sec_data.items():
                    if k not in rec:
                        rec[k] = v

                sec_records.append(rec)

            meta["section_enrichment"] = sec_records


        # --- Salvataggio ---
        json_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _log(f"[ENRICH] Salvato JSON arricchito: {json_path.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse PDF con GROBID e arricchisci i JSON con metadata LLM (documento + sezioni)."
    )
    ap.add_argument(
        "--docs-dir",
        required=True,
        help="Directory con i documenti in ingresso (PDF o JSON).",
    )
    ap.add_argument(
        "--docs-type",
        choices=["paper", "json"],
        default="paper",
        help="Tipo di documenti in --docs-dir. 'paper' = PDF da parsare con GROBID; 'json' = JSON già parsati.",
    )
    ap.add_argument(
        "--grobid-url",
        default="http://localhost:8070",
        help="URL base del server GROBID (usato solo se --docs-type paper).",
    )
    ap.add_argument(
        "--out-json-dir",
        default=None,
        help="Directory di output per i JSON. Se omesso e docs-type=paper, usa DOCS-DIR + '_json'.",
    )
    ap.add_argument(
        "--enrich",
        dest="enrich_yaml",
        default=None,
        help="File YAML di profilo per l'arricchimento LLM (documento + sezione). Se omesso, fa solo il parsing GROBID.",
    )
    ap.add_argument(
        "--relevance-threshold",
        type=int,
        default=50,
        help="Soglia su relevance_percentage per impostare la variabile binaria 'relevance' (default: 50).",
    )

    args = ap.parse_args()

    docs_dir = Path(args.docs_dir)
    if not docs_dir.is_dir():
        raise SystemExit(f"[ERROR] docs-dir non esiste o non è una directory: {docs_dir}")

    # Decidi la directory JSON
    if args.docs_type == "paper":
        if args.out_json_dir:
            json_dir = Path(args.out_json_dir)
        else:
            json_dir = docs_dir.parent / f"{docs_dir.name}_json"

        run_grobid_papers(
            pdf_dir=docs_dir,
            out_json_dir=json_dir,
            grobid_url=args.grobid_url,
            max_pages=None,
        )
    else:
        # docs-type = json: docs-dir contiene già i JSON
        json_dir = docs_dir

    # Enrichment opzionale
    if args.enrich_yaml:
        enrich_json_documents(
            json_dir=json_dir,
            enrich_yaml=Path(args.enrich_yaml),
            relevance_threshold=args.relevance_threshold,
        )
    else:
        _log("[INFO] Nessun file YAML di enrichment specificato: parsing GROBID completato.")

if __name__ == "__main__":
    main()


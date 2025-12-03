#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
parse_documents.py

Pipeline per:
  1) Parsare PDF con GROBID in JSON "grezzi" (docs-type=paper)
  2) Arricchire i JSON con annotazioni LLM:
       - livello documento (llm_enrichment)
       - livello sezione/paragrafo (section_enrichment)
     salvando gli enrichment in file separati:
       - <stem>_metadata.json        (solo enrichment)
       - <stem>_keywords.xml         (albero di keywords)

Inoltre crea un file di indice globale:
  metadata_index.json

Riusa le utility implementate in pdf_relevance_pipeline.py
senza modificarle.
"""

import argparse
import json
import re
import csv
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
    Parsare tutti i PDF in pdf_dir con GROBID e scrivere un JSON "grezzo" per paper in out_json_dir.
    (Nessun arricchimento LLM; solo estrazione GROBID + clean_metadata.)
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
        _log(f"[GROBID] Salvato JSON grezzo: {json_path}")


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


def _ensure_list(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    if x in (None, "", "N/A"):
        return []
    return [x]


def _extract_keyword_records(value: Any) -> List[Dict[str, str]]:
    """
    Convert a 'keywords' field (various formats) into a list of records with:
      - keyword
      - relevance
      - macro_area
      - specific_area
      - scientific_role
      - study_object
      - research_question

    Supported input formats:
      - JSON-encoded string: list of objects with keys like
            keyword, relevance_score, macro_area, specific_area,
            scientific_role, study_object, research_question
      - list of such dicts
      - plain strings (separated by ; , | or newline), without metadata
    """
    records: List[Dict[str, str]] = []

    def _add(
        kw: Any,
        relevance: Any = "",
        macro_area: Any = "",
        specific_area: Any = "",
        scientific_role: Any = "",
        study_object: Any = "",
        research_question: Any = "",
    ) -> None:
        kw_str = str(kw).strip()
        if not kw_str:
            return
        rec: Dict[str, str] = {
            "keyword": kw_str,
            "relevance": str(relevance) if relevance is not None else "",
            "macro_area": str(macro_area) if macro_area is not None else "",
            "specific_area": str(specific_area) if specific_area is not None else "",
            "scientific_role": str(scientific_role) if scientific_role is not None else "",
            "study_object": str(study_object) if study_object is not None else "",
            "research_question": str(research_question) if research_question is not None else "",
        }
        records.append(rec)

    if value is None:
        return records

    # If it is a string, first try json.loads
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return records
        try:
            parsed = json.loads(s)
            value = parsed
        except Exception:
            # Not JSON -> treat as a list of simple keyword strings
            parts = re.split(r"[;,|\n]", s)
            for p in parts:
                p = p.strip()
                if p:
                    _add(p)
            return records

    # If it is a list, recurse on each item
    if isinstance(value, list):
        for item in value:
            records.extend(_extract_keyword_records(item))
        return records

    # Single dict: extract all known fields
    if isinstance(value, dict):
        kw = value.get("keyword") or value.get("term") or ""
        relevance = value.get("relevance_score", value.get("relevance", ""))
        macro_area = value.get("macro_area", "")
        specific_area = value.get("specific_area", "")
        scientific_role = value.get("scientific_role", "")
        study_object = value.get("study_object", "")
        research_question = value.get("research_question", "")
        _add(
            kw,
            relevance,
            macro_area,
            specific_area,
            scientific_role,
            study_object,
            research_question,
        )
        return records

    # Generic fallback
    _add(str(value))
    return records


def _extract_keywords_from_metadata(metadata: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Estrae tutti i record keyword/relevance/macro_area/specific_area dal
    metadata JSON (<stem>_metadata.json), sia a livello documento che sezione.
    """
    all_records: List[Dict[str, str]] = []

    # --- livello documento ---
    llm_enr = metadata.get("llm_enrichment") or {}
    if isinstance(llm_enr, dict):
        for k, v in llm_enr.items():
            if "keyword" not in k.lower():
                continue
            all_records.extend(_extract_keyword_records(v))

    # --- livello sezione ---
    sections = metadata.get("section_enrichment") or []
    if isinstance(sections, list):
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            for k, v in sec.items():
                if "keyword" not in k.lower():
                    continue
                all_records.extend(_extract_keyword_records(v))

    return all_records


def _save_keywords_csv_from_metadata_json(
    metadata_json_path: Path,
    keywords_csv_path: Path,
) -> None:
    """
    Load <stem>_metadata.json, extract keywords (with relevance, discipline,
    and semantic fields), and save as CSV:

      keyword,relevance,macro_area,specific_area,scientific_role,study_object,research_question
      Gaussian functions,70,mathematics,statistics,method,signals,resolution improvement
      image embedding,90,computer science,image processing,method,images,feature extraction
      ...
    """
    try:
        data = json.loads(metadata_json_path.read_text(encoding="utf-8"))
    except Exception as e:
        _log(f"[KEYWORDS][WARN] Cannot read metadata JSON {metadata_json_path}: {e}")
        return

    records = _extract_keywords_from_metadata(data)

    try:
        with keywords_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            # header with the new fields
            writer.writerow(
                [
                    "keyword",
                    "relevance",
                    "macro_area",
                    "specific_area",
                    "scientific_role",
                    "study_object",
                    "research_question",
                ]
            )
            # rows
            for rec in records:
                writer.writerow(
                    [
                        rec.get("keyword", ""),
                        rec.get("relevance", ""),
                        rec.get("macro_area", ""),
                        rec.get("specific_area", ""),
                        rec.get("scientific_role", ""),
                        rec.get("study_object", ""),
                        rec.get("research_question", ""),
                    ]
                )
        _log(f"[KEYWORDS] Saved CSV keywords: {keywords_csv_path.name}")
    except Exception as e:
        _log(f"[KEYWORDS][WARN] Cannot write CSV {keywords_csv_path}: {e}")


def enrich_json_documents(
    json_dir: Path,
    enrich_yaml: Path,
    relevance_threshold: int = 50,
    raw_docs_dir: Optional[Path] = None,
) -> None:
    """
    Arricchisce tutti i JSON grezzi in json_dir usando il profilo in enrich_yaml.

    Invece di modificare il JSON originale, crea per ogni documento:

      - <stem>_metadata.json : enrichment a livello di articolo + sezione
      - <stem>_keywords.xml  : keywords estratte dal metadata_json

    Inoltre crea un indice globale:

      - metadata_index.json : lista di record con mapping dei file
    """
    if not json_dir.is_dir():
        raise SystemExit(f"[ERROR] JSON directory non trovata: {json_dir}")

    if not enrich_yaml.is_file():
        raise SystemExit(f"[ERROR] File YAML per enrichment non trovato: {enrich_yaml}")

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

    # indice globale
    index_records: List[Dict[str, Any]] = []

    raw_dir_str = ""
    if raw_docs_dir is not None:
        # aggiungo trailing slash per coerenza con l'esempio
        r = str(raw_docs_dir)
        raw_dir_str = r if r.endswith("/") else r + "/"

    meta_dir_str = str(json_dir)
    meta_dir_str = meta_dir_str if meta_dir_str.endswith("/") else meta_dir_str + "/"

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

        # --- Section / paragraph-level enrichment ---
        sec_records: List[Dict[str, Any]] = []
        if have_section_enrichment and llm_for_sections is not None:
            sections = meta.get("sections") or []

            for s_idx, sec in enumerate(sections):
                sec_text = (sec.get("text") or "").strip()
                sec_title = (sec.get("title") or "").strip()
                if not sec_text:
                    continue

                truncated_text = sec_text[:MAX_SECTION_CHARS]

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

        # --- Costruzione del record di metadata (enrichment) separato ---
        metadata_record: Dict[str, Any] = {
            "pdf_filename": pdf_filename,
            "title": title,
            "llm_enrichment": llm_enrichment,
        }
        if sec_records:
            metadata_record["section_enrichment"] = sec_records

        # path dei file di output per questo documento
        stem = json_path.stem
        raw_text_json_filename = f"{stem}.json"
        metadata_json_filename = f"{stem}_metadata.json"
        keywords_filename = f"{stem}_keywords.csv"

        metadata_json_path = json_dir / metadata_json_filename
        keywords_csv_path = json_dir / keywords_filename

        # salva metadata JSON
        metadata_json_path.write_text(
            json.dumps(metadata_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _log(f"[ENRICH] Salvato JSON metadata: {metadata_json_path.name}")

        # genera e salva CSV di keywords a partire dal metadata JSON
        _save_keywords_csv_from_metadata_json(metadata_json_path, keywords_csv_path)

        # aggiungi record all'indice globale
        index_record = {
            "raw_directory": raw_dir_str,
            "metadataDirectory": meta_dir_str,
            "raw_filename": pdf_filename,
            "raw_text_json_filename": raw_text_json_filename,
            "metadata_json_filename": metadata_json_filename,
            "keywords_file": keywords_filename,
        }

        index_records.append(index_record)

    # --- Salvataggio indice globale ---
    if index_records:
        index_path = json_dir / "metadata_index.json"
        index_path.write_text(
            json.dumps(index_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _log(f"[INDEX] Salvato indice globale: {index_path}")
    else:
        _log("[INDEX] Nessun record di metadata generato; indice non creato.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse PDF con GROBID e arricchisci i JSON con metadata LLM (documento + sezioni) salvando i risultati in file separati."
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

    # Decidi la directory JSON (metadataDirectory) e la raw_directory da mettere nell'indice
    if args.docs_type == "paper":
        if args.out_json_dir:
            json_dir = Path(args.out_json_dir)
        else:
            json_dir = docs_dir.parent / f"{docs_dir.name}_json"

        # parsing GROBID -> JSON grezzi
        run_grobid_papers(
            pdf_dir=docs_dir,
            out_json_dir=json_dir,
            grobid_url=args.grobid_url,
            max_pages=None,
        )
        raw_docs_dir = docs_dir
    else:
        # docs-type = json: docs-dir contiene già i JSON grezzi
        json_dir = docs_dir
        raw_docs_dir = docs_dir  # in questo caso raw_directory coincide con json_dir

    # Enrichment opzionale: genera <stem>_metadata.json, <stem]_keywords.xml e metadata_index.json
    if args.enrich_yaml:
        enrich_json_documents(
            json_dir=json_dir,
            enrich_yaml=Path(args.enrich_yaml),
            relevance_threshold=args.relevance_threshold,
            raw_docs_dir=raw_docs_dir,
        )
    else:
        _log("[INFO] Nessun file YAML di enrichment specificato: parsing GROBID completato.")

if __name__ == "__main__":
    main()


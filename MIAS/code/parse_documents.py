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
    OPENAI_MODEL,
    read_pdf_text,
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


def run_plain_papers(
    pdf_dir: Path,
    out_json_dir: Path,
    max_pages: Optional[int] = None,
) -> None:
    """
    Parsare tutti i PDF in pdf_dir SENZA GROBID e scrivere un JSON "grezzo" per paper
    in out_json_dir usando una semplice estrazione di testo da PDF.

    Struttura del JSON per ogni PDF:

      {
        "pdf_filename": "<nome.pdf>",
        "title": "",
        "abstract": "",
        "fulltext": "<testo completo>",
        "sections": [
          {
            "id": "body",
            "type": "body",
            "n": "",
            "title": "",
            "text": "<testo completo>"
          }
        ]
      }

    Questo è compatibile con _build_full_text e con la logica di section_enrichment
    (otterrai una singola "sezione" con il testo integrale).
    """
    out_json_dir.mkdir(parents=True, exist_ok=True)
    pdf_list = sorted(pdf_dir.glob("*.pdf"))

    _log(
        f"[INFO] Parsing plain-text di {len(pdf_list)} PDF da "
        f"{pdf_dir.name} verso {out_json_dir.name} (senza GROBID)."
    )

    for pdf_path in pdf_list:
        try:
            text = read_pdf_text(
                path=pdf_path,
                max_pages=max_pages,
                use_ocr_fallback=True,
                with_tables=True,
            )
        except Exception as e:
            _log(f"[PLAIN][WARN] Impossibile leggere testo da {pdf_path.name}: {e}")
            text = ""

        meta: Dict[str, Any] = {
            "pdf_filename": pdf_path.name,
            "title": "",
            "abstract": "",
            "fulltext": text,
            "sections": [
                {
                    "id": "body",
                    "type": "body",
                    "n": "",
                    "title": "",
                    "text": text,
                }
            ],
        }

        json_path = out_json_dir / (pdf_path.stem + ".json")
        json_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _log(f"[PLAIN] Salvato JSON grezzo (no GROBID): {json_path}")


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


def _save_keywords_json_and_csv_from_metadata_json(
    metadata_json_path: Path,
    global_json_path: Path,
    global_csv_path: Path,
    specific_json_path: Path,
    specific_csv_path: Path,
    do_global: bool,
    do_specific: bool,
) -> Dict[str, bool]:
    """
    From <stem>_metadata.json build four files:

      - <stem>_keywords_global.json
      - <stem>_keywords_global.csv
      - <stem>_keywords_specific.json
      - <stem>_keywords_specific.csv

    Global = document-level keyword lists in llm_enrichment.
    Specific = section-level keyword lists in section_enrichment.

    CSVs contain ONLY keyword-level fields:
      keyword, relevance, macro_area, specific_area,
      scientific_role, study_object, research_question
    """

    written = {"global": False, "specific": False}

    try:
        data = json.loads(metadata_json_path.read_text(encoding="utf-8"))
    except Exception as e:
        _log(f"[KEYWORDS][WARN] Cannot read metadata JSON {metadata_json_path}: {e}")
        return written

    pdf_filename = data.get("pdf_filename") or "N/A"
    title = (data.get("title") or "").strip()

    # ----------------------- GLOBAL -----------------------
    if do_global:
        llm_enr = data.get("llm_enrichment") or {}
        if not isinstance(llm_enr, dict):
            llm_enr = {}

        # JSON payload
        global_payload: Dict[str, Any] = {
            "pdf_filename": pdf_filename,
            "title": title,
            "general_keywords_scored": llm_enr.get("general_keywords_scored") or [],
            "methods_keywords_scored": llm_enr.get("methods_keywords_scored") or [],
            "results_keywords_scored": llm_enr.get("results_keywords_scored") or [],
        }

        try:
            global_json_path.write_text(
                json.dumps(global_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            _log(f"[KEYWORDS][WARN] Cannot write GLOBAL keywords JSON {global_json_path}: {e}")
        else:
            # CSV payload: flatten all keyword lists, NO group/section columns
            rows: List[List[Any]] = []

            def _add_rows(kw_list: Any) -> None:
                if not isinstance(kw_list, list):
                    return
                for rec in kw_list:
                    if not isinstance(rec, dict):
                        continue
                    rows.append(
                        [
                            rec.get("keyword", ""),
                            rec.get("relevance_score", rec.get("relevance", "")),
                            rec.get("macro_area", ""),
                            rec.get("specific_area", ""),
                            rec.get("scientific_role", ""),
                            rec.get("study_object", ""),
                            rec.get("research_question", ""),
                        ]
                    )

            _add_rows(global_payload["general_keywords_scored"])
            _add_rows(global_payload["methods_keywords_scored"])
            _add_rows(global_payload["results_keywords_scored"])

            try:
                with global_csv_path.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
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
                    writer.writerows(rows)
                _log(f"[KEYWORDS] Saved GLOBAL keywords JSON and CSV for {pdf_filename}")
                written["global"] = True
            except Exception as e:
                _log(f"[KEYWORDS][WARN] Cannot write GLOBAL keywords CSV {global_csv_path}: {e}")

    # --------------------- SPECIFIC -----------------------
    if do_specific:
        sections_in = data.get("section_enrichment") or []
        sections_out: List[Dict[str, Any]] = []
        csv_rows: List[List[Any]] = []

        if isinstance(sections_in, list):
            for sec in sections_in:
                if not isinstance(sec, dict):
                    continue

                sec_index = sec.get("section_index")
                sec_id = sec.get("section_id") or ""
                sec_type = sec.get("section_type") or ""
                sec_title = sec.get("section_title") or ""
                sec_n = sec.get("section_n") or ""

                # JSON structure (full section metadata preserved in JSON)
                sections_out.append(
                    {
                        "section_index": sec_index,
                        "section_id": sec_id,
                        "section_type": sec_type,
                        "section_title": sec_title,
                        "section_n": sec_n,
                        "section_general_keywords": sec.get("section_general_keywords") or [],
                        "section_methods_keywords": sec.get("section_methods_keywords") or [],
                        "section_results_keywords": sec.get("section_results_keywords") or [],
                    }
                )

                # CSV rows: flatten keywords, NO section / group columns
                def _add_section_rows(kw_list: Any) -> None:
                    if not isinstance(kw_list, list):
                        return
                    for rec in kw_list:
                        if not isinstance(rec, dict):
                            continue
                        csv_rows.append(
                            [
                                rec.get("keyword", ""),
                                rec.get("relevance_score", rec.get("relevance", "")),
                                rec.get("macro_area", ""),
                                rec.get("specific_area", ""),
                                rec.get("scientific_role", ""),
                                rec.get("study_object", ""),
                                rec.get("research_question", ""),
                            ]
                        )

                _add_section_rows(sec.get("section_general_keywords"))
                _add_section_rows(sec.get("section_methods_keywords"))
                _add_section_rows(sec.get("section_results_keywords"))

        specific_payload: Dict[str, Any] = {
            "pdf_filename": pdf_filename,
            "title": title,
            "sections": sections_out,
        }

        try:
            specific_json_path.write_text(
                json.dumps(specific_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            _log(f"[KEYWORDS][WARN] Cannot write SPECIFIC keywords JSON {specific_json_path}: {e}")
        else:
            try:
                with specific_csv_path.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
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
                    writer.writerows(csv_rows)
                _log(f"[KEYWORDS] Saved SPECIFIC keywords JSON and CSV for {pdf_filename}")
                written["specific"] = True
            except Exception as e:
                _log(f"[KEYWORDS][WARN] Cannot write SPECIFIC keywords CSV {specific_csv_path}: {e}")

    return written


def enrich_json_documents(
    json_dir: Path,
    enrich_yaml: Path,
    relevance_threshold: int = 50,
    raw_docs_dir: Optional[Path] = None,
    index_path: Optional[Path] = None,
    no_global: bool = False,
    no_specific: bool = False,
    no_recompute: bool = False,
) -> None:
    """
    Arricchisce tutti i JSON grezzi in json_dir usando il profilo in enrich_yaml.

    Per ogni documento "stem" in json_dir (escludendo *_metadata.json, *_keywords_*.json
    ed eventuali file di indice), crea:

      - <stem>_metadata.json
          riepilogo di articolo con:
            - pdf_filename, title
            - author, journal, type of study, year, doi
            - llm_model
            - llm_enrichment (record completo globale)
            - section_enrichment (se abilitato)

      - <stem>_keywords_global.json
      - <stem>_keywords_global.csv
      - <stem>_keywords_specific.json
      - <stem>_keywords_specific.csv

    Inoltre crea (o sovrascrive) un indice globale (index_path se fornito,
    altrimenti json_dir/metadata_index.json) con, per ciascun documento:

      - raw_directory
      - metadataDirectory
      - raw_filename
      - raw_text_json_filename
      - metadata_json_filename
      - keywords_global_file, keywords_global_csv
      - keywords_specific_file, keywords_specific_csv
      - doi
      - llm_model
      - title, author, journal, type of study, year
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

    do_global = not no_global
    do_specific = have_section_enrichment and not no_specific

    _log(
        f"[ENRICH] Profilo: {profile.get('name')} | TOPIC: {topic} | "
        f"global={do_global} | specific={do_specific}"
    )

    json_files = sorted(json_dir.glob("*.json"))
    _log(f"[ENRICH] Numero di file JSON trovati: {len(json_files)}")

    llm_for_sections = make_llm() if do_specific else None
    llm_model_name = OPENAI_MODEL

    # indice globale
    index_records: List[Dict[str, Any]] = []

    # raw_directory stringa (es. 'cryoEM_documents/')
    raw_dir_str = ""
    if raw_docs_dir is not None:
        r = str(raw_docs_dir)
        raw_dir_str = r if r.endswith("/") else r + "/"

    # metadataDirectory stringa (es. 'dir1/project1/')
    meta_dir_str = str(json_dir)
    meta_dir_str = meta_dir_str if meta_dir_str.endswith("/") else meta_dir_str + "/"

    for idx, json_path in enumerate(json_files, start=1):
        name = json_path.name
        stem = json_path.stem

        # Salta file che NON sono JSON "grezzi" da arricchire
        if (
            name.endswith("_metadata.json")
            or "_keywords_" in name
            or name == "metadata_index.json"
        ):
            continue

        _log(f"[ENRICH] [{idx}/{len(json_files)}] {name}")

        # percorsi di output per questo documento
        raw_text_json_path = json_path
        metadata_json_path = json_dir / f"{stem}_metadata.json"

        global_keywords_json_path = json_dir / f"{stem}_keywords_global.json"
        global_keywords_csv_path = json_dir / f"{stem}_keywords_global.csv"

        specific_keywords_json_path = json_dir / f"{stem}_keywords_specific.json"
        specific_keywords_csv_path = json_dir / f"{stem}_keywords_specific.csv"

        # ------------ modalità no-recompute ------------
        if no_recompute:
            needed_paths: List[Path] = [metadata_json_path]
            if do_global:
                needed_paths.extend(
                    [global_keywords_json_path, global_keywords_csv_path]
                )
            if do_specific:
                needed_paths.extend(
                    [specific_keywords_json_path, specific_keywords_csv_path]
                )

            if all(p.exists() for p in needed_paths):
                _log(
                    f"[ENRICH][SKIP] Output già presente per {stem}, salto per --no-recompute."
                )
                # carica metadata per compilare l'indice
                try:
                    existing = json.loads(
                        metadata_json_path.read_text(encoding="utf-8")
                    )
                except Exception as e:
                    _log(
                        f"[ENRICH][WARN] Impossibile leggere metadata {metadata_json_path}: {e}"
                    )
                    continue

                pdf_filename = (
                    existing.get("pdf_filename") or (stem + ".pdf")
                )
                title = (existing.get("title") or "").strip()
                doi = (existing.get("doi") or "N/A").strip() or "N/A"
                author = (existing.get("author") or "N/A").strip() or "N/A"
                journal = (existing.get("journal") or "N/A").strip() or "N/A"
                type_of_study = (
                    existing.get("type of study") or "N/A"
                ).strip() or "N/A"
                year = (existing.get("year") or "N/A").strip() or "N/A"

                index_record = {
                    "raw_directory": raw_dir_str,
                    "metadataDirectory": meta_dir_str,
                    "raw_filename": pdf_filename,
                    "raw_text_json_filename": str(raw_text_json_path),
                    "metadata_json_filename": str(metadata_json_path),
                    "keywords_global_file": str(global_keywords_json_path)
                    if do_global
                    else "",
                    "keywords_global_csv": str(global_keywords_csv_path)
                    if do_global
                    else "",
                    "keywords_specific_file": str(specific_keywords_json_path)
                    if do_specific
                    else "",
                    "keywords_specific_csv": str(specific_keywords_csv_path)
                    if do_specific
                    else "",
                    "doi": doi,
                    "llm_model": existing.get("llm_model", llm_model_name),
                    "title": title,
                    "author": author,
                    "journal": journal,
                    "type of study": type_of_study,
                    "year": year,
                }
                index_records.append(index_record)
                continue

        # ------------ carica JSON grezzo ------------
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            _log(f"[ENRICH][WARN] Impossibile leggere JSON {json_path}: {e}")
            continue

        pdf_filename = meta.get("pdf_filename") or (stem + ".pdf")
        title = (meta.get("title") or "").strip()
        full_text = _build_full_text(meta)

        if not full_text.strip():
            _log(
                f"[ENRICH][WARN] Testo vuoto per {pdf_filename}, salto arricchimento."
            )
            continue

        # ------------ Analisi globale (relevance + global_extraction) ------------
        if do_global:
            rel = score_relevance(topic=topic, text=full_text, profile=profile)
            rel_pct = int(rel.get("relevance_percentage", 0))
            relevance_flag = 1 if rel_pct >= relevance_threshold else 0
            rel_rationale = (rel.get("rationale") or "").strip()

            global_record = extract_global(
                title=title, text=full_text, profile=profile
            )
            if not isinstance(global_record, dict):
                global_record = {}
            global_record = fill_na_with_rag(global_record, full_text, profile)
        else:
            rel_pct = 0
            relevance_flag = 0
            rel_rationale = "global_analysis_disabled"
            global_record = {}

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

        # DOI + info bibliografiche dal global_record o dal JSON grezzo
        doi = (global_record.get("doi") or meta.get("doi") or "N/A").strip() or "N/A"
        author = (global_record.get("author") or meta.get("author") or "N/A").strip() or "N/A"
        journal = (global_record.get("journal") or meta.get("journal") or "N/A").strip() or "N/A"
        type_of_study = (
            global_record.get("type of study")
            or meta.get("type of study")
            or "N/A"
        ).strip() or "N/A"
        year = (global_record.get("year") or meta.get("year") or "N/A").strip() or "N/A"

        # ------------ Section / paragraph-level enrichment ------------
        sec_records: List[Dict[str, Any]] = []
        if do_specific and llm_for_sections is not None:
            sections = meta.get("sections") or []

            for s_idx, sec in enumerate(sections):
                if not isinstance(sec, dict):
                    continue

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
                    "section_general_keywords": [],
                    "section_methods_keywords": [],
                    "section_results_keywords": [],
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
                    _log(
                        f"[ENRICH][WARN] Section-level enrichment failed for {pdf_filename} sec {s_idx}: {e}"
                    )
                    sec_data = dict(fallback_section)

                sec_data["section_secondary_questions"] = _ensure_list(
                    sec_data.get("section_secondary_questions")
                )
                sec_data["section_qa_local"] = _ensure_list(
                    sec_data.get("section_qa_local")
                )

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

        # ------------ Costruzione del record di metadata (enrichment) ------------
        metadata_record: Dict[str, Any] = {
            "pdf_filename": pdf_filename,
            "title": title,
            "llm_enrichment": llm_enrichment,
            "doi": doi,
            "llm_model": llm_model_name,
            "author": author,
            "journal": journal,
            "type of study": type_of_study,
            "year": year,
        }
        if sec_records:
            metadata_record["section_enrichment"] = sec_records

        # salva metadata JSON
        metadata_json_path.write_text(
            json.dumps(metadata_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _log(f"[ENRICH] Salvato JSON metadata: {metadata_json_path.name}")

        # genera JSON+CSV di keywords globali / specifiche
        _save_keywords_json_and_csv_from_metadata_json(
            metadata_json_path=metadata_json_path,
            global_json_path=global_keywords_json_path,
            global_csv_path=global_keywords_csv_path,
            specific_json_path=specific_keywords_json_path,
            specific_csv_path=specific_keywords_csv_path,
            do_global=do_global,
            do_specific=do_specific,
        )

        # ------------ Record per indice globale ------------
        index_record = {
            "raw_directory": raw_dir_str,
            "metadataDirectory": meta_dir_str,
            "raw_filename": pdf_filename,
            "raw_text_json_filename": str(raw_text_json_path),
            "metadata_json_filename": str(metadata_json_path),
            "keywords_global_file": str(global_keywords_json_path)
            if do_global
            else "",
            "keywords_global_csv": str(global_keywords_csv_path)
            if do_global
            else "",
            "keywords_specific_file": str(specific_keywords_json_path)
            if do_specific
            else "",
            "keywords_specific_csv": str(specific_keywords_csv_path)
            if do_specific
            else "",
            "doi": doi,
            "llm_model": llm_model_name,
            "title": title,
            "author": author,
            "journal": journal,
            "type of study": type_of_study,
            "year": year,
        }

        index_records.append(index_record)

    # ------------ Salvataggio indice globale ------------
    if index_records:
        if index_path is None:
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
        description=(
            "Parse PDF con GROBID e arricchisci i JSON con LLM "
            "(documento + sezioni) salvando i risultati in file separati."
        )
    )
    ap.add_argument(
        "--docs-dir",
        help="Directory con i documenti in ingresso (PDF o JSON). Obbligatoria se non si usa --i.",
    )
    ap.add_argument(
        "--docs-type",
        choices=["paper", "json"],
        default="paper",
        help=(
            "Tipo di documenti in --docs-dir. "
            "'paper' = PDF da parsare (GROBID se --grobid, altrimenti estrazione testuale); "
            "'json' = JSON già parsati."
        ),
    )
    ap.add_argument(
        "--grobid-url",
        default="http://localhost:8070",
        help="URL base del server GROBID (usato solo se --docs-type paper).",
    )
    ap.add_argument(
        "--out-json-dir",
        default=None,
        help=(
            "Compatibilità con versione precedente. Ora interpretato come root di output "
            "se --out-dir non è specificato."
        ),
    )
    ap.add_argument(
        "--out-dir",
        dest="out_dir",
        default=None,
        help=(
            "Directory radice per gli output. "
            "Con --o project1 crea OUT_DIR/project1.json e OUT_DIR/project1/."
        ),
    )
    ap.add_argument(
        "--o",
        dest="index_basename",
        default="metadata_index",
        help=(
            "Basename del file di indice. "
            "Con --o project1 crea project1.json (o OUT_DIR/project1.json) e directory project1/."
        ),
    )
    ap.add_argument(
        "--enrich",
        dest="enrich_yaml",
        default=None,
        help=(
            "File YAML di profilo per l'arricchimento LLM (documento + sezione). "
            "Se omesso, fa solo il parsing dei documenti."
        ),
    )
    ap.add_argument(
        "--relevance-threshold",
        type=int,
        default=50,
        help=(
            "Soglia su relevance_percentage per impostare la variabile binaria "
            "'relevance' (default: 50)."
        ),
    )
    ap.add_argument(
        "--i",
        dest="input_index",
        default=None,
        help=(
            "File JSON di indice esistente (es. metadata_index.json o project1.json). "
            "In questo caso il programma legge i percorsi raw_text_json_filename dall'indice "
            "e fa solo l'enrichment LLM."
        ),
    )
    ap.add_argument(
        "--no-global",
        action="store_true",
        help="Disabilita l'analisi globale (relevance + global_extraction).",
    )
    ap.add_argument(
        "--no-specific",
        action="store_true",
        help="Disabilita l'analisi a livello di sezione (section_enrichment).",
    )
    ap.add_argument(
        "--no-recompute",
        action="store_true",
        help=(
            "Se i file di output (metadata + keywords richieste) esistono già, "
            "non ricalcola l'enrichment LLM."
        ),
    )
    ap.add_argument(
        "--grobid",
        action="store_true",
        help=(
            "Se presente, usa GROBID per parsare i PDF (richiede server GROBID). "
            "Se assente, usa una estrazione testuale standard da PDF."
        ),
    )

    args = ap.parse_args()

    # Caso: arricchimento da indice esistente
    if args.input_index:
        index_path = Path(args.input_index)
        if not index_path.is_file():
            raise SystemExit(f"[ERROR] File di indice non trovato: {index_path}")
        if not args.enrich_yaml:
            raise SystemExit(
                "[ERROR] Con --i è necessario specificare anche --enrich <profilo.yml>."
            )

        records = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(records, list) or not records:
            raise SystemExit(f"[ERROR] Indice {index_path} vuoto o malformato.")

        # si assume che tutti i JSON siano in un'unica directory
        json_dirs = {
            Path(rec["raw_text_json_filename"]).parent
            for rec in records
            if "raw_text_json_filename" in rec
        }
        if len(json_dirs) != 1:
            raise SystemExit(
                "[ERROR] Con --i è supportata solo una singola directory di JSON."
            )

        json_dir = next(iter(json_dirs))
        json_dir.mkdir(parents=True, exist_ok=True)

        enrich_json_documents(
            json_dir=json_dir,
            enrich_yaml=Path(args.enrich_yaml),
            relevance_threshold=args.relevance_threshold,
            raw_docs_dir=None,
            index_path=index_path,
            no_global=args.no_global,
            no_specific=args.no_specific,
            no_recompute=args.no_recompute,
        )
        return

    # Caso standard: partenza da docs-dir
    if not args.docs_dir:
        raise SystemExit("[ERROR] È necessario specificare --docs-dir oppure --i.")

    docs_dir = Path(args.docs_dir)
    if not docs_dir.is_dir():
        raise SystemExit(
            f"[ERROR] docs-dir non esiste o non è una directory: {docs_dir}"
        )

    # root di output: precedenza a --out-dir, poi --out-json-dir, altrimenti CWD
    if args.out_dir:
        root_out_dir = Path(args.out_dir)
    elif args.out_json_dir:
        root_out_dir = Path(args.out_json_dir)
    else:
        root_out_dir = Path(".")
    root_out_dir.mkdir(parents=True, exist_ok=True)

    index_basename = args.index_basename
    project_dir = root_out_dir / index_basename
    project_dir.mkdir(parents=True, exist_ok=True)

    # path dell'indice
    index_path = root_out_dir / f"{index_basename}.json"

    # Decidi raw_docs_dir (per l'indice) e json_dir (directory con i JSON grezzi + metadata)
    if args.docs_type == "paper":
        # parsing -> JSON grezzi in project_dir
        raw_docs_dir = docs_dir
        json_dir = project_dir

        if args.grobid:
            _log("[INFO] Opzione --grobid attiva: uso GROBID per parsare i PDF.")
            run_grobid_papers(
                pdf_dir=docs_dir,
                out_json_dir=project_dir,
                grobid_url=args.grobid_url,
                max_pages=None,
            )
        else:
            _log(
                "[INFO] Opzione --grobid NON attiva: uso estrazione testuale semplice "
                "da PDF (senza GROBID)."
            )
            run_plain_papers(
                pdf_dir=docs_dir,
                out_json_dir=project_dir,
                max_pages=None,
            )
    else:
        # docs-type = json: copia i JSON grezzi in project_dir per uniformare la struttura
        raw_docs_dir = docs_dir
        json_dir = project_dir
        for src in sorted(docs_dir.glob("*.json")):
            dst = json_dir / src.name
            if not dst.exists():
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    # Enrichment opzionale: genera <stem>_metadata.json, <stem>_keywords_global.json,
    # <stem>_keywords_specific.json e l'indice globale (index_path).
    if args.enrich_yaml:
        enrich_json_documents(
            json_dir=json_dir,
            enrich_yaml=Path(args.enrich_yaml),
            relevance_threshold=args.relevance_threshold,
            raw_docs_dir=raw_docs_dir,
            index_path=index_path,
            no_global=args.no_global,
            no_specific=args.no_specific,
            no_recompute=args.no_recompute,
        )
    else:
        _log(
            "[INFO] Nessun file YAML di enrichment specificato: parsing completato."
        )


if __name__ == "__main__":
    main()

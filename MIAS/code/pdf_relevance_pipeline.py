#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF Relevance & Structured Extraction Pipeline (profile-based)

This version is compatible with:
    langchain==0.0.349
    langchain-core==0.0.13
    langchain-community==0.0.1

It uses ONLY the classic LangChain imports:
  - langchain.chat_models.ChatOpenAI
  - langchain.embeddings.HuggingFaceEmbeddings
  - langchain.vectorstores.Chroma
  - langchain.schema.SystemMessage, HumanMessage

All prompts / schema / questions live in the YAML profile (e.g. marine_valuation.yml).
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import yaml

# ---- .env ----
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in environment or .env")

# ---- PDF / OCR / tables ----
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import pytesseract
    from pdf2image import convert_from_path
except Exception:
    pytesseract = None
    convert_from_path = None

try:
    import camelot
except Exception:
    camelot = None

import pandas as pd

# ---- LangChain (old monolithic API) ----
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


# ----------------------------------------------------------------------
# Profile loading (YAML)
# ----------------------------------------------------------------------
def load_profile(config_path: Path) -> Dict[str, Any]:
    """
    Load a profile YAML (e.g. marine_valuation.yml) and normalise structure.

    Expected top-level keys:
      - profile_name: str
      - description: str (optional)
      - topic: str
      - schema_fields: list[str]
      - evidence_exclude: list[str] (optional)
      - relevance.system_prompt: str
      - global_extraction.system_prompt: str
      - fields: mapping field_name -> {question: str, evidence: bool}
    """
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Profile YAML must be a mapping: {config_path}")

    schema_fields = data.get("schema_fields")
    if not schema_fields or not isinstance(schema_fields, list):
        raise ValueError("Profile YAML missing 'schema_fields' (list).")

    schema_fields = [str(f) for f in schema_fields]

    fields_cfg_raw = data.get("fields", {}) or {}
    if not isinstance(fields_cfg_raw, dict):
        raise ValueError("'fields' must be a mapping of field_name -> config dict.")

    fields_cfg: Dict[str, Dict[str, Any]] = {}
    field_queries: Dict[str, str] = {}

    for name, cfg in fields_cfg_raw.items():
        if cfg is None:
            cfg = {}
        if not isinstance(cfg, dict):
            cfg = {}
        fname = str(name)
        q = cfg.get("question") or ""
        fields_cfg[fname] = {
            "question": str(q),
            "evidence": bool(cfg.get("evidence", True)),
        }
        if q.strip():
            field_queries[fname] = str(q)

    rel_prompt = ((data.get("relevance") or {}).get("system_prompt") or "").strip()
    if not rel_prompt:
        raise ValueError("Profile YAML must define relevance.system_prompt.")

    glob_prompt = ((data.get("global_extraction") or {}).get("system_prompt") or "").strip()
    if not glob_prompt:
        raise ValueError("Profile YAML must define global_extraction.system_prompt.")

    evidence_exclude = set(map(str, data.get("evidence_exclude", [])))

    profile: Dict[str, Any] = {
        "name": data.get("profile_name", "default_profile"),
        "description": data.get("description", ""),
        "topic": str(data.get("topic", "")).strip(),
        "schema_fields": schema_fields,
        "fields": fields_cfg,
        "field_queries": field_queries,
        "evidence_exclude": evidence_exclude,
        "relevance_prompt": rel_prompt,
        "global_extraction_prompt": glob_prompt,
    }
    return profile


# ----------------------------------------------------------------------
# LLM / Embeddings
# ----------------------------------------------------------------------
def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model_name=OPENAI_MODEL,
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )


def make_embed() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={"device": "cpu"},encode_kwargs={"device": "cpu"},)


# ----------------------------------------------------------------------
# PDF reading: text + OCR + tables
# ----------------------------------------------------------------------
def _pymupdf_text(path: Path, max_pages: Optional[int]) -> str:
    if not fitz:
        return ""
    text_parts: List[str] = []
    try:
        with fitz.open(str(path)) as doc:
            n = len(doc)
            if max_pages:
                n = min(n, max_pages)
            for i in range(n):
                page = doc.load_page(i)
                text_parts.append(f"\n\n=== PAGE {i+1} ===\n")
                text_parts.append(page.get_text("text") or "")
    except Exception:
        return ""
    return "".join(text_parts).strip()


def _ocr_text(path: Path, max_pages: Optional[int]) -> str:
    if not (pytesseract and convert_from_path):
        return ""
    try:
        images = convert_from_path(str(path), dpi=300)
    except Exception:
        return ""
    lines: List[str] = []
    for idx, img in enumerate(images[: (max_pages or len(images))]):
        try:
            txt = pytesseract.image_to_string(img)
        except Exception:
            txt = ""
        if txt and txt.strip():
            lines.append(f"\n\n=== PAGE {idx+1} (OCR) ===\n{txt}")
    return "\n".join(lines).strip()


def _tables_text(path: Path, max_pages: Optional[int]) -> str:
    if not camelot:
        return ""
    try:
        pages = "1-{}".format(max_pages if max_pages else 200)
        tables = camelot.read_pdf(str(path), pages=pages)
        parts: List[str] = []
        for t in tables:
            parts.append(t.df.to_csv(index=False))
        if parts:
            return "\n\n[TABLES]\n" + "\n\n---\n\n".join(parts)
    except Exception:
        pass
    return ""


def read_pdf_text(
    path: Path,
    max_pages: Optional[int] = None,
    use_ocr_fallback: bool = True,
    with_tables: bool = True,
) -> str:
    text = _pymupdf_text(path, max_pages)
    if use_ocr_fallback and len(text) < 500:
        ocr = _ocr_text(path, max_pages)
        if len(ocr) > len(text):
            text = ocr
    if with_tables:
        tb = _tables_text(path, max_pages)
        if tb:
            text = f"{text}\n\n{tb}"
    return text.strip()


# ----------------------------------------------------------------------
# Chunking & Retriever (RAG)
# ----------------------------------------------------------------------
def build_retriever_from_text(text: str, k: int = 8):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    docs = splitter.create_documents([text])
    vs = Chroma.from_documents(docs, embedding=make_embed())
    return vs.as_retriever(search_kwargs={"k": k})


def top_k_snippets_for_query(text: str, query: str, k: int = 8) -> str:
    retriever = build_retriever_from_text(text, k=k)
    docs = retriever.get_relevant_documents(query)
    return "\n\n---\n\n".join(d.page_content[:2000] for d in docs)


# ----------------------------------------------------------------------
# Relevance classification (profile-driven)
# ----------------------------------------------------------------------
def score_relevance(topic: str, text: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns: {'relevance_percentage': int[0..100], 'rationale': str}
    Binary 'relevance' is decided in the calling code (threshold).
    """
    ctx = top_k_snippets_for_query(text, topic, k=10)
    llm = make_llm()
    sys_prompt = profile["relevance_prompt"]
    usr = f"TOPIC:\n{topic}\n\nCONTEXT:\n{ctx[:12000]}"
    msg = llm(
        [SystemMessage(content=sys_prompt), HumanMessage(content=usr)]
    )
    content = getattr(msg, "content", "").strip()
    try:
        m = re.search(r"\{.*\}", content, flags=re.S)
        data = json.loads(m.group(0) if m else content)
        rp = int(data.get("relevance_percentage", 0))
        rp = max(0, min(100, rp))
        rat = (data.get("rationale") or "").strip()
        return {"relevance_percentage": rp, "rationale": rat}
    except Exception:
        return {"relevance_percentage": 0, "rationale": "parse_error"}


# ----------------------------------------------------------------------
# Extractors: global + per-field (RAG)
# ----------------------------------------------------------------------
def extract_global(title: str, text: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    llm = make_llm()
    sys_prompt = profile["global_extraction_prompt"]
    usr = f"KNOWN_TITLE: {title}\n\nCONTEXT:\n{text[:12000]}"
    msg = llm(
        [SystemMessage(content=sys_prompt), HumanMessage(content=usr)]
    )
    content = getattr(msg, "content", "").strip()
    try:
        m = re.search(r"\{.*\}", content, flags=re.S)
        return json.loads(m.group(0) if m else content)
    except Exception:
        schema_fields: List[str] = profile["schema_fields"]
        out = {
            k: "N/A"
            for k in schema_fields
            if k not in ("pdf_filename", "relevance", "relevance_percentage")
        }
        out["title"] = title or "N/A"
        return out


def build_field_retriever(text: str):
    return build_retriever_from_text(text, k=6)


def ask_field(
    field: str,
    question: str,
    retriever,
) -> Dict[str, str]:
    docs = retriever.get_relevant_documents(question)
    ctx = "\n\n---\n\n".join(d.page_content[:2000] for d in docs)
    llm = make_llm()
    sys = (
        "Extract the requested field from the CONTEXT.\n"
        "Rules:\n"
        "- Be precise and conservative: do not infer beyond the text.\n"
        "- If the information is not present, answer exactly 'N/A'.\n"
        "- Return ONLY valid JSON with keys 'value' and 'evidence'.\n"
        "- 'evidence' must be a short literal quote from the context (or empty if N/A)."
    )
    usr = f"FIELD: {field}\nINSTRUCTIONS:\n{question}\n\nCONTEXT:\n{ctx[:12000]}"
    msg = llm(
        [SystemMessage(content=sys), HumanMessage(content=usr)]
    )
    content = getattr(msg, "content", "").strip()
    try:
        m = re.search(r"\{.*\}", content, flags=re.S)
        data = json.loads(m.group(0) if m else content)
        val = (data.get("value") or "").strip() or "N/A"
        ev = (data.get("evidence") or "").strip()
        return {"value": val, "evidence": ev}
    except Exception:
        return {"value": "N/A", "evidence": ""}


def fill_na_with_rag(
    base_record: Dict[str, Any],
    full_text: str,
    profile: Dict[str, Any],
    target_fields: Optional[set] = None,
) -> Dict[str, Any]:
    """
    For each field with a configured question in the profile:
      - if target_fields is provided and the field is in target_fields,
        force a RAG-based re-extraction (even if the current value is not N/A);
      - otherwise, if the current value is empty or 'N/A', query the retriever and fill it.

    Evidence columns are added only if field_config['evidence'] is True.
    """
    out = dict(base_record)
    retriever = build_field_retriever(full_text)
    fields_cfg: Dict[str, Dict[str, Any]] = profile["fields"]

    # normalizza target_fields a set di stringhe
    if target_fields is not None:
        target_fields = {str(f).strip() for f in target_fields if str(f).strip()}

    for field, cfg in fields_cfg.items():
        question = (cfg.get("question") or "").strip()
        if not question:
            continue

        # campo attuale
        cur = str(out.get(field, "")).strip()

        # caso 1: re-estrazione forzata se il campo è nei target_fields
        if target_fields is not None and field in target_fields:
            res = ask_field(field, question, retriever)
            out[field] = res["value"] if res["value"] else "N/A"
            if cfg.get("evidence", True):
                out[f"{field}__evidence"] = res["evidence"]
            continue

        # caso 2: riempi solo se vuoto / N/A
        if cur in ("", "N/A"):
            res = ask_field(field, question, retriever)
            out[field] = res["value"] if res["value"] else "N/A"
            if cfg.get("evidence", True):
                out[f"{field}__evidence"] = res["evidence"]

    return out


# ----------------------------------------------------------------------
# Pipeline for a single PDF
# ----------------------------------------------------------------------
def process_pdf(
    pdf_path: Path,
    topic: str,
    profile: Dict[str, Any],
    relevance_threshold: int = 50,
    max_pages: Optional[int] = 12,
    per_file_action: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    schema_fields: List[str] = profile["schema_fields"]

    row: Dict[str, Any] = {k: "N/A" for k in schema_fields}
    row["pdf_filename"] = pdf_path.name

    text = read_pdf_text(pdf_path, max_pages=max_pages, use_ocr_fallback=True, with_tables=True)
    if not text:
        row["relevance"] = 0
        row["relevance_percentage"] = 0
        row["note"] = "empty_text_or_unreadable"
        return row

    # --- relevance scoring ---
    rel = score_relevance(topic, text, profile=profile)
    row["relevance_percentage"] = int(rel.get("relevance_percentage", 0))
    row["relevance"] = 1 if row["relevance_percentage"] >= int(relevance_threshold) else 0
    row["note"] = (rel.get("rationale") or "").strip()

    # se per_file_action dice che potrebbe essere un falso negativo, forza relevance=1
    action_code = None
    fields_to_focus_set: Optional[set] = None
    if per_file_action is not None:
        action_code = str(per_file_action.get("action_code", "")).strip()
        fields_to_focus = str(per_file_action.get("fields_to_focus", "") or "").strip()
        if fields_to_focus:
            fields_to_focus_set = {f.strip() for f in fields_to_focus.split(";") if f.strip()}

        if action_code == "check_relevance_false_negative":
            # forza l'articolo come "rilevante" per garantire l'estrazione
            row["relevance"] = 1

    # se ancora non rilevante, fermati qui
    if row["relevance"] == 0:
        return row

    # --- global extraction + RAG ---
    title_guess = _guess_title(text) or pdf_path.stem.replace("_", " ")
    global_data = extract_global(title_guess, text, profile=profile)

    for k, v in global_data.items():
        if k in row:
            if isinstance(v, str):
                row[k] = v if v.strip() else row[k]
            else:
                row[k] = v

    # se ci sono campi target da re-estrarre, passali a fill_na_with_rag
    row = fill_na_with_rag(
        base_record=row,
        full_text=text,
        profile=profile,
        target_fields=fields_to_focus_set,
    )

    # normalizzazioni
    row["protected area"] = _norm_binary(row.get("protected area", "N/A"))
    row["single site/multiple site"] = _norm_binary(row.get("single site/multiple site", "N/A"))
    row["CVM_elicitation"] = _norm_binary(row.get("CVM_elicitation", "N/A"))
    row["estimate"] = _only_number(row.get("estimate", "N/A"))

    return row


def load_per_file_actions_csv(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load a CSV with per-file actions (from refinement agent) into a dict
    mapping pdf_filename -> action_row_dict.

    Expected columns include at least: 'pdf_filename'.
    Additional columns (action_code, fields_to_focus, agent_instruction, etc.)
    are kept as-is in the dict.
    """
    if path is None or not path.is_file():
        return {}

    df = pd.read_csv(path)
    if "pdf_filename" not in df.columns:
        return {}

    actions: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        fname = str(row["pdf_filename"])
        actions[fname] = row.to_dict()
    return actions


# ----------------------------------------------------------------------
# Pipeline for a directory of PDFs
# ----------------------------------------------------------------------
def process_directory(
    pdf_dir: Path,
    topic: str,
    out_csv: Path,
    profile: Dict[str, Any],
    relevance_threshold: int = 50,
    max_pages: Optional[int] = 12,
    actions_csv: Optional[Path] = None,
) -> Tuple[int, int]:
    """
    Process all PDFs in `pdf_dir` and write a CSV with fixed schema and
    optional evidence columns defined by the profile.

    Optionally, use a per-file actions CSV (from the refinement agent) to:
      - override relevance decisions for specific files (e.g. suspected false negatives);
      - trigger targeted re-extraction of specific fields (fields_to_focus).

    Returns:
        (n_all, n_relevant)
    """
    import time
    schema_fields: List[str] = profile["schema_fields"]
    fields_cfg: Dict[str, Dict[str, Any]] = profile["fields"]

    # mappa pdf_filename -> dict con action_code, fields_to_focus, etc.
    per_file_actions_map: Dict[str, Dict[str, Any]] = {}
    if actions_csv is not None:
        per_file_actions_map = load_per_file_actions_csv(actions_csv)

    rows: List[Dict[str, Any]] = []

    evid_cols: List[str] = []
    for field_name, cfg in fields_cfg.items():
        if cfg.get("evidence", True):
            evid_cols.append(f"{field_name}__evidence")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    pdf_list = sorted(pdf_dir.glob("*.pdf"))
    n_files = len(pdf_list)
    if n_files == 0:
        pd.DataFrame(columns=schema_fields + evid_cols).to_csv(out_csv, index=False)
        return 0, 0

    print(f"[INFO] Found {n_files} PDF files in {pdf_dir}")
    start_time = time.perf_counter()

    for idx, p in enumerate(pdf_list, start=1):
        t0 = time.perf_counter()
        per_file_action = per_file_actions_map.get(p.name)

        r = process_pdf(
            pdf_path=p,
            topic=topic,
            profile=profile,
            relevance_threshold=relevance_threshold,
            max_pages=max_pages,
            per_file_action=per_file_action,
        )
        t1 = time.perf_counter()

        # ensure all schema + evidence columns exist
        for col in schema_fields + evid_cols:
            if col.endswith("__evidence"):
                r.setdefault(col, "")
            else:
                r.setdefault(col, "N/A")
        rows.append(r)

        # progress + ETA
        elapsed_total = t1 - start_time
        avg_per_file = elapsed_total / idx
        remaining = avg_per_file * (n_files - idx)

        print(
            f"[PROGRESS] [{idx}/{n_files}] {p.name} done in {t1 - t0:.1f}s | "
            f"avg {avg_per_file:.1f}s/file | ETA {remaining/60:.1f} min"
        )

    if not rows:
        pd.DataFrame(columns=schema_fields + evid_cols).to_csv(out_csv, index=False)
        return 0, 0

    df = pd.DataFrame(rows, columns=schema_fields + evid_cols)
    df.to_csv(out_csv, index=False)

    if "relevance" in df:
        n_rel = int(df["relevance"].astype(str).astype(float).sum())
    else:
        n_rel = 0

    total_time = time.perf_counter() - start_time
    print(f"[INFO] Completed processing of {n_files} PDFs in {total_time/60:.1f} min")

    return len(df), n_rel


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _guess_title(text: str) -> str:
    m = re.search(r"\bAbstract\b|\bABSTRACT\b", text)
    head = text[: m.start()] if m else text[:1000]
    lines = [ln.strip() for ln in head.splitlines() if ln.strip()]
    if not lines:
        return ""
    lines_sorted = sorted(lines, key=len, reverse=True)
    for ln in lines_sorted:
        if 20 <= len(ln) <= 180:
            return ln
    return lines_sorted[0][:180]


def _norm_binary(x: str) -> str:
    s = (x or "").strip().lower()
    if s in ("1", "yes", "true", "y"):
        return "1"
    if s in ("0", "no", "false", "n"):
        return "0"
    if s in ("", "n/a"):
        return "0"
    return x


def _only_number(x: str) -> str:
    s = (x or "").strip()
    m = re.search(r"-?\d+(\.\d+)?", s.replace(",", ""))
    return m.group(0) if m else "N/A"

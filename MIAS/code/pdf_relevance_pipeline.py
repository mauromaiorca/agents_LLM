#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF Relevance & Structured Extraction Pipeline (v2, richieste utente)
- Relevance scoring 0..100 calibrato dal LLM con rubrica esplicita (niente default a 85)
- 'relevance' binaria calcolata nel codice in base alla soglia (--threshold)
- Rimozione dal CSV di: id_studio, id_sito, current, relevance_rationale
- Rimozione colonne evidence per: CVM_elicitation, metrics, estimate, description
- Lettura robusta PDF (PyMuPDF + OCR opzionale) + (facoltativo) tabelle
- Estrazione: globale + RAG per riempire i "N/A"
- Output CSV in <PDF_DIR>/analysis/ (colonne fisse)

Dipendenze base:
  pip install langchain langchain-openai langchain-community chromadb pymupdf pypdf python-dotenv
Opzionali (OCR/tabelle):
  brew install tesseract poppler ghostscript
  pip install pdf2image pytesseract camelot-py pdfplumber
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# ---- .env ----
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY non trovata in ambiente/.env")

# --- robust file/text handling
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

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Embeddings: preferisci langchain-huggingface, altrimenti fallback community
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # avviso deprecabile accettabile

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ----------------------------
# Schema colonne (ordine finale del CSV)
# ----------------------------
# Rimossi: id_studio, id_sito, current, relevance_rationale
SCHEMA_FIELDS = [
    "pdf_filename",
    "relevance",                 # 0/1, deciso dal codice (>= soglia)
    "relevance_percentage",      # 0..100, deciso dal LLM
    "author",
    "year",
    "title",
    "journal",
    "country",
    "spatial coordinates",
    "distance coast",
    "protected area",
    "number of marine alien species",
    "marine alien species",
    "respondent",
    "sample size",
    "type of sample",
    "survey mode",
    "type of study",
    "single site/multiple site",
    "CVM_elicitation",
    "metrics",
    "year of estimate",
    "u.m. estimate",
    "estimate",
    "description",
    "note",
]

# ---------------------------------------------
# FIELD_QUERIES (senza 'current' come richiesto)
# ---------------------------------------------
FIELD_QUERIES: Dict[str, str] = {
    "country": (
        "Task: identify the country/countries of the study site(s).\n"
        "Rules:\n"
        "- Extract only explicit site countries; do NOT infer from author affiliations.\n"
        "- If multiple countries are stated, list them separated by '; '.\n"
        "- If only a broad region is given (e.g., 'Mediterranean Basin'), output that literal region.\n"
        "- If countries are clearly more than one without an exact list, return 'multicountries'.\n"
        "- If absent, return 'N/A'.\n"
        "Output: a single line string."
    ),
    "spatial coordinates": (
        "Task: extract geographic coordinates (latitude/longitude) for the study site(s).\n"
        "Rules:\n"
        "- Prefer decimal degrees; if DMS, convert to decimal (5 decimals).\n"
        "- If multiple sites, list pairs as 'lat,lon' separated by ' | '.\n"
        "- If coordinates appear only in figures without explicit values, return 'N/A'.\n"
        "- Do NOT invent values.\n"
        "Output: e.g., '45.4370,12.3330 | 45.5100,12.3000' or 'N/A'."
    ),
    "distance coast": (
        "Task: extract the distance from the coast (in meters) if explicitly reported.\n"
        "Rules:\n"
        "- If the value is in km, convert to meters (e.g., 1.2 km -> 1200).\n"
        "- If a range is given, return the midpoint and indicate 'mean of range' in evidence.\n"
        "- If not explicitly stated, return 'N/A'.\n"
        "Output: numeric string (e.g., '1200') or 'N/A'."
    ),
    "protected area": (
        "Task: determine whether the site is inside a Marine Protected Area (MPA).\n"
        "Rules:\n"
        "- Return '1' if the text explicitly states the site is within an MPA/reserve.\n"
        "- Return '0' if the text states it is not in an MPA, or if unspecified.\n"
        "- Do not infer from general context.\n"
        "Output: '1' or '0'."
    ),
    "number of marine alien species": (
        "Task: number of marine alien (non-indigenous) species the estimate refers to.\n"
        "Rules:\n"
        "- Extract a clear integer count if explicitly reported; if multiple taxa listed, count them.\n"
        "- If ambiguous or not stated, return 'N/A'.\n"
        "Output: integer as string, or 'N/A'."
    ),
    "marine alien species": (
        "Task: list names of the marine alien species involved.\n"
        "Rules:\n"
        "- Use binomial names if given; otherwise the exact names reported.\n"
        "- Separate multiple species with '; '. If only higher taxa or ambiguous mention, return 'N/A'.\n"
        "Output: e.g., 'Rugulopteryx okamurae; Callinectes sapidus' or 'N/A'."
    ),
    "respondent": (
        "Task: identify the respondent group if there is a survey (resident, tourist, fisher, bather, sailor, etc.).\n"
        "Rules:\n"
        "- If no survey-based data collection, return 'N/A'. Prefer the exact label used by authors.\n"
        "Output: one term (lowercase) or 'N/A'."
    ),
    "sample size": (
        "Task: extract the sample size of respondents.\n"
        "Rules:\n"
        "- Return a single integer. If multiple waves/groups, return the stated total; otherwise the main group.\n"
        "- If absent, return 'N/A'.\n"
        "Output: integer as string, or 'N/A'."
    ),
    "type of sample": (
        "Task: identify the sampling design.\n"
        "Rules:\n"
        "- Choose: 'random', 'stratified', 'cluster', 'systematic', 'convenience', 'snowball', 'other', 'N/A'.\n"
        "- If multiple designs, pick the primary used for estimation.\n"
        "Output: one token."
    ),
    "survey mode": (
        "Task: identify the primary survey mode.\n"
        "Rules:\n"
        "- Choose: 'online', 'mail', 'telephone', 'face to face', 'mixed', 'N/A'. If multiple, use 'mixed'.\n"
        "Output: one token."
    ),
    "type of study": (
        "Task: classify the economic study type.\n"
        "Rules:\n"
        "- Choose: 'choice model', 'choice experiment', 'contingent valuation', 'costs', 'prices', "
        "'hedonic pricing', 'travel cost model', 'production functions', 'benefit transfer', 'other', 'N/A'.\n"
        "Output: one token."
    ),
    "single site/multiple site": (
        "Task: indicate if it is a travel-cost multiple-sites study.\n"
        "Rules:\n"
        "- Return '1' only if TCM with multiple sites; '0' otherwise (including non-TCM).\n"
        "Output: '1' or '0'."
    ),
    "CVM_elicitation": (
        "Task: indicate presence of contingent valuation and elicitation format.\n"
        "Rules:\n"
        "- Return '1' if CVM is used AND the elicitation format is stated (single/double/one-and-one-half/payment card);\n"
        "  return '0' otherwise.\n"
        "Output: '1' or '0'."
    ),
    "metrics": (
        "Task: identify the welfare metric reported.\n"
        "Rules:\n"
        "- Choose: 'WTP', 'WTA', 'CS', 'N/A'. If multiple, choose the main metric used in conclusions.\n"
        "Output: one token."
    ),
    "year of estimate": (
        "Task: the year (or base year) of the economic estimate, not necessarily publication year.\n"
        "Rules:\n"
        "- Extract explicit year/base-year; if range, return midpoint. If absent, 'N/A'.\n"
        "Output: YYYY or 'N/A'."
    ),
    "u.m. estimate": (
        "Task: unit of measure of the estimate.\n"
        "Rules:\n"
        "- Examples: 'EUR/person', 'EUR/household', 'EUR/visit'. Use singular; if absent, 'N/A'.\n"
        "Output: normalized unit or 'N/A'."
    ),
    "estimate": (
        "Task: central numeric estimate.\n"
        "Rules:\n"
        "- If several, choose main/central. Return only the numeric value (no unit/currency). "
        "If range only, return midpoint. If absent, 'N/A'.\n"
        "Output: numeric string or 'N/A'."
    ),
    "description": (
        "Task: brief description of the environmental change valued.\n"
        "Rules:\n"
        "- 1–2 sentences; plain text; mention driver and affected component; no citations.\n"
        "Output: 1–2 sentences."
    ),
}

# evidence da escludere nel CSV (richiesta utente)
EVIDENCE_EXCLUDE = {"CVM_elicitation", "metrics", "estimate", "description"}

# -----------------------
# Utility: LLM / Embedding
# -----------------------
def make_llm():
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=OPENAI_API_KEY or None)

def make_embed():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------
# Lettura PDF + OCR + Tabelle
# ---------------------------
def _pymupdf_text(path: Path, max_pages: Optional[int]) -> str:
    if not fitz:
        return ""
    text_parts = []
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
    lines = []
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
        parts = []
        for t in tables:
            parts.append(t.df.to_csv(index=False))
        if parts:
            return "\n\n[TABLES]\n" + "\n\n---\n\n".join(parts)
    except Exception:
        pass
    return ""

def read_pdf_text(path: Path, max_pages: Optional[int] = None, use_ocr_fallback: bool = True, with_tables: bool = True) -> str:
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

# --------------------------
# Chunking & Retriever (RAG)
# --------------------------
def build_retriever_from_text(text: str, k: int = 8):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    docs = splitter.create_documents([text])
    vs = Chroma.from_documents(docs, embedding=make_embed())  # in-memory
    return vs.as_retriever(search_kwargs={"k": k})

def top_k_snippets_for_query(text: str, query: str, k: int = 8) -> str:
    retriever = build_retriever_from_text(text, k=k)
    docs = retriever.invoke(query)  # no deprecation warning
    return "\n\n---\n\n".join(d.page_content[:2000] for d in docs)

# ----------------------
# Relevance classification (calibrata)
# ----------------------
def score_relevance(topic: str, text: str) -> Dict[str, Any]:
    """
    Restituisce: {'relevance_percentage': int[0..100], 'rationale': str}
    Il binario 'relevance' è deciso nel codice (>= soglia).
    """
    ctx = top_k_snippets_for_query(text, topic, k=10)
    llm = make_llm()
    sys = (
        "You are a careful research assistant. Given a TOPIC and CONTEXT snippets from a PDF, "
        "rate how relevant the PDF is to the topic on a 0–100 scale using the rubric:\n"
        " - 90–100: Directly about the topic; multiple strong matches in methods/results/discussion.\n"
        " - 75–89: Substantially related; at least one section (methods/results) is on-topic.\n"
        " - 50–74: Partially related; topic appears but tangential/limited or only in background.\n"
        " - 25–49: Weakly related; brief mention without substantive analysis.\n"
        " - 0–24: Not related.\n"
        "Do not default to round numbers; calibrate to the evidence. "
        "Return ONLY valid JSON with keys: relevance_percentage (integer 0..100), rationale (short string)."
    )
    usr = f"TOPIC:\n{topic}\n\nCONTEXT:\n{ctx[:12000]}"
    msg = llm.invoke([SystemMessage(content=sys), HumanMessage(content=usr)])
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

# --------------------------
# Estrattori: globale + per campo
# --------------------------
def extract_global(title: str, text: str) -> Dict[str, Any]:
    llm = make_llm()
    # NOTA: rimosso 'current' dalla lista dei campi attesi
    sys = (
        "Extract bibliographic and study fields from the CONTEXT. "
        "If a field is not present, write exactly 'N/A'. "
        "Return a flat JSON object with the following keys: "
        "author (string, '; ' separated), year (YYYY), title, journal, "
        "country, spatial coordinates, distance coast, protected area, "
        "number of marine alien species, marine alien species, respondent, sample size, "
        "type of sample, survey mode, type of study, single site/multiple site, "
        "CVM_elicitation, metrics, year of estimate, u.m. estimate, estimate, description, note."
    )
    usr = f"KNOWN_TITLE: {title}\n\nCONTEXT:\n{text[:12000]}"
    msg = llm.invoke([SystemMessage(content=sys), HumanMessage(content=usr)])
    content = getattr(msg, "content", "").strip()
    try:
        m = re.search(r"\{.*\}", content, flags=re.S)
        return json.loads(m.group(0) if m else content)
    except Exception:
        out = {k: "N/A" for k in SCHEMA_FIELDS if k not in ("pdf_filename","relevance","relevance_percentage")}
        out["title"] = title or "N/A"
        return out

def build_field_retriever(text: str):
    return build_retriever_from_text(text, k=6)

def ask_field(field: str, question: str, retriever, model: Optional[str] = None) -> Dict[str, str]:
    docs = retriever.invoke(question)
    ctx = "\n\n---\n\n".join(d.page_content[:2000] for d in docs)
    llm = make_llm() if not model else ChatOpenAI(model=model, temperature=0, api_key=OPENAI_API_KEY or None)
    sys = (
        "Extract the requested field from the CONTEXT.\n"
        "Rules:\n"
        "- Be precise and conservative: do not infer beyond the text.\n"
        "- If the information is not present, answer exactly 'N/A'.\n"
        "- Return ONLY valid JSON with keys 'value' and 'evidence'.\n"
        "- 'evidence' must be a short literal quote from the context (or empty if N/A)."
    )
    usr = f"FIELD: {field}\nINSTRUCTIONS:\n{question}\n\nCONTEXT:\n{ctx[:12000]}"
    msg = llm.invoke([SystemMessage(content=sys), HumanMessage(content=usr)])
    content = getattr(msg, "content", "").strip()
    try:
        m = re.search(r"\{.*\}", content, flags=re.S)
        data = json.loads(m.group(0) if m else content)
        val = (data.get("value") or "").strip() or "N/A"
        ev = (data.get("evidence") or "").strip()
        return {"value": val, "evidence": ev}
    except Exception:
        return {"value": "N/A", "evidence": ""}

def fill_na_with_rag(base_record: Dict[str, Any], full_text: str) -> Dict[str, Any]:
    out = dict(base_record)
    retriever = build_field_retriever(full_text)
    for field, prompt in FIELD_QUERIES.items():
        cur = str(out.get(field, "")).strip()
        if cur in ("", "N/A"):
            res = ask_field(field, prompt, retriever)
            out[field] = res["value"] if res["value"] else "N/A"
            # salva evidence SOLO se il campo NON è escluso
            if field not in EVIDENCE_EXCLUDE:
                out[f"{field}__evidence"] = res["evidence"]
    return out

# --------------------------
# Pipeline per una cartella
# --------------------------
def process_pdf(
    pdf_path: Path,
    topic: str,
    relevance_threshold: int = 50,
    max_pages: Optional[int] = 12
) -> Dict[str, Any]:
    row = {k: "N/A" for k in SCHEMA_FIELDS}
    row["pdf_filename"] = pdf_path.name

    text = read_pdf_text(pdf_path, max_pages=max_pages, use_ocr_fallback=True, with_tables=True)
    if not text:
        row["relevance"] = 0
        row["relevance_percentage"] = 0
        row["note"] = "empty_text_or_unreadable"
        return row

    rel = score_relevance(topic, text)
    row["relevance_percentage"] = int(rel.get("relevance_percentage", 0))
    # relevance binaria calcolata qui (coerente con la soglia scelta a runtime)
    row["relevance"] = 1 if row["relevance_percentage"] >= int(relevance_threshold) else 0
    row["note"] = (rel.get("rationale") or "").strip()

    # Se sotto soglia, non procedere all'estrazione dettagliata
    if row["relevance"] == 0:
        return row

    # estrazione globale + riempimento N/A con RAG
    title_guess = _guess_title(text) or pdf_path.stem.replace("_", " ")
    global_data = extract_global(title_guess, text)

    for k, v in global_data.items():
        if k in row:
            row[k] = v if (isinstance(v, str) and v.strip()) else row[k]

    row = fill_na_with_rag(row, text)

    # normalizzazioni
    row["protected area"] = _norm_binary(row.get("protected area", "N/A"))
    row["single site/multiple site"] = _norm_binary(row.get("single site/multiple site", "N/A"))
    row["CVM_elicitation"] = _norm_binary(row.get("CVM_elicitation", "N/A"))
    row["estimate"] = _only_number(row.get("estimate", "N/A"))

    return row

def process_directory(
    pdf_dir: Path,
    topic: str,
    out_csv: Path,
    relevance_threshold: int = 50,
    max_pages: Optional[int] = 12
) -> Tuple[int, int]:
    rows: List[Dict[str, Any]] = []

    # prepara solo le colonne evidence che NON sono escluse
    evid_cols = [f"{k}__evidence" for k in FIELD_QUERIES.keys() if k not in EVIDENCE_EXCLUDE]

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    for p in sorted(pdf_dir.glob("*.pdf")):
        r = process_pdf(p, topic, relevance_threshold=relevance_threshold, max_pages=max_pages)
        # assicurati che tutte le colonne finali + evidenze siano presenti
        for col in SCHEMA_FIELDS + evid_cols:
            r.setdefault(col, "" if col.endswith("__evidence") else "N/A")
        rows.append(r)

    if not rows:
        pd.DataFrame(columns=SCHEMA_FIELDS + evid_cols).to_csv(out_csv, index=False)
        return 0, 0

    df = pd.DataFrame(rows, columns=SCHEMA_FIELDS + evid_cols)
    df.to_csv(out_csv, index=False)
    n_rel = int(df["relevance"].astype(str).astype(float).sum()) if "relevance" in df else 0
    return len(df), n_rel

# -----------------
# Helpers
# -----------------
def _guess_title(text: str) -> str:
    m = re.search(r"\bAbstract\b|\bABSTRACT\b", text)
    head = text[: m.start()] if m else text[:1000]
    lines = [ln.strip() for ln in head.splitlines() if ln.strip()]
    if not lines:
        return ""
    lines = sorted(lines, key=len, reverse=True)
    for ln in lines:
        if 20 <= len(ln) <= 180:
            return ln
    return lines[0][:180]

def _norm_binary(x: str) -> str:
    s = (x or "").strip().lower()
    if s in ("1", "yes", "true", "y"):
        return "1"
    if s in ("0", "no", "false", "n"):
        return "0"
    return "0" if s == "" or s == "n/a" else x

def _only_number(x: str) -> str:
    s = (x or "").strip()
    m = re.search(r"-?\d+(\.\d+)?", s.replace(",", ""))
    return m.group(0) if m else "N/A"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="PDF relevance & extraction (v2)")
    ap.add_argument("--pdf-dir", required=True, help="Cartella con PDF")
    ap.add_argument("--out-csv", required=True, help="Output CSV")
    ap.add_argument("--topic", required=True, help="TOPIC di rilevanza")
    ap.add_argument("--threshold", type=int, default=50, help="Soglia % per considerare 'rilevante' (attiva estrazione)")
    ap.add_argument("--max-pages", type=int, default=12, help="Pagine max da leggere per PDF (0=tutte)")
    args = ap.parse_args()

    max_pages = None if args.max_pages <= 0 else args.max_pages
    n_all, n_rel = process_directory(
        Path(args.pdf_dir),
        args.topic,
        Path(args.out_csv),
        relevance_threshold=args.threshold,
        max_pages=max_pages
    )
    print(f"[DONE] Processati: {n_all}, rilevanti ≥ soglia: {n_rel} -> {args.out_csv}")
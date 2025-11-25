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

import requests
import xml.etree.ElementTree as ET

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
def extract_figure_captions_from_pdftext(
    path: Path,
    max_pages: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Heuristic extraction of figure captions directly from the PDF text layout,
    independent of GROBID.

    Strategy (simple but effective for many scientific PDFs):
      - use PyMuPDF to get text blocks per page;
      - within each block, look for lines that start with 'Fig.' or 'Figure';
      - treat the entire line as a caption.

    Returns a list of dicts:
      { 'page': int, 'label': 'Fig. 1', 'caption': 'Fig. 1. ...' }
    """
    if not fitz:
        return []

    captions: List[Dict[str, str]] = []
    try:
        with fitz.open(str(path)) as doc:
            n_pages = len(doc)
            if max_pages is not None:
                n_pages = min(n_pages, max_pages)

            for page_index in range(n_pages):
                page = doc.load_page(page_index)
                # blocks: (x0, y0, x1, y1, "text", block_no, block_type, ...)
                blocks = page.get_text("blocks")
                for b in blocks:
                    if len(b) < 5:
                        continue
                    text_block = b[4] or ""
                    if not text_block.strip():
                        continue
                    for line in text_block.splitlines():
                        line_strip = line.strip()
                        # match "Fig. 1", "Fig 1", "Figure 1", "Figure 1.", etc.
                        if re.match(r"^(Fig\.?|Figure)\s+\d+", line_strip, flags=re.IGNORECASE):
                            # label = "Fig. 1", caption = full line
                            m = re.match(r"^((Fig\.?|Figure)\s+\d+)", line_strip, flags=re.IGNORECASE)
                            label = m.group(1) if m else "Figure"
                            captions.append(
                                {
                                    "page": page_index + 1,
                                    "label": label.strip(),
                                    "caption": line_strip,
                                }
                            )
    except Exception as e:
        print(f"[WARN] extract_figure_captions_from_pdftext failed for {path.name}: {e}")

    return captions

def extract_figure_images_and_ocr(
    path: Path,
    out_dir: Optional[Path] = None,
    max_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Extract raster images from the PDF (likely including figures) and optionally run OCR
    on each image using Tesseract, if available.

    Returns a list of dicts with:
      {
        'page': int,
        'image_index': int,
        'file_path': str or '',
        'ocr_text': str,
      }

    If out_dir is not None, each image is also saved to disk.
    """
    results: List[Dict[str, Any]] = []
    if not fitz:
        return results

    # Tesseract may or may not be available
    ocr_available = pytesseract is not None
    pil_available = False
    try:
        from PIL import Image  # type: ignore
        pil_available = True
    except Exception:
        pil_available = False

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import io

        with fitz.open(str(path)) as doc:
            n_pages = len(doc)
            if max_pages is not None:
                n_pages = min(n_pages, max_pages)

            for page_index in range(n_pages):
                page = doc.load_page(page_index)
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image.get("image", None)
                    ext = base_image.get("ext", "png")
                    file_path_str = ""

                    # Save image to disk if requested
                    if out_dir is not None and image_bytes:
                        file_name = f"{path.stem}_p{page_index+1}_img{img_index+1}.{ext}"
                        file_path = out_dir / file_name
                        with open(file_path, "wb") as f:
                            f.write(image_bytes)
                        file_path_str = str(file_path)

                    # OCR
                    ocr_text = ""
                    if ocr_available and pil_available and image_bytes:
                        try:
                            from PIL import Image  # type: ignore

                            img_pil = Image.open(io.BytesIO(image_bytes))
                            ocr_text = pytesseract.image_to_string(img_pil)
                        except Exception:
                            ocr_text = ""

                    results.append(
                        {
                            "page": page_index + 1,
                            "image_index": img_index + 1,
                            "file_path": file_path_str,
                            "ocr_text": ocr_text.strip(),
                        }
                    )
    except Exception as e:
        print(f"[WARN] extract_figure_images_and_ocr failed for {path.name}: {e}")

    return results


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

def _grobid_fulltext_tei(pdf_path: Path, grobid_url: Optional[str] = None) -> str:
    """
    Chiama GROBID (processFulltextDocument) e restituisce il TEI XML come stringa.
    Se qualcosa va storto, ritorna stringa vuota.
    """
    base_url = grobid_url or os.getenv("GROBID_URL", "http://localhost:8070")
    endpoint = base_url.rstrip("/") + "/api/processFulltextDocument"

    try:
        with pdf_path.open("rb") as f:
            files = {"input": (pdf_path.name, f, "application/pdf")}
            resp = requests.post(endpoint, files=files, timeout=120)
        resp.raise_for_status()
        return resp.text or ""
    except Exception as e:
        print(f"[WARN] GROBID request failed for {pdf_path.name}: {e}")
        return ""


def _strip_ns(tag: str) -> str:
    """
    Remove XML namespace from a tag name, e.g. '{http://www.tei-c.org/ns/1.0}TEI' -> 'TEI'.
    """
    if not isinstance(tag, str):
        return ""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def parse_tei_full(tei_xml: str) -> Dict[str, Any]:
    """
    Parse the TEI XML returned by GROBID and extract as much structured information
    as reasonably possible:

      - title
      - authors (with name, email, orcid, affiliation refs)
      - affiliations (id + organisation + address/country text)
      - abstract
      - sections (id, type, n, title, text)
      - figures (from <figure> and <div type='figure'>: label + caption)
      - tables  (from <table> and <figure type='table'> / <div type='table'>)
      - references (biblStruct → authors, title, journal, year, DOI, raw text)
      - fulltext: a single plain-text string concatenating all above.

    This is still a best-effort parse: TEI is rich and heterogeneous, but the goal
    is to keep *all* relevant textual information in a structured way.
    """
    out: Dict[str, Any] = {
        "title": "",
        "authors": [],
        "affiliations": [],
        "abstract": "",
        "sections": [],
        "figures": [],
        "tables": [],
        "references": [],
        "fulltext": "",
    }

    if not tei_xml.strip():
        return out

    try:
        root = ET.fromstring(tei_xml)
    except Exception:
        # If TEI is malformed, keep everything as raw text
        out["fulltext"] = tei_xml.strip()
        return out

    # ------------------------------
    # HEADER: title, authors, affiliations
    # ------------------------------
    tei_header = None
    for elem in root:
        if _strip_ns(elem.tag) == "teiHeader":
            tei_header = elem
            break

    title = ""
    authors: List[Dict[str, Any]] = []
    affiliations: Dict[str, Dict[str, Any]] = {}

    if tei_header is not None:
        # fileDesc → titleStmt, sourceDesc, etc.
        file_desc = None
        for el in tei_header.iter():
            if _strip_ns(el.tag) == "fileDesc":
                file_desc = el
                break

        if file_desc is not None:
            # title: first <title> inside <titleStmt>
            for title_stmt in file_desc.iter():
                if _strip_ns(title_stmt.tag) == "titleStmt":
                    for t in title_stmt.iter():
                        if _strip_ns(t.tag) == "title":
                            title = " ".join(t.itertext()).strip()
                            break
                    break

            # collect affiliations with xml:id (if present) so we can link them
            for aff in file_desc.iter():
                if _strip_ns(aff.tag) == "affiliation":
                    aff_id = aff.get("{http://www.w3.org/XML/1998/namespace}id") or aff.get("xml:id") or ""
                    aff_text = " ".join(t.strip() for t in aff.itertext()).strip()
                    # try to extract organisation and country separately
                    org_names = []
                    country = ""
                    for ch in aff.iter():
                        ch_tag = _strip_ns(ch.tag)
                        if ch_tag == "orgName":
                            org_names.append(" ".join(ch.itertext()).strip())
                        elif ch_tag == "country":
                            country = " ".join(ch.itertext()).strip()
                    affiliation_rec = {
                        "id": aff_id,
                        "text": aff_text,
                        "org_names": org_names,
                        "country": country,
                    }
                    if aff_id:
                        affiliations[aff_id] = affiliation_rec
                    else:
                        # fallback: synthetic id
                        synthetic_id = f"aff_{len(affiliations)+1}"
                        affiliation_rec["id"] = synthetic_id
                        affiliations[synthetic_id] = affiliation_rec

            # authors
            for auth in file_desc.iter():
                if _strip_ns(auth.tag) == "author":
                    a: Dict[str, Any] = {
                        "full_name": "",
                        "forenames": [],
                        "surnames": [],
                        "email": "",
                        "orcid": "",
                        "aff_refs": [],
                    }
                    # persName with forename/surname
                    pers = None
                    for ch in auth:
                        if _strip_ns(ch.tag) == "persName":
                            pers = ch
                            break
                    if pers is not None:
                        for ch in pers:
                            tag = _strip_ns(ch.tag)
                            if tag in ("forename", "given"):
                                txt = " ".join(ch.itertext()).strip()
                                if txt:
                                    a["forenames"].append(txt)
                            elif tag in ("surname", "family"):
                                txt = " ".join(ch.itertext()).strip()
                                if txt:
                                    a["surnames"].append(txt)

                    # email, orcid, affiliation refs
                    for ch in auth.iter():
                        tag = _strip_ns(ch.tag)
                        if tag == "email":
                            a["email"] = " ".join(ch.itertext()).strip() or a["email"]
                        elif tag in ("idno", "id", "idno-type-orcid"):
                            # GROBID sometimes annotates ORCID as <idno type="ORCID">
                            if ch.get("type", "").lower() == "orcid":
                                a["orcid"] = " ".join(ch.itertext()).strip() or a["orcid"]
                        elif tag == "affiliation":
                            # could be a reference via @ref="#aff1" or inline text
                            ref = ch.get("ref") or ""
                            ref = ref.lstrip("#")
                            if ref:
                                if ref not in a["aff_refs"]:
                                    a["aff_refs"].append(ref)

                    # full name fallback
                    if a["forenames"] or a["surnames"]:
                        a["full_name"] = " ".join(a["forenames"] + a["surnames"]).strip()
                    else:
                        # if no structured name, use all text
                        a["full_name"] = " ".join(auth.itertext()).strip()

                    authors.append(a)

    out["title"] = title
    out["authors"] = authors
    out["affiliations"] = list(affiliations.values())

    # ------------------------------
    # ABSTRACT
    # ------------------------------
    abstracts: List[str] = []
    for abs_el in root.iter():
        if _strip_ns(abs_el.tag) == "abstract":
            txt = " ".join(t.strip() for t in abs_el.itertext())
            if txt:
                abstracts.append(txt)
    out["abstract"] = "\n\n".join(abstracts).strip()

    # ------------------------------
    # BODY: sections (div/head + text, with type/n)
    # ------------------------------
    sections = []
    for div in root.iter():
        if _strip_ns(div.tag) == "div":
            div_type = div.get("type", "") or ""
            div_n = div.get("n", "") or ""

            # section title: any <head> child
            heads = []
            for h in div:
                if _strip_ns(h.tag) == "head":
                    ht = " ".join(h.itertext()).strip()
                    if ht:
                        heads.append(ht)
            sec_title = " / ".join(heads)

            # section text: all text inside div (including nested <p>, etc.)
            sec_text = " ".join(t.strip() for t in div.itertext()).strip()
            if sec_text:
                sections.append(
                    {
                        "id": div.get("{http://www.w3.org/XML/1998/namespace}id")
                        or div.get("xml:id")
                        or "",
                        "type": div_type,
                        "n": div_n,
                        "title": sec_title,
                        "text": sec_text,
                    }
                )

    out["sections"] = sections

    # ------------------------------
    # FIGURES: from <figure> and <div type='figure'>
    # ------------------------------
    figures: List[Dict[str, Any]] = []

    # (a) explicit <figure> elements
    for fig in root.iter():
        if _strip_ns(fig.tag) in ("figure", "fig"):
            label = fig.get("n", "") or fig.get("xml:id") or ""
            caption = ""
            for child in fig.iter():
                if _strip_ns(child.tag) in ("figDesc", "head", "caption"):
                    caption = " ".join(child.itertext()).strip()
                    if caption:
                        break
            if not caption:
                caption = " ".join(t.strip() for t in fig.itertext()).strip()
            figures.append(
                {
                    "id": fig.get("{http://www.w3.org/XML/1998/namespace}id")
                    or fig.get("xml:id")
                    or "",
                    "label": label,
                    "caption": caption,
                }
            )

    # (b) <div type="figure"> used as figure containers
    for div in root.iter():
        if _strip_ns(div.tag) == "div" and div.get("type", "").lower() == "figure":
            label = div.get("n", "") or div.get("xml:id") or ""
            caption = ""
            # often <head> inside this div is the caption
            for child in div.iter():
                if _strip_ns(child.tag) in ("head", "figDesc", "caption"):
                    caption = " ".join(child.itertext()).strip()
                    if caption:
                        break
            if not caption:
                caption = " ".join(t.strip() for t in div.itertext()).strip()
            figures.append(
                {
                    "id": div.get("{http://www.w3.org/XML/1998/namespace}id")
                    or div.get("xml:id")
                    or "",
                    "label": label,
                    "caption": caption,
                }
            )

    out["figures"] = figures

    # ------------------------------
    # TABLES: <table>, <figure type='table'>, <div type='table'>
    # ------------------------------
    tables: List[Dict[str, Any]] = []

    for tbl in root.iter():
        tag = _strip_ns(tbl.tag)
        if tag == "table" or (tag == "figure" and tbl.get("type", "").lower() == "table"):
            label = tbl.get("n", "") or tbl.get("xml:id") or ""
            caption = ""
            for child in tbl.iter():
                if _strip_ns(child.tag) in ("head", "figDesc", "caption"):
                    caption = " ".join(child.itertext()).strip()
                    if caption:
                        break
            if not caption:
                caption = " ".join(t.strip() for t in tbl.itertext()).strip()
            tables.append(
                {
                    "id": tbl.get("{http://www.w3.org/XML/1998/namespace}id")
                    or tbl.get("xml:id")
                    or "",
                    "label": label,
                    "caption": caption,
                }
            )

    for div in root.iter():
        if _strip_ns(div.tag) == "div" and div.get("type", "").lower() == "table":
            label = div.get("n", "") or div.get("xml:id") or ""
            caption = ""
            for child in div.iter():
                if _strip_ns(child.tag) in ("head", "figDesc", "caption"):
                    caption = " ".join(child.itertext()).strip()
                    if caption:
                        break
            if not caption:
                caption = " ".join(t.strip() for t in div.itertext()).strip()
            tables.append(
                {
                    "id": div.get("{http://www.w3.org/XML/1998/namespace}id")
                    or div.get("xml:id")
                    or "",
                    "label": label,
                    "caption": caption,
                }
            )

    out["tables"] = tables

    # ------------------------------
    # REFERENCES (bibliography): biblStruct / bibl
    # ------------------------------
    references: List[Dict[str, Any]] = []

    for bibl in root.iter():
        tag = _strip_ns(bibl.tag)
        if tag not in ("biblStruct", "bibl"):
            continue

        ref_id = bibl.get("{http://www.w3.org/XML/1998/namespace}id") or bibl.get("xml:id") or ""
        ref_txt = " ".join(t.strip() for t in bibl.itertext()).strip()

        # try to parse some structure
        ref_authors: List[str] = []
        ref_title = ""
        ref_journal = ""
        ref_year = ""
        ref_doi = ""

        # titles can be nested: analytic/monogr/imprint
        for el in bibl.iter():
            el_tag = _strip_ns(el.tag)
            if el_tag == "author":
                a_txt = " ".join(el.itertext()).strip()
                if a_txt:
                    ref_authors.append(a_txt)
            elif el_tag == "title":
                # use the first title as main title, others as fallbacks
                if not ref_title:
                    ref_title = " ".join(el.itertext()).strip()
            elif el_tag in ("journal", "titleLevel"):
                if not ref_journal:
                    ref_journal = " ".join(el.itertext()).strip()
            elif el_tag == "date":
                if el.get("when"):
                    ref_year = el.get("when")[:4]
            elif el_tag == "idno":
                if el.get("type", "").lower() == "doi":
                    ref_doi = " ".join(el.itertext()).strip()

        references.append(
            {
                "id": ref_id,
                "authors": ref_authors,
                "title": ref_title,
                "journal": ref_journal,
                "year": ref_year,
                "doi": ref_doi,
                "text": ref_txt,
            }
        )

    out["references"] = references

    # ------------------------------
    # FULLTEXT: concatenate everything in a structured way
    # ------------------------------
    parts: List[str] = []

    if out["title"]:
        parts.append("=== TITLE ===\n" + out["title"])

    if out["authors"]:
        parts.append(
            "=== AUTHORS ===\n"
            + "\n".join(a["full_name"] or " ".join(a["forenames"] + a["surnames"]) for a in out["authors"])
        )

    if out["affiliations"]:
        parts.append(
            "=== AFFILIATIONS ===\n"
            + "\n\n".join(
                f"[{aff.get('id','')}] {aff.get('text','')}" for aff in out["affiliations"]
            )
        )

    if out["abstract"]:
        parts.append("=== ABSTRACT ===\n" + out["abstract"])

    if out["sections"]:
        sec_blocks = []
        for i, sec in enumerate(out["sections"], start=1):
            title_prefix = sec["title"].strip() or sec["type"] or f"Section {i}"
            sec_blocks.append(f"=== SECTION: {title_prefix} ===\n{sec['text']}")
        parts.append("\n\n".join(sec_blocks))

    if out["figures"]:
        fig_blocks = []
        for fig in out["figures"]:
            label = fig["label"] or fig["id"] or "Figure"
            fig_blocks.append(f"{label}: {fig['caption']}".strip())
        parts.append("=== FIGURES (captions from TEI) ===\n" + "\n\n".join(fig_blocks))

    if out["tables"]:
        tbl_blocks = []
        for tbl in out["tables"]:
            label = tbl["label"] or tbl["id"] or "Table"
            tbl_blocks.append(f"{label}: {tbl['caption']}".strip())
        parts.append("=== TABLES (captions from TEI) ===\n" + "\n\n".join(tbl_blocks))

    if out["references"]:
        ref_blocks = []
        for ref in out["references"]:
            ref_blocks.append(ref["text"])
        parts.append("=== REFERENCES ===\n" + "\n\n".join(ref_blocks))

    fulltext = "\n\n".join(parts).strip()
    if not fulltext:
        fulltext = " ".join(t.strip() for t in root.itertext()).strip()

    out["fulltext"] = fulltext
    return out


def _tei_to_plain_text(tei_xml: str) -> str:
    """
    Convert TEI XML returned by GROBID into plain text, using parse_tei_full.

    This keeps *all* main textual information:
      - title, authors, affiliations
      - abstract
      - body sections
      - figure captions (from figure/div type='figure')
      - table captions (from table/figure type='table'/div type='table')
      - references

    If parsing fails, fall back to raw TEI text.
    """
    parsed = parse_tei_full(tei_xml)
    fulltext = parsed.get("fulltext", "").strip()
    if fulltext:
        return fulltext
    return tei_xml.strip()


def read_pdf_text_grobid(
    path: Path,
    grobid_url: Optional[str] = None,
    max_pages: Optional[int] = None,
    with_tables: bool = True,
    include_layout_figures: bool = True,
    include_figure_ocr: bool = False,
    figure_ocr_dir: Optional[Path] = None,
) -> str:
    """
    Usa GROBID per ottenere il TEI e lo converte in testo piano tramite parse_tei_full.
    Opzionalmente:
      - aggiunge le tabelle estratte con Camelot (with_tables=True),
      - aggiunge didascalie di figura estratte dal layout PDF
        (include_layout_figures=True),
      - aggiunge testo OCR delle immagini delle figure
        (include_figure_ocr=True, salvando le immagini in figure_ocr_dir se fornito).
    """
    tei = _grobid_fulltext_tei(path, grobid_url=grobid_url)
    if not tei:
        return ""

    text = _tei_to_plain_text(tei)

    # Tabelle come prima
    if with_tables:
        tb = _tables_text(path, max_pages)
        if tb:
            text = f"{text}\n\n{tb}"

    # Figure captions dal layout PDF
    if include_layout_figures:
        captions = extract_figure_captions_from_pdftext(path, max_pages=max_pages)
        if captions:
            lines = []
            for cap in captions:
                label = cap["label"] or "Figure"
                lines.append(
                    f"{label} (page {cap['page']}): {cap['caption']}"
                )
            text = (
                f"{text}\n\n=== FIGURES (captions from PDF layout) ===\n"
                + "\n\n".join(lines)
            )

    # OCR sulle immagini delle figure
    if include_figure_ocr:
        ocr_dir = figure_ocr_dir
        if ocr_dir is not None and not isinstance(ocr_dir, Path):
            ocr_dir = Path(ocr_dir)
        ocr_results = extract_figure_images_and_ocr(
            path, out_dir=ocr_dir, max_pages=max_pages
        )
        if ocr_results:
            lines = []
            for res in ocr_results:
                if not res["ocr_text"]:
                    continue
                lines.append(
                    f"[page {res['page']} image {res['image_index']}] OCR:\n{res['ocr_text']}"
                )
            if lines:
                text = (
                    f"{text}\n\n=== FIGURES (OCR from images) ===\n"
                    + "\n\n".join(lines)
                )

    return text.strip()



def grobid_extract_all(
    pdf_path: Path,
    grobid_url: Optional[str] = None,
    max_pages: Optional[int] = None,
    with_tables: bool = True,
) -> Dict[str, Any]:
    """
    Full structured extraction for a single PDF using GROBID + Camelot.

    Returns a dict with:
      - 'title'
      - 'authors'
      - 'affiliations'
      - 'abstract'
      - 'sections'
      - 'figures'
      - 'tables'        (captions from TEI)
      - 'references'
      - 'fulltext'
      - 'raw_tei'
      - 'tables_csv'    (list of CSV strings for tables from Camelot, if any)
    """
    tei = _grobid_fulltext_tei(pdf_path, grobid_url=grobid_url)
    if not tei:
        return {}

    parsed = parse_tei_full(tei)
    parsed["raw_tei"] = tei

    # Optional: attach Camelot CSV tables as well
    tables_csv: List[str] = []
    if with_tables:
        tb_block = _tables_text(pdf_path, max_pages)
        if tb_block:
            tables_csv.append(tb_block)
    parsed["tables_csv"] = tables_csv

    return parsed




def text_convert_directory(
    pdf_dir: Path,
    out_dir: Path,
    docs_type: str = "generic",
    grobid_url: Optional[str] = None,
    max_pages: Optional[int] = None,
    use_ocr_fallback: bool = True,
    with_tables: bool = True,
) -> None:
    """
    Converte tutti i PDF in `pdf_dir` in file di testo semplice.
    Usa:
        - read_pdf_text_grobid  se docs_type == "paper"
        - read_pdf_text         se docs_type == "generic"
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_list = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_list:
        print(f"[INFO] Nessun PDF trovato in {pdf_dir}")
        return

    print(f"[INFO] Conversione testo ({docs_type}) per {len(pdf_list)} PDF in {pdf_dir}")

    for p in pdf_list:
        print(f"[TEXT-CONVERT] Converto {p.name} ...")

        if docs_type == "paper":
            text = read_pdf_text_grobid(
                path=p,
                grobid_url=grobid_url,
                max_pages=max_pages,
                with_tables=with_tables,
            )
            if not text:
                print(f"[WARN] GROBID returned empty text for {p.name}, falling back to generic.")
                text = read_pdf_text(
                    path=p,
                    max_pages=max_pages,
                    use_ocr_fallback=use_ocr_fallback,
                    with_tables=with_tables,
                )
        else:
            text = read_pdf_text(
                path=p,
                max_pages=max_pages,
                use_ocr_fallback=use_ocr_fallback,
                with_tables=with_tables,
            )

        out_path = out_dir / (p.stem + ".txt")
        out_path.write_text(text, encoding="utf-8")
        print(f"[TEXT-CONVERT] Salvato in {out_path}")



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
    docs_type: str = "generic",
    grobid_url: Optional[str] = None,
) -> Dict[str, Any]:
    schema_fields: List[str] = profile["schema_fields"]

    row: Dict[str, Any] = {k: "N/A" for k in schema_fields}
    row["pdf_filename"] = pdf_path.name

    # Se docs_type == "paper", prova prima GROBID, poi fallback alla lettura generica
    if docs_type == "paper":
        text = read_pdf_text_grobid(pdf_path, grobid_url=grobid_url, max_pages=max_pages, with_tables=True)
        if not text:
            print(f"[WARN] GROBID returned empty text for {pdf_path.name}, falling back to generic extractor.")
            text = read_pdf_text(
                pdf_path,
                max_pages=max_pages,
                use_ocr_fallback=True,
                with_tables=True,
                include_layout_figures=True,
                include_figure_ocr=True,
                figure_ocr_dir=True,
            )
    else:
        # comportamento originale ("generic")
        text = read_pdf_text(
            pdf_path,
            max_pages=max_pages,
            use_ocr_fallback=True,
            with_tables=True,
        )
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
    docs_type: str = "generic",
    grobid_url: Optional[str] = None,
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
            docs_type=docs_type,
            grobid_url=grobid_url,
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

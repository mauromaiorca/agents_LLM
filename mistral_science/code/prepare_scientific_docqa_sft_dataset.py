#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_scientific_docqa_sft_dataset.py

Costruisce un dataset di Supervised Fine-Tuning (SFT) per Mistral
a partire dai JSON arricchiti prodotti da parse_documents.py.

Per ogni articolo:
  - Usa le QA globali in llm_enrichment (qa_overall, qa_methods, qa_results, qa_limitations)
  - Usa le QA locali per sezione in llm_section_enrichment[*].section_qa_local (se presenti)
  - Crea esempi in formato "chat-style" (messages: system / user / assistant)

Esempio d'uso:

  python prepare_scientific_docqa_sft_dataset.py \\
      --json-dir cryoEM_json \\
      --output-file mistral_scientific_docqa_sft.jsonl \\
      --max-qa-per-type 6 \\
      --include-section-qa

"""

import argparse
import json
import os
import random
import sys
from typing import Dict, Any, Iterable, List, Optional, Tuple


# -----------------------------------------------------------
# Utilità di base
# -----------------------------------------------------------

def iter_json_files(json_dir: str) -> Iterable[str]:
    """Itera sui file .json nella directory (non ricorsivo)."""
    for name in sorted(os.listdir(json_dir)):
        if name.lower().endswith(".json"):
            yield os.path.join(json_dir, name)


def load_json(path: str) -> Optional[Dict[str, Any]]:
    """Carica un JSON, con gestione semplice degli errori."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Impossibile leggere JSON '{path}': {e}", file=sys.stderr)
        return None


def build_article_context(doc: Dict[str, Any],
                          max_chars: Optional[int] = None) -> str:
    """
    Costruisce il testo di contesto per l'articolo.

    Preferisce:
      - doc["fulltext"], se presente
    altrimenti:
      - TITLE + ABSTRACT + sezioni (title + text)
    """
    title = doc.get("title") or doc.get("llm_enrichment", {}).get("title") or ""
    abstract = doc.get("abstract") or ""

    if "fulltext" in doc and isinstance(doc["fulltext"], str):
        base = doc["fulltext"]
    else:
        parts = []
        if title:
            parts.append(f"=== TITLE ===\n{title}")
        if abstract:
            parts.append(f"=== ABSTRACT ===\n{abstract}")

        sections = doc.get("sections") or []
        for idx, sec in enumerate(sections):
            stitle = (sec.get("title") or "").strip()
            stext = (sec.get("text") or "").strip()
            if not stitle and not stext:
                continue
            header = stitle or f"Section {idx}"
            chunk = f"=== SECTION: {header} ===\n{stext}"
            parts.append(chunk)

        base = "\n\n".join(parts)

    base = base.strip()
    if max_chars is not None and len(base) > max_chars:
        return base[:max_chars] + "\n\n[TRUNCATED]"
    return base


def build_section_context(doc: Dict[str, Any],
                          section_index: int,
                          sec_title: str,
                          sec_text: str,
                          max_chars: Optional[int] = None) -> str:
    """
    Costruisce il contesto per una singola sezione.

    Usa il titolo dell'articolo (se disponibile) + indice/titolo della sezione + testo della sezione.
    """
    art_title = doc.get("title") or doc.get("llm_enrichment", {}).get("title") or ""
    header_lines = []
    if art_title:
        header_lines.append(f"ARTICLE TITLE:\n{art_title}")
    header_lines.append(f"SECTION {section_index}: {sec_title or '[untitled section]'}")

    body = sec_text.strip()
    if max_chars is not None and len(body) > max_chars:
        body = body[:max_chars] + "\n\n[TRUNCATED]"

    return "\n\n".join(header_lines) + "\n\nSECTION TEXT:\n" + body


# -----------------------------------------------------------
# Estrazione delle QA dai JSON arricchiti
# -----------------------------------------------------------

def parse_qa_field(raw: Any) -> List[Dict[str, str]]:
    """
    'raw' dovrebbe essere una stringa JSON che contiene un array di oggetti
    con chiavi 'question' e 'answer'. Se non è valido, restituisce [].
    """
    if raw is None:
        return []
    if not isinstance(raw, str):
        return []

    raw = raw.strip()
    if not raw or raw.upper() == "N/A":
        return []

    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            return []
        out = []
        for item in data:
            if not isinstance(item, dict):
                continue
            q = (item.get("question") or "").strip()
            a = (item.get("answer") or "").strip()
            if q and a:
                out.append({"question": q, "answer": a})
        return out
    except Exception:
        # Se il campo è danneggiato o non JSON, ignoriamo
        return []


def iter_article_level_qa(doc: Dict[str, Any],
                          max_per_type: Optional[int] = None) -> Iterable[Tuple[str, str, str]]:
    """
    Ritorna triple (qa_type, question, answer) per il livello di articolo.

      qa_type ∈ {"overall", "methods", "results", "limitations"}
    """
    llm = doc.get("llm_enrichment") or {}
    qa_map = {
        "overall": llm.get("qa_overall"),
        "methods": llm.get("qa_methods"),
        "results": llm.get("qa_results"),
        "limitations": llm.get("qa_limitations"),
    }

    for qa_type, raw in qa_map.items():
        qa_list = parse_qa_field(raw)
        if not qa_list:
            continue

        if max_per_type is not None and len(qa_list) > max_per_type:
            qa_list = qa_list[:max_per_type]

        for item in qa_list:
            yield (qa_type, item["question"], item["answer"])


def _guess_section_enrichment_container(doc: Dict[str, Any]) -> Optional[Any]:
    """
    Prova a trovare il contenitore delle arricchiture di sezione.
    Di default usa 'llm_section_enrichment', ma è flessibile.
    """
    for key in ["llm_section_enrichment", "section_enrichment", "sections_enrichment"]:
        if key in doc:
            return doc[key]
    return None


def iter_section_level_qa(doc: Dict[str, Any],
                          max_qa_per_section: Optional[int] = None) -> Iterable[Tuple[int, str, str, str, str]]:
    """
    Ritorna (section_index, section_title, section_text, question, answer)
    per tutte le sezioni che hanno section_qa_local.

    Gestisce due possibili forme:
      - dict: { "0": { ... }, "1": { ... }, ... }
      - list: [ { ... }, { ... }, ... ]
    """
    sections_meta = doc.get("sections") or []
    sec_enrich = _guess_section_enrichment_container(doc)
    if sec_enrich is None:
        return

    def get_sec_title_text(idx: int) -> Tuple[str, str]:
        if 0 <= idx < len(sections_meta):
            sec = sections_meta[idx]
            return (sec.get("title") or "").strip(), (sec.get("text") or "").strip()
        else:
            return "", ""

    # Caso: dict con chiavi indice
    if isinstance(sec_enrich, dict):
        for key, value in sec_enrich.items():
            try:
                idx = int(key)
            except Exception:
                # Se la chiave non è un intero, saltiamo
                continue
            if not isinstance(value, dict):
                continue

            raw_qa = value.get("section_qa_local") or value.get("qa_local")
            qa_list = parse_qa_field(raw_qa)
            if not qa_list:
                continue

            if max_qa_per_section is not None and len(qa_list) > max_qa_per_section:
                qa_list = qa_list[:max_qa_per_section]

            sec_title, sec_text = get_sec_title_text(idx)
            for item in qa_list:
                yield (idx, sec_title, sec_text, item["question"], item["answer"])

    # Caso: lista allineata con sections
    elif isinstance(sec_enrich, list):
        for idx, value in enumerate(sec_enrich):
            if not isinstance(value, dict):
                continue
            raw_qa = value.get("section_qa_local") or value.get("qa_local")
            qa_list = parse_qa_field(raw_qa)
            if not qa_list:
                continue

            if max_qa_per_section is not None and len(qa_list) > max_qa_per_section:
                qa_list = qa_list[:max_qa_per_section]

            sec_title, sec_text = get_sec_title_text(idx)
            for item in qa_list:
                yield (idx, sec_title, sec_text, item["question"], item["answer"])

    # Altri tipi: ignoriamo
    else:
        return


# -----------------------------------------------------------
# Costruzione delle conversazioni (messages)
# -----------------------------------------------------------

ARTICLE_TEMPLATES = [
    "You are an expert assistant answering questions about the following scientific article. "
    "Use only the information in the article text.",

    "You are a domain expert. Read the article below and answer the question as precisely as possible, "
    "using only what is stated in the text.",

    "Answer the question based solely on the scientific article provided."
]

SECTION_TEMPLATES = [
    "You are an expert assistant. You are given a single section of a scientific article. "
    "Answer the question using only this section.",

    "Consider the following section from a research article. Answer the question based strictly on this text.",

    "Use only the information in this section to answer the question."
]


def build_article_qa_messages(title: str,
                              article_context: str,
                              question: str,
                              answer: str,
                              qa_type: str) -> Dict[str, Any]:
    """
    Costruisce un oggetto 'messages' per una QA a livello di articolo.
    """
    system_content = random.choice(ARTICLE_TEMPLATES)

    user_parts = [
        f"TITLE:\n{title}" if title else "",
        "ARTICLE TEXT:",
        article_context.strip(),
        "",
        f"QUESTION ({qa_type}):",
        question.strip(),
    ]
    user_content = "\n".join(p for p in user_parts if p.strip())

    assistant_content = answer.strip()

    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def build_section_qa_messages(section_context: str,
                              question: str,
                              answer: str,
                              section_index: int) -> Dict[str, Any]:
    """
    Costruisce un oggetto 'messages' per una QA a livello di sezione.
    """
    system_content = random.choice(SECTION_TEMPLATES)

    user_parts = [
        f"This is section index {section_index} of a scientific article.",
        "",
        section_context.strip(),
        "",
        "QUESTION:",
        question.strip(),
    ]
    user_content = "\n".join(p for p in user_parts if p.strip())

    assistant_content = answer.strip()

    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


# -----------------------------------------------------------
# Pipeline principale
# -----------------------------------------------------------

def build_docqa_sft_dataset(json_dir: str,
                            output_file: str,
                            max_article_context_chars: int = 8000,
                            max_section_context_chars: int = 4000,
                            max_qa_per_type: Optional[int] = None,
                            max_qa_per_section: Optional[int] = None,
                            include_section_qa: bool = True,
                            seed: int = 42) -> None:
    """
    Costruisce il dataset SFT e lo salva in formato JSONL.
    """
    random.seed(seed)
    total_examples = 0
    total_docs = 0

    with open(output_file, "w", encoding="utf-8") as fout:
        for path in iter_json_files(json_dir):
            doc = load_json(path)
            if doc is None:
                continue

            total_docs += 1
            pdf_name = os.path.basename(path)

            llm_enrich = doc.get("llm_enrichment")
            if not isinstance(llm_enrich, dict):
                print(f"[WARN] '{pdf_name}' non ha 'llm_enrichment'; salto.", file=sys.stderr)
                continue

            title = llm_enrich.get("title") or doc.get("title") or ""

            # Costruisci contesto articolo una sola volta (per efficienza)
            article_context = build_article_context(doc, max_chars=max_article_context_chars)

            # -------- QA a livello di articolo --------
            for qa_type, q, a in iter_article_level_qa(doc, max_per_type=max_qa_per_type):
                example = build_article_qa_messages(
                    title=title,
                    article_context=article_context,
                    question=q,
                    answer=a,
                    qa_type=qa_type,
                )
                fout.write(json.dumps(example, ensure_ascii=False))
                fout.write("\n")
                total_examples += 1

            # -------- QA a livello di sezione --------
            if include_section_qa:
                for sec_idx, sec_title, sec_text, q, a in iter_section_level_qa(
                    doc, max_qa_per_section=max_qa_per_section
                ):
                    if not sec_text.strip():
                        continue
                    section_context = build_section_context(
                        doc,
                        section_index=sec_idx,
                        sec_title=sec_title,
                        sec_text=sec_text,
                        max_chars=max_section_context_chars,
                    )
                    example = build_section_qa_messages(
                        section_context=section_context,
                        question=q,
                        answer=a,
                        section_index=sec_idx,
                    )
                    fout.write(json.dumps(example, ensure_ascii=False))
                    fout.write("\n")
                    total_examples += 1

    print(f"[INFO] JSON dir      : {json_dir}")
    print(f"[INFO] Output file   : {output_file}")
    print(f"[INFO] Documenti letti: {total_docs}")
    print(f"[INFO] Esempi SFT     : {total_examples}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara un dataset SFT per Mistral (scientific document Q&A) dai JSON arricchiti."
    )
    parser.add_argument(
        "--json-dir",
        type=str,
        required=True,
        help="Directory contenente i JSON arricchiti (output di parse_documents.py).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="mistral_scientific_docqa_sft.jsonl",
        help="Percorso del file JSONL di output per il dataset SFT.",
    )
    parser.add_argument(
        "--max-article-context-chars",
        type=int,
        default=8000,
        help="Numero massimo di caratteri del contesto articolo (per ridurre i token).",
    )
    parser.add_argument(
        "--max-section-context-chars",
        type=int,
        default=4000,
        help="Numero massimo di caratteri del contesto sezione.",
    )
    parser.add_argument(
        "--max-qa-per-type",
        type=int,
        default=None,
        help="Numero massimo di QA per tipo (overall/methods/results/limitations) per articolo. "
             "Se None, usa tutte.",
    )
    parser.add_argument(
        "--max-qa-per-section",
        type=int,
        default=None,
        help="Numero massimo di QA locali per sezione. Se None, usa tutte.",
    )
    parser.add_argument(
        "--include-section-qa",
        action="store_true",
        help="Se presente, include anche le QA a livello di sezione.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed per random (per scegliere i template di prompt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_docqa_sft_dataset(
        json_dir=args.json_dir,
        output_file=args.output_file,
        max_article_context_chars=args.max_article_context_chars,
        max_section_context_chars=args.max_section_context_chars,
        max_qa_per_type=args.max_qa_per_type,
        max_qa_per_section=args.max_qa_per_section,
        include_section_qa=args.include_section_qa,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


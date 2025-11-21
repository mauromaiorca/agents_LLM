#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Profile refinement agent for marine valuation YAML.

This script:
  - loads a base YAML profile (e.g. marine_valuation.yml)
  - loads a CSV with extracted data (e.g. extractions_from_pdfs.csv)
  - optionally loads analysis_instructions.yml
  - computes diagnostics on the CSV (missingness, flat scores, plausibility)
  - asks an LLM to propose an improved YAML profile
  - saves the improved profile to disk as <basename>.yml
  - generates per-file action suggestions and saves them as <basename>.csv

Environment:
  - Uses OPENAI_MODEL and OPENAI_API_KEY from .env, as in pdf_relevance_pipeline.py.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import yaml
import pandas as pd

from dotenv import load_dotenv, find_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import re

# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------
load_dotenv(find_dotenv(usecwd=True))

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in environment or .env")


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model_name=OPENAI_MODEL,
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )


# ----------------------------------------------------------------------
# Loading helpers
# ----------------------------------------------------------------------
def load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must be a mapping")
    return data


def load_analysis_instructions(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "analysis_goal": (
                "Improve the YAML profile for marine valuation so that "
                "future extractions are more discriminant, complete and consistent."
            )
        }
    return load_yaml(path)


# ----------------------------------------------------------------------
# CSV diagnostics
# ----------------------------------------------------------------------
def summarize_missingness(df: pd.DataFrame) -> str:
    lines: List[str] = []
    n = len(df)
    if n == 0:
        return "The CSV is empty; no diagnostics can be computed."

    # NaN
    mask_na = df.isna()
    # literal "N/A"
    str_df = df.astype(str)
    mask_na_literal = str_df.apply(lambda col: col.str.strip().eq("N/A"))
    na_counts = (mask_na | mask_na_literal).sum()

    lines.append(f"Total rows: {n}")
    for col in df.columns:
        na = int(na_counts.get(col, 0))
        frac = na / float(n)
        if frac > 0:
            lines.append(f"Column '{col}': {na} missing or 'N/A' ({frac:.2%})")

    return "\n".join(lines)


def summarize_relevance(df: pd.DataFrame) -> str:
    lines: List[str] = []
    if "relevance" in df.columns:
        try:
            rel = df["relevance"].astype(float)
            lines.append(
                f"relevance: min={rel.min()}, max={rel.max()}, "
                f"mean={rel.mean():.3f}, std={rel.std(ddof=0):.3f}"
            )
            if rel.nunique() == 1:
                lines.append(
                    "All relevance values are identical "
                    f"({rel.iloc[0]}). This suggests a flat classifier."
                )
        except Exception:
            lines.append("Could not interpret 'relevance' as numeric.")
    if "relevance_percentage" in df.columns:
        try:
            rp = df["relevance_percentage"].astype(float)
            lines.append(
                f"relevance_percentage: min={rp.min()}, max={rp.max()}, "
                f"mean={rp.mean():.2f}, std={rp.std(ddof=0):.2f}"
            )
            if rp.nunique() == 1:
                lines.append(
                    "All relevance_percentage values are identical "
                    f"({rp.iloc[0]}). This suggests a flat scoring rubric."
                )
        except Exception:
            lines.append("Could not interpret 'relevance_percentage' as numeric.")
    if not lines:
        return "No relevance fields found."
    return "\n".join(lines)


def summarize_plausibility(df: pd.DataFrame) -> str:
    lines: List[str] = []

    # estimate numeric plausibility
    if "estimate" in df.columns:
        non_na = df["estimate"].astype(str).str.strip()
        non_na = non_na[~non_na.isin(["", "N/A"])]
        n = len(non_na)
        bad_numeric = 0
        for v in non_na:
            try:
                float(v.replace(",", ""))
            except Exception:
                bad_numeric += 1
        if n > 0:
            frac_bad = bad_numeric / float(n)
            lines.append(
                f"estimate: {n} non-empty values, {bad_numeric} non-numeric "
                f"({frac_bad:.2%})."
            )

    # year plausibility
    for col in ["year", "year of estimate"]:
        if col in df.columns:
            non_na = df[col].astype(str).str.strip()
            non_na = non_na[~non_na.isin(["", "N/A"])]
            years: List[int] = []
            for v in non_na:
                try:
                    y = int(v[:4])
                    years.append(y)
                except Exception:
                    continue
            if years:
                y_min, y_max = min(years), max(years)
                lines.append(f"{col}: observed year range {y_min}–{y_max}.")
                if y_min < 1950:
                    lines.append(
                        f"{col}: some years earlier than 1950 may be misparsed."
                    )
                if y_max > 2100:
                    lines.append(
                        f"{col}: some years later than 2100 are likely wrong."
                    )

    # units plausibility
    if "u.m. estimate" in df.columns:
        um = df["u.m. estimate"].astype(str).str.strip()
        non_na = um[~um.isin(["", "N/A"])]
        eur_like = non_na.str.contains("EUR", case=False, na=False).sum()
        lines.append(
            f"u.m. estimate: {len(non_na)} non-empty values, "
            f"{eur_like} containing 'EUR'."
        )

    if not lines:
        return "No basic plausibility checks computed."
    return "\n".join(lines)


def sample_high_relevance_rows(df: pd.DataFrame, top_k: int = 20) -> str:
    if "relevance_percentage" not in df.columns:
        return "No relevance_percentage column; cannot sample high-score rows."
    try:
        df_sorted = df.sort_values("relevance_percentage", ascending=False)
    except Exception:
        return "Could not sort by relevance_percentage."
    subset = df_sorted.head(top_k)
    cols = [c for c in subset.columns if c in (
        "pdf_filename",
        "relevance",
        "relevance_percentage",
        "type of study",
        "metrics",
        "year of estimate",
        "u.m. estimate",
        "estimate",
        "description",
        "note",
    )]
    if not cols:
        cols = list(subset.columns)[:10]
    return subset[cols].to_csv(index=False)


def build_diagnostics_text(df: pd.DataFrame) -> str:
    parts = [
        "=== MISSINGNESS SUMMARY ===",
        summarize_missingness(df),
        "",
        "=== RELEVANCE SUMMARY ===",
        summarize_relevance(df),
        "",
        "=== PLAUSIBILITY SUMMARY ===",
        summarize_plausibility(df),
        "",
        "=== SAMPLE OF HIGH-RELEVANCE ROWS ===",
        sample_high_relevance_rows(df, top_k=20),
    ]
    return "\n".join(parts)


# ----------------------------------------------------------------------
# Per-file actions (rule-based)
# ----------------------------------------------------------------------
KEY_FIELDS_FOR_ACTIONS = [
    "metrics",
    "year of estimate",
    "u.m. estimate",
    "estimate",
    "marine alien species",
    "type of study",
]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    s = str(value).strip()
    return s == "" or s == "N/A"



def _row_action_for_file(row: pd.Series) -> Dict[str, Any]:
    """
    Produce a structured per-file action dict with:
    - action_code            : categorical tag for downstream scripts
    - action_text            : human-readable description
    - priority               : high / medium / low
    - n_missing_key_fields   : number of missing economic/species fields
    - missing_key_fields     : list of missing fields
    - estimate_non_numeric   : 0/1
    - fields_to_focus        : fields that need re-extraction
    - agent_instruction      : instruction phrased so an LLM agent can directly use it
    """

    # --- extract relevance ---
    rel = None
    rel_pct = None
    try:
        if "relevance" in row:
            rel = float(row["relevance"])
    except Exception:
        rel = None
    try:
        if "relevance_percentage" in row:
            rel_pct = float(row["relevance_percentage"])
    except Exception:
        rel_pct = None

    # --- compute missing key fields ---
    missing_key_fields: list[str] = []
    for f in KEY_FIELDS_FOR_ACTIONS:
        if f in row.index and _is_missing(row[f]):
            missing_key_fields.append(f)
    n_missing_key_fields = len(missing_key_fields)

    # --- check estimate numeric format ---
    estimate_non_numeric = 0
    if "estimate" in row.index and not _is_missing(row["estimate"]):
        v = str(row["estimate"]).strip()
        try:
            float(v.replace(",", ""))
        except Exception:
            estimate_non_numeric = 1

    # --- default outcomes ---
    action_code = "no_specific_action"
    action_text = (
        "No specific automated action; keep for baseline extraction or manual review if needed."
    )
    priority = "low"
    agent_instruction = ""
    fields_to_focus = "; ".join(missing_key_fields) if missing_key_fields else ""

    # ============================================================
    # RULES (in order of priority)
    # ============================================================

    # 1. HIGH RELEVANCE + MANY MISSING FIELDS
    if rel == 1 and rel_pct is not None and rel_pct >= 80 and n_missing_key_fields >= 4:
        action_code = "high_relevance_fill_key_fields_strict"
        action_text = (
            "High relevance article with many missing key economic fields; "
            "plan a focused re-extraction of core economic/species attributes."
        )
        priority = "high"
        agent_instruction = (
            "Re-run a targeted extraction for this PDF focusing ONLY on the missing fields: "
            f"{fields_to_focus}. Use the improved YAML profile. Examine methods, results, and tables. "
            "Avoid hallucinating values; return N/A if not explicitly present."
        )

    # 2. HIGH RELEVANCE + FEW MISSING FIELDS
    elif rel == 1 and rel_pct is not None and rel_pct >= 80 and 1 <= n_missing_key_fields <= 3:
        action_code = "high_relevance_fill_key_fields_light"
        action_text = (
            "High relevance article with a few missing fields; a light targeted re-extraction is suggested."
        )
        priority = "medium"
        agent_instruction = (
            "Perform a light targeted re-extraction focusing on: "
            f"{fields_to_focus}. Use the retriever to inspect the most relevant paragraphs. "
            "Fill only values explicitly stated in the text."
        )

    # 3. GOOD REFERENCE EXAMPLE
    elif rel == 1 and rel_pct is not None and rel_pct >= 80 and n_missing_key_fields == 0:
        action_code = "good_reference_example"
        action_text = (
            "Good-quality extraction; use as a reference example for refining prompts."
        )
        priority = "medium"
        agent_instruction = (
            "No re-extraction required. Use this PDF as a reference example for prompt tuning "
            "and schema refinement."
        )

    # 4. FALSE NEGATIVE RELEVANCE (rel=0 but file has economic/species info)
    elif rel == 0 and not all(_is_missing(row.get(f, None)) for f in KEY_FIELDS_FOR_ACTIONS):
        action_code = "check_relevance_false_negative"
        action_text = (
            "File marked as not relevant but contains economic/species information; "
            "review relevance criteria."
        )
        priority = "high"
        agent_instruction = (
            "Re-evaluate relevance for this file using a stricter rubric: "
            "check if BOTH marine alien species AND economic valuation appear explicitly. "
            "If yes, set relevance=1 and re-run full extraction."
        )

    # 5. BAD ESTIMATE FORMAT
    if estimate_non_numeric == 1 and action_code == "no_specific_action":
        action_code = "fix_estimate_format"
        action_text = (
            "Estimate field is non-numeric; standardisation required."
        )
        priority = "medium"
        agent_instruction = (
            "Normalise the 'estimate' field so it contains only a numeric value. "
            "Move any currency symbol or unit to the appropriate fields (estimate_currency, estimate_unit)."
        )

    # 6. DEFAULT (NO SPECIFIC ACTION)
    if action_code == "no_specific_action":
        agent_instruction = (
            "No specific action is required. Use this file for baseline extraction "
            "or manual review if needed."
        )

    # ------------------------------------------------------------
    # FINAL STRUCTURED OUTPUT
    # ------------------------------------------------------------
    return {
        "action_code": action_code,
        "action_text": action_text,
        "priority": priority,
        "n_missing_key_fields": n_missing_key_fields,
        "missing_key_fields": "; ".join(missing_key_fields) if missing_key_fields else "",
        "estimate_non_numeric": estimate_non_numeric,
        "fields_to_focus": fields_to_focus,
        "agent_instruction": agent_instruction,
    }


def build_per_file_actions(df: pd.DataFrame) -> pd.DataFrame:
    if "pdf_filename" not in df.columns:
        return pd.DataFrame(
            columns=[
                "pdf_filename",
                "relevance",
                "relevance_percentage",
                "action_code",
                "action_text",
            ]
        )

    if "relevance_percentage" in df.columns:
        df_sorted = df.sort_values("relevance_percentage", ascending=False)
    else:
        df_sorted = df.copy()

    reps = df_sorted.groupby("pdf_filename", as_index=False).first()

    actions: List[Dict[str, Any]] = []
    for _, row in reps.iterrows():
        meta = _row_action_for_file(row)
        actions.append(
            {
                "pdf_filename": row.get("pdf_filename", ""),
                "relevance": row.get("relevance", ""),
                "relevance_percentage": row.get("relevance_percentage", ""),
                **meta,  # unpack structured info
            }
        )

    return pd.DataFrame(actions)


# ----------------------------------------------------------------------
# Agent call: ask LLM to produce better YAML
# ----------------------------------------------------------------------
def build_agent_system_prompt() -> str:
    return (
        "You are an expert assistant in economic valuation metadata design.\n"
        "You are given:\n"
        "  1) an existing YAML profile for a PDF extraction pipeline;\n"
        "  2) a diagnostic report summarising how that profile performs on real data;\n"
        "  3) high-level analysis instructions.\n\n"
        "Your task is to propose an improved YAML profile that keeps the "
        "same overall structure but refines:\n"
        "  - relevance scoring rubric;\n"
        "  - field schema (add/rename/remove fields if justified);\n"
        "  - per-field questions and formatting rules;\n"
        "  - handling of missing values and implausible values;\n"
        "  - representation of multiple valuation estimates per article when needed.\n\n"
        "Output rules:\n"
        "- Return ONLY a valid YAML document for the new profile.\n"
        "- Preserve the top-level keys expected by the pipeline:\n"
        "    profile_name, description, topic, schema_fields, evidence_exclude,\n"
        "    relevance, global_extraction, fields.\n"
        "- Keep the content consistent and machine-readable.\n"
    )


def build_agent_user_prompt(
    base_profile_yaml: str,
    diagnostics: str,
    analysis_instructions: Dict[str, Any],
) -> str:
    instr_json = json.dumps(analysis_instructions, indent=2)
    return (
        "Here is the existing YAML profile (CURRENT_PROFILE):\n\n"
        f"```yaml\n{base_profile_yaml}\n```\n\n"
        "Here is the diagnostic report on the CSV extracted with this profile "
        "(DIAGNOSTICS):\n\n"
        f"```text\n{diagnostics}\n```\n\n"
        "Here are the analysis instructions (ANALYSIS_INSTRUCTIONS) as JSON:\n\n"
        f"```json\n{instr_json}\n```\n\n"
        "Using this information, produce an improved YAML profile that addresses "
        "the issues in DIAGNOSTICS and the goals in ANALYSIS_INSTRUCTIONS.\n"
        "Do not include any explanation outside the YAML.\n"
    )


def refine_profile(
    base_profile_path: Path,
    csv_path: Path,
    analysis_instructions_path: Optional[Path],
    output_profile_path: Path,
    per_file_actions_path: Path,
) -> None:
    # Debug: stampa dove dovrebbe scrivere
    print(f"[DEBUG] Base profile: {base_profile_path.resolve()}")
    print(f"[DEBUG] CSV: {csv_path.resolve()}")
    print(f"[DEBUG] Output YAML: {output_profile_path.resolve()}")
    print(f"[DEBUG] Per-file actions CSV: {per_file_actions_path.resolve()}")

    base_profile_text = base_profile_path.read_text(encoding="utf-8")
    base_profile = yaml.safe_load(base_profile_text)
    if not isinstance(base_profile, dict):
        raise ValueError("Base profile YAML must be a mapping")

    df = pd.read_csv(csv_path)
    diagnostics = build_diagnostics_text(df)
    analysis_instructions = load_analysis_instructions(analysis_instructions_path)

    # Scrive sempre un file di diagnostica per capire che il codice è passato di qui
    diag_path = output_profile_path.with_suffix(".diagnostics.txt")
    diag_path.write_text(diagnostics, encoding="utf-8")
    print(f"[OK] Wrote diagnostics to: {diag_path}")

    # CSV con azioni per file: anche questo viene sempre scritto
    actions_df = build_per_file_actions(df)
    actions_df.to_csv(per_file_actions_path, index=False)
    print(f"[OK] Wrote per-file actions to: {per_file_actions_path}")

    # Prova a chiamare l'LLM per generare un profilo migliorato
    try:
        llm = make_llm()
        sys_prompt = build_agent_system_prompt()
        usr_prompt = build_agent_user_prompt(
            base_profile_yaml=base_profile_text,
            diagnostics=diagnostics,
            analysis_instructions=analysis_instructions,
        )

        msg = llm(
            [SystemMessage(content=sys_prompt), HumanMessage(content=usr_prompt)]
        )
        content = getattr(msg, "content", "").strip()

        # Estrarre solo il blocco YAML se l'LLM ha usato i fence ```yaml ... ```
        yaml_text = content
        if "```" in content:
            m = re.search(r"```(?:yaml)?\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
            if m:
                yaml_text = m.group(1).strip()

        new_profile = yaml.safe_load(yaml_text)
        if not isinstance(new_profile, dict):
            raise ValueError("Parsed YAML is not a mapping")


        required_keys = [
            "profile_name",
            "topic",
            "schema_fields",
            "relevance",
            "global_extraction",
            "fields",
        ]
        missing = [k for k in required_keys if k not in new_profile]
        if missing:
            raise RuntimeError(
                f"Improved profile is missing required keys: {missing}"
            )

        # Se tutto va bene, scrive il profilo migliorato
        output_profile_path.write_text(
            yaml.safe_dump(new_profile, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print(f"[OK] Wrote improved profile to: {output_profile_path}")

    except Exception as e:
        # Fallback: scrive almeno il profilo originale, così hai SEMPRE un file
        print("[WARN] Failed to generate improved profile with LLM.")
        print(f"[WARN] Error was: {e}")
        print("[WARN] Writing base profile as fallback.")

        output_profile_path.write_text(
            yaml.safe_dump(base_profile, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print(f"[OK] Wrote fallback base profile to: {output_profile_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "Refine a marine valuation YAML profile using CSV diagnostics and "
            "produce per-file action suggestions."
        )
    )
    ap.add_argument(
        "--base-profile",
        default="marine_valuation.yml",
        help="Path to base YAML profile (default: marine_valuation.yml)",
    )
    ap.add_argument(
        "--csv",
        default="extractions_from_pdfs.csv",
        help="CSV with extracted data (default: extractions_from_pdfs.csv)",
    )
    ap.add_argument(
        "--instructions",
        default="analysis_instructions.yml",
        help="Optional analysis instructions YAML (default: analysis_instructions.yml)",
    )
    ap.add_argument(
        "--o-basename",
        default="better_marine_valuation",
        help=(
            "Base name for outputs. The script will write "
            "<basename>.yml (improved profile), "
            "<basename>.csv (per-file actions), and "
            "<basename>.diagnostics.txt. "
            "Default: better_marine_valuation"
        ),
    )

    args = ap.parse_args()

    base_profile_path = Path(args.base_profile)
    csv_path = Path(args.csv)
    instr_path = Path(args.instructions)
    basename = Path(args.o_basename)

    if not base_profile_path.is_file():
        raise SystemExit(f"[ERROR] Base profile not found: {base_profile_path}")
    if not csv_path.is_file():
        raise SystemExit(f"[ERROR] CSV file not found: {csv_path}")
    if not instr_path.is_file():
        print(f"[WARN] Instructions file not found, using defaults: {instr_path}")
        instr_path = None

    out_profile_path = basename.with_suffix(".yml")
    per_file_actions_path = basename.with_suffix(".csv")

    refine_profile(
        base_profile_path=base_profile_path,
        csv_path=csv_path,
        analysis_instructions_path=instr_path,
        output_profile_path=out_profile_path,
        per_file_actions_path=per_file_actions_path,
    )


if __name__ == "__main__":
    main()


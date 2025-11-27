#!/usr/bin/env python

import os
import json
import argparse
import textwrap

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

BASE_SYSTEM_PROMPT = (
    "You are a domain-specific scientific question-answering assistant. "
    "You receive a question about a scientific article and, optionally, "
    "context extracted from that article. Use only the given context to answer. "
    "If the answer is not clearly contained in the context, reply with: "
    "\"I do not know based on the provided context.\""
)


def load_context_from_file(path, max_chars=20000):
    """
    Load context from a JSON or plain text file and truncate to max_chars.
    """
    if path is None:
        return ""

    if not os.path.exists(path):
        raise FileNotFoundError(f"Context file not found: {path}")

    _, ext = os.path.splitext(path)
    ext = ext.lower()

    try:
        if ext == ".json":
            with open(path, "r") as f:
                obj = json.load(f)
            parts = []

            # Title
            title = obj.get("title") or obj.get("llm_enrichment", {}).get("title")
            if title:
                parts.append(f"TITLE: {title}")

            # Abstract or enriched summary
            abstract = obj.get("abstract") or obj.get("llm_enrichment", {}).get("summary_three_sentences")
            if abstract:
                parts.append(f"ABSTRACT_OR_SUMMARY: {abstract}")

            # Sections
            sections = obj.get("sections") or []
            for sec in sections:
                sec_title = (sec.get("title") or "").strip()
                sec_text = (sec.get("text") or "").strip()

                # If raw section text is empty, try enrichment fields
                if not sec_text:
                    enr = sec.get("llm_enrichment") or sec.get("section_enrichment") or {}
                    sec_text = (
                        enr.get("section_key_ideas")
                        or enr.get("section_methods_summary")
                        or enr.get("section_results_summary")
                        or ""
                    )

                if sec_title or sec_text:
                    shortened = textwrap.shorten(sec_text, width=1000, placeholder=" ...")
                    parts.append(f"SECTION: {sec_title}\n{shortened}")

            context = "\n\n".join(parts)
        else:
            # Plain text file
            with open(path, "r") as f:
                context = f.read()
    except Exception as e:
        raise RuntimeError(f"Error reading context file {path}: {e}")

    context = context.strip()
    if len(context) > max_chars:
        context = context[:max_chars]
    return context


def build_messages(question, context):
    if context:
        user_content = (
            "You are given context from a scientific article.\n\n"
            "CONTEXT:\n"
            f"{context}\n\n"
            "QUESTION:\n"
            f"{question}\n\n"
            "Provide a concise answer in 2–4 sentences, using only the information in the context."
        )
    else:
        user_content = (
            "Answer the following scientific question as well as you can.\n\n"
            "QUESTION:\n"
            f"{question}\n\n"
            "If you are uncertain, say that you do not know."
        )

    messages = [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages


def main(args):
    # GPU selection
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"[INFO] Using GPU index {args.gpu} via CUDA_VISIBLE_DEVICES")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    print(f"[INFO] Adapter dir     : {args.adapter_dir}")
    print(f"[INFO] Base model      : {MODEL_NAME}")
    print(f"[INFO] Max input tokens: {args.max_input_tokens}")
    print(f"[INFO] Max new tokens  : {args.max_new_tokens}")
    print(f"[INFO] Temperature     : {args.temperature}")
    print(f"[INFO] Top-p           : {args.top_p}")

    # Load context (if provided)
    context = ""
    if args.context_file is not None:
        print(f"[INFO] Loading context from: {args.context_file}")
        context = load_context_from_file(args.context_file, max_chars=args.max_context_chars)
        print(f"[INFO] Context length (chars): {len(context)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base model + LoRA, or base only (debug)
    if args.base_only:
        print("[INFO] Loading base model only (no LoRA adapter)")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
    else:
        print("[INFO] Loading base model and LoRA adapter...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    model.eval()
    if device == "cpu":
        model.to("cpu")

    # Build chat-style prompt
    messages = build_messages(args.question, context)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenise with truncation
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_tokens,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    prompt_len = input_ids.shape[1]
    print(f"[INFO] Prompt tokens: {prompt_len}")
    
    effective_max_length = prompt_len + args.max_new_tokens

    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": args.max_new_tokens,
        "max_length": effective_max_length,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    # Sampling vs greedy
    if args.temperature > 0.0:
        gen_kwargs.update(
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    else:
        gen_kwargs.update(
            do_sample=False,
        )

    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)

    # Only tokens after the prompt
    generated_ids = output_ids[0, prompt_len:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print("\n================= ANSWER =================\n")
    if answer:
        print(answer)
    else:
        # If slice is empty, show full output for debugging
        full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("[WARN] Empty answer slice; showing full decoded output instead:\n")
        print(full_output)
    print("\n==========================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query a LoRA-fine-tuned Mistral-7B model for scientific document QA."
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        required=True,
        help="Directory with the saved LoRA adapter (output of training script).",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="User question to ask.",
    )
    parser.add_argument(
        "--context-file",
        type=str,
        default=None,
        help="Optional path to a JSON or text file containing article context.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU index to use (e.g. 0 or 1). If not set, use all visible GPUs / CPU.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens for the prompt (context + question).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate for the answer.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature. Use 0.0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter when temperature > 0.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=20000,
        help="Maximum number of characters to load from the context file.",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="If set, ignore the adapter and use the base model only (for debugging).",
    )

    args = parser.parse_args()
    main(args)


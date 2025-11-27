#!/usr/bin/env python
"""
LoRA fine-tuning of Mistral-7B-Instruct-v0.3 on scientific DocQA conversations.

Expected input: a JSONL file where each line is a JSON object with at least:
  {
    "messages": [
      {"role": "system"|"user"|"assistant", "content": "..."},
      ...
    ]
  }

We assume that the **last** message is the assistant answer we want to learn,
and all previous messages form the context (system + user + eventual previous turns).

We train only on assistant tokens (prompt tokens are masked with -100 in labels).
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model


DEFAULT_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_OUTPUT_DIR = "./mistral-docqa-lora"
DEFAULT_MAX_LENGTH = 2048


# ---------------------------------------------------------------------
# Utilities for loading SFT data
# ---------------------------------------------------------------------


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    samples = []
    with open(path, "r") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                samples.append(obj)
            except Exception as e:
                print(f"[SKIP] Line {i}: JSON parse error: {e}")
    print(f"[INFO] Loaded {len(samples)} samples from {path}")
    return samples


# ---------------------------------------------------------------------
# Conversation → (input_ids, labels) for Mistral
# ---------------------------------------------------------------------


def build_prompt_and_labels_for_sample(
    example: Dict[str, Any],
    tokenizer,
    max_length: int,
) -> Optional[Dict[str, Any]]:
    """
    Convert one sample with "messages" into input_ids, attention_mask, labels.

    Strategy:
      - Use all messages except the last as context.
      - Use last message (must be assistant) as supervised target.
      - If a chat template exists (Mistral), we use it.
      - Otherwise we fall back to a simple "System:\n... User:\n... Assistant:" style.

    We enforce max_length **without** truncating the answer: if the full sequence
    would exceed max_length, we drop the sample.
    """

    msgs: List[Dict[str, str]] = example.get("messages", [])
    if len(msgs) < 2:
        # Too short to be meaningful
        return None

    if msgs[-1].get("role") != "assistant":
        # We only train on examples whose last message is an assistant answer.
        return None

    context_messages = msgs[:-1]
    answer_text = msgs[-1]["content"].strip()

    if not answer_text:
        return None

    # --- Try to use tokenizer chat template if available --------------------
    has_chat_template = getattr(tokenizer, "chat_template", None) is not None

    if has_chat_template:
        # Build the prompt including all context messages, leaving the answer out.
        # add_generation_prompt=True appende il tag per l'assistant ma non il contenuto.
        prompt_text = tokenizer.apply_chat_template(
            context_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # Tokenize prompt and answer separately (no truncation, no extra specials).
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        answer_ids = tokenizer(
            answer_text + tokenizer.eos_token,
            add_special_tokens=False,
        ).input_ids

    else:
        # Fallback manual template if no chat_template is defined
        # We encode the conversation in a simple readable way.
        prompt_parts = []
        for m in context_messages:
            role = m.get("role", "user")
            content = (m.get("content") or "").strip()
            if not content:
                continue
            # Normalise role label a bit
            if role == "system":
                prefix = "System"
            elif role == "user":
                prefix = "User"
            elif role == "assistant":
                prefix = "Assistant"
            else:
                prefix = role.capitalize()
            prompt_parts.append(f"{prefix}:\n{content}")

        prompt_text = "\n\n".join(prompt_parts) + "\n\nAssistant:\n"
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        answer_ids = tokenizer(
            answer_text + tokenizer.eos_token,
            add_special_tokens=False,
        ).input_ids

    # -------------------------------------------------------------
    # Check total length.
    # Se troppo lungo:
    #   - prima proviamo a tagliare il PROMPT dalla testa,
    #   - se perfino la sola risposta è troppo lunga, teniamo
    #     solo la coda della risposta (e scartiamo il prompt).
    # -------------------------------------------------------------
    total_length = len(prompt_ids) + len(answer_ids)

    if total_length > max_length:
        # spazio massimo per il prompt, tenendo intera la risposta
        max_prompt_len = max_length - len(answer_ids)

        if max_prompt_len <= 0:
            # Caso estremo: la risposta da sola eccede max_length.
            # Tenere SOLO la coda della risposta, lunga max_length - 1,
            # e nessun prompt.
            answer_ids = answer_ids[-(max_length - 1):]
            prompt_ids = []
        else:
            # Tagliare il prompt dalla testa (left-truncation),
            # tenendo gli ultimi max_prompt_len token.
            if len(prompt_ids) > max_prompt_len:
                prompt_ids = prompt_ids[-max_prompt_len:]

        total_length = len(prompt_ids) + len(answer_ids)
        # a questo punto total_length <= max_length garantito


    input_ids = prompt_ids + answer_ids
    # Mask prompt tokens in labels
    labels = [-100] * len(input_ids)
    for i in range(len(prompt_ids), len(input_ids)):
        labels[i] = input_ids[i]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def preprocess_dataset(
    raw_dataset: Dataset,
    tokenizer,
    max_length: int,
) -> Dataset:
    """
    Map the raw dataset (with 'messages') to tokenized examples suitable for Trainer.
    """

    def _map_fn(example):
        out = build_prompt_and_labels_for_sample(
            example,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        if out is None:
            # Return empty; we'll filter these out
            return {
                "input_ids": [],
                "labels": [],
                "attention_mask": [],
            }
        return out

    tokenized = raw_dataset.map(
        _map_fn,
        remove_columns=raw_dataset.column_names,
    )

    # Filter out empty ones
    tokenized = tokenized.filter(lambda e: len(e["input_ids"]) > 0)

    print(f"[INFO] Tokenized dataset size: {len(tokenized)}")
    return tokenized


# ---------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="LoRA SFT of Mistral-7B-Instruct on scientific DocQA conversations."
    )
    parser.add_argument(
        "--sft-data",
        type=str,
        required=True,
        help="Path to JSONL SFT dataset with a 'messages' field.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Base model name or path (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where to save the LoRA adapters and tokenizer (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"Maximum sequence length (default: {DEFAULT_MAX_LENGTH})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device train batch size (default: 1)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8)",
    )
    parser.add_argument(
        "--num-epochs",
        type=float,
        default=2.0,
        help="Number of training epochs (default: 2.0)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Warmup ratio for LR scheduler (default: 0.03)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.0,
        help="Fraction of data used for validation (0.0 = no validation, default: 0.0)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500)",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log training metrics every N steps (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU index to use (e.g. 0 or 1). If not set, use all visible GPUs.",
    )

    args = parser.parse_args()

    # --- GPU selection ------------------------------------------------------
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"[INFO] Using GPU index {args.gpu} via CUDA_VISIBLE_DEVICES")

    # --- Seed ---------------------------------------------------------------
    set_seed(args.seed)

    # --- Load data ----------------------------------------------------------
    samples = load_jsonl(args.sft_data)
    if not samples:
        raise RuntimeError(f"No data loaded from {args.sft_data}")

    raw_dataset = Dataset.from_list(samples)
    print(f"[INFO] Raw dataset size: {len(raw_dataset)}")

    # --- Tokenizer ----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Preprocess dataset (tokenize + label masking) ----------------------
    tokenized_dataset = preprocess_dataset(
        raw_dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    if args.val_size and args.val_size > 0.0 and args.val_size < 1.0:
        split = tokenized_dataset.train_test_split(
            test_size=args.val_size,
            seed=args.seed,
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(
            f"[INFO] Train size: {len(train_dataset)}, "
            f"Eval size: {len(eval_dataset)}"
        )
    else:
        train_dataset = tokenized_dataset
        eval_dataset = None
        print(f"[INFO] Using full dataset for training only: {len(train_dataset)} examples")

    # --- Load base model in fp16 on GPU ------------------------------------
    print(f"[INFO] Loading base model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Disabilita la cache per il training
    if hasattr(model, "config"):
        model.config.use_cache = False

    # --- LoRA configuration -------------------------------------------------
    # You can tune target_modules; for Mistral it's common to include attention and MLP projections.
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Training arguments -------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=False,  # modello è già in float16; niente AMP/GradScaler per evitare problemi
        optim="adamw_torch",
        report_to="none",
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.save_steps if eval_dataset is not None else None,
        load_best_model_at_end=bool(eval_dataset is not None),
        remove_unused_columns=False,  # important when we control inputs/labels explicitly
    )

    # Optional: gradient checkpointing for memory (slows training, but can help)
    # model.gradient_checkpointing_enable()

    # --- Trainer ------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    print("[INFO] Starting training...")
    trainer.train()
    print("[INFO] Training finished.")

    # --- Save LoRA adapters + tokenizer ------------------------------------
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[INFO] LoRA adapters and tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    main()


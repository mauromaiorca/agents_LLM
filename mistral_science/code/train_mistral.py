
#!/usr/bin/env python

import torch
import bitsandbytes as bnb
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DATA_PATTERN = "data/*.json"
OUTPUT_DIR = "./mistral-finetuned-json"
BLOCK_SIZE = 2048

def build_text(example):
    parts = []

    title = example.get("title") or ""
    abstract = example.get("abstract") or ""
    sections = example.get("sections") or []

    if title:
        parts.append("[TITLE] " + title)

    if abstract:
        parts.append("[ABSTRACT] " + abstract)

    for sec in sections:
        sec_title = sec.get("title") or ""
        sec_text = sec.get("text") or ""
        if sec_title or sec_text:
            parts.append(f"[SECTION] {sec_title}\n{sec_text}")

    full_text = "\n\n".join(parts)
    return {"text": full_text}


def main():
    # 1. Carica tutti i json
    dataset = load_dataset(
        "json",
        data_files={"train": DATA_PATTERN},
        split="train"
    )

    # 2. Costruisci un campo "text" unico per articolo
    dataset_text = dataset.map(build_text)

    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=False
        )

    tokenized = dataset_text.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset_text.column_names
    )

    # 4. Raggruppa i token in blocchi fissi
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
        result = {
            k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized.map(group_texts, batched=True)

    # 5. Carica il modello in 4-bit (BitsAndBytesConfig)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config,
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1.0,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=200,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()

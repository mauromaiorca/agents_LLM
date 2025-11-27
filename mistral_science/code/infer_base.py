#!/usr/bin/env python

import os
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"


def main(args):
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"[INFO] Using GPU index {args.gpu} via CUDA_VISIBLE_DEVICES")

    print(f"[INFO] Loading base model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    print("[INFO] Ready. Type a prompt, or 'quit' to exit.")

    while True:
        try:
            user_input = input("\n>> ")
        except EOFError:
            break

        if not user_input.strip():
            continue
        if user_input.strip().lower() in {"quit", "exit", "q"}:
            break

        inputs = tokenizer(
            user_input,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=not args.greedy,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n[MODEL]\n")
        print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference with base Mistral-7B."
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU index to use (e.g. 0 or 1).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling p.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding instead of sampling.",
    )

    args = parser.parse_args()
    main(args)


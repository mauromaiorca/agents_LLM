# Mistral science


## Installa
mkdir ~/mistral_science && cd ~/mistral_science
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121

Verifica da Python:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

pip install "transformers>=4.40.0" "datasets>=2.19.0" "accelerate>=0.30.0"
pip install "peft>=0.10.0" bitsandbytes sentencepiece einops

pip install  transformers==4.40.0 "datasets>=2.19.0" "accelerate==0.30.0"  "peft==0.11.1" bitsandbytes sentencepiece einops
pip install   torch==2.1.0+cu118   torchvision==0.16.0+cu118   torchaudio==2.1.0+cu118   --index-url https://download.pytorch.org/whl/cu118

creare un account su hugging face, e fare
hf auth login

## Prepara i dati
```bash
source .venv/bin/activate
./prepare_scientific_docqa_sft_dataset.py --json-dir cryoEM_enriched_json --output-file mistral_scientific_docqa_sft.jsonl --include-section-qa --max-qa-per-type 6 --max-qa-per-section 6
```


## Esegui il training

```bash
source .venv/bin/activate
./train_mistral_scientific_docqa_lora.py --sft-data mistral_scientific_docqa_sft.jsonl --gpu 1 --output-dir mistral-docqa-lora --max-length 512 --batch-size 1 --grad-accum 4 --num-epochs 2 --learning-rate 2e-4
```

```bash
source .venv/bin/activate
./train_mistral.py --docs data --gpu 1 --block-size 512
```
## Esegui l'inferring

inferring base:

```bash
source .venv/bin/activate
./infer_base.py
```

```bash
source .venv/bin/activate
./query_mistral_scientific_docqa_lora.py --adapter-dir mistral-docqa-lora --gpu 1 --max-input-tokens 1500 --max-new-tokens 256 --temperature 0.0 --context-file data/EMS208934.json --question "what is chemEM?"
```

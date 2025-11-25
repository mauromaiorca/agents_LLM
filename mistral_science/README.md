# Mistral science


## 
mkdir ~/mistral_science && cd ~/mistral_science
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121

Verifica da Python:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

pip install "transformers>=4.40.0" "datasets>=2.19.0" "accelerate>=0.30.0"
pip install "peft>=0.10.0" bitsandbytes sentencepiece einops

pip install \
    "torch" \
    "transformers>=4.40.0" \
    "datasets>=2.19.0" \
    "accelerate>=0.30.0" \
    "peft>=0.10.0" \
    bitsandbytes \
    sentencepiece \
    einops

creare un account su hugging face, e fare
hf auth login

./train_mistral.py

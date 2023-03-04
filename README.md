# Inference LLaMA models using CPU only

This repository is intended as a minimal, hackable and readable example to load [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) ([arXiv](https://arxiv.org/abs/2302.13971v1)) models and run inference by using only CPU. Thus requires no videocard, but 64 (better 128 Gb) of RAM and modern processor is required.

### Conda Environment Setup Example for Windows 10+
Download and install Anaconda Python https://www.anaconda.com
Run Anaconda prompt
```
conda create -n llama python=3.10
conda activate llama
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

### Setup
In a conda env with pytorch / cuda available, run
```
pip install -r requirements.txt
```
Then in this repository
```
pip install -e .
```

### Download tokenizer and models
magnet:?xt=urn:btih:ZXXDAUWYLRUXXBHUYEMS6Q5CE5WA3LVA&dn=LLaMA

### CPU Inference
Place tokenizer.model and tokenizer_checklist.chk into [/tokenizer] folder

Place three files of 7B model into [/model] folder

Run it:
```
python example-cpu.py
```

### Model Card
See [MODEL_CARD.md](MODEL_CARD.md)

### License
See the [LICENSE](LICENSE) file.

# Inference LLaMA models using CPU only

This repository is intended as a minimal, hackable and readable example to load [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) ([arXiv](https://arxiv.org/abs/2302.13971v1)) models and run inference by using only CPU. Thus requires no videocard, but 64 (better 128 Gb) of RAM and modern processor is required.

### Conda Environment Setup Example for Windows 10+
Download and install Anaconda Python https://www.anaconda.com and run Anaconda Prompt
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

### CPU Inference of 7B model
Place tokenizer.model file from torrent into repo's [/tokenizer] folder.

Place consolidated.00.pth and params.json from 7B torrent folder into repo's [/model] folder.

Run the example:
```
python example-cpu.py
```

### CPU Inference of 13B, 30B and 65B models
A little bit tricky part is that we need to unshard the checkpoints first. In this example, D:\Downloads\LLaMA is a root folder of downloaded torrent with models. Run the following command to create merged weights checkpoint:
```
python merge-weights.py --input_dir D:\Downloads\LLaMA --model_size 13B
```
This will create merged.pth file in the repo's root folder. Move this file into [/model] folder.

Place corresponding params.json file (from 13B torrent folder) into repo's [/model] folder.

So, you should end up with two files in [/model] folder: merged.pth and params.json.

Place tokenizer.model file from torrent into repo's [/tokenizer] folder.

Run the example:
```
python example-cpu.py
```

### Model Card
See [MODEL_CARD.md](MODEL_CARD.md)

### License
See the [LICENSE](LICENSE) file.

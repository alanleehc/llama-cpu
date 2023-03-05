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

or

magnet:xt=urn:btih:b8287ebfa04f879b048d4d4404108cf3e8014352&dn=LLaMA&tr=udp%3a%2f%2ftracker.opentrackr.org%3a1337%2fannounce

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

### Interactive chat with LLaMA

```
python example-chat.py
```

### Some measurements

Running model with single prompt on Windows computer equipped with 12700k, fast nvme and 128 Gb of RAM.

| model  | RAM usage, fp32 | RAM usage, bf16 | fp32 inference | bf16 inference | fp32 load model |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 7B   | 44 Gb, peak 56 Gb  | 22 Gb | 170 seconds | 850 seconds | 23 seconds |
| 13B  | 77 Gb, peak 100 Gb | 38 Gb | 340 seconds |  | 61 seconds |
| 30B  | 180 Gb, peak 258 Gb | 89 Gb | 48 minutes | 67 minutes | 372 seconds |

### RAM usage optimization
By default, torch uses Float32 precision while running on CPU, which leads, for example, to use 44 GB of RAM for 7B model. We may use Bfloat16 precision on CPU too, which decreases RAM consumption/2, down to 22 GB for 7B model, but inference processing much slower.

Uncomment this line in the example-cpu.py or example-chat.py to enable Bfloat16 and save memory.
```
torch.set_default_dtype(torch.bfloat16)
```

### Model Card
See [MODEL_CARD.md](MODEL_CARD.md)

### License
See the [LICENSE](LICENSE) file.

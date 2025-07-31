# llm-benchmark

This repository contains a small script to measure the approximate tokens per minute (TPM) for a few open models: **Llama 3.1 8B**, **Qwen 2.5**, and **Gemma 2B**.

The script relies on the [transformers](https://github.com/huggingface/transformers) library and PyTorch. Ensure these dependencies are installed before running the benchmark. Each model will be loaded using `device_map="auto"` to utilize available GPU or CPU resources.

## Usage

```bash
python benchmark.py
```

Optional arguments allow customizing the prompt and number of generated tokens:

```bash
python benchmark.py --prompt "Tell me a story" --tokens 256
```

The script prints the measured TPM for each model or a failure message if the model cannot be loaded.

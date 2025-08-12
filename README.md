# llm-benchmark

This repository contains a small script to measure the approximate tokens per minute
(TPM) for a few open models. By default it benchmarks **Llama 3.1 8B**,
**Qwen 2.5**, and **Gemma 2B**, but any additional model from Hugging Face
can be supplied at the command line.

The script relies on the
[transformers](https://github.com/huggingface/transformers) library and
PyTorch. Ensure these dependencies are installed before running the
benchmark. Each model is loaded with `device_map="auto"` to utilize
available GPU or CPU resources.

## Usage

```bash
python benchmark.py
```

Optional arguments allow customizing the prompt, number of generated tokens,
timed runs, and warm‑up steps:

```bash
python benchmark.py --prompt "Tell me a story" --tokens 256 --runs 3 --warmup 1
```

Additional models can be benchmarked using `--model`, which accepts
`label=repo_id` pairs and may be provided multiple times:

```bash
python benchmark.py --model mytiny=sshleifer/tiny-gpt2 --tokens 32
```

The script prints the measured TPM for each model or a failure message
if the model cannot be loaded.

import argparse
import gc
import time
from typing import Dict, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def measure_model(
    model_name: str,
    prompt: str,
    num_tokens: int = 128,
    runs: int = 1,
    warmup: int = 1,
) -> float:
    """Load a model and measure tokens per minute for short generations."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        for _ in range(warmup):
            model.generate(**inputs, max_new_tokens=1)

    total_generated = 0
    total_time = 0.0

    with torch.inference_mode():
        for _ in range(runs):
            start = time.perf_counter()
            output = model.generate(**inputs, max_new_tokens=num_tokens)
            end = time.perf_counter()
            total_generated += output.shape[-1] - inputs.input_ids.shape[-1]
            total_time += end - start

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if total_time == 0:
        return 0.0
    return (total_generated / total_time) * 60


def main():
    parser = argparse.ArgumentParser(description="Benchmark open LLMs")
    parser.add_argument(
        "--prompt",
        default="Write a short poem about AI.",
        help="Prompt used for generation.",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=128,
        help="Number of tokens to generate for measurement.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of timed runs per model.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warm-up runs before timing.",
    )
    parser.add_argument(
        "--model",
        action="append",
        metavar="LABEL=REPO",
        help=(
            "Model repository to benchmark. "
            "Format: label=repo_id. Can be specified multiple times."
        ),
    )
    args = parser.parse_args()

    default_models = [
        "Llama3-8B=meta-llama/Meta-Llama-3.1-8B",
        "Qwen-2.5=Qwen/Qwen2.5",
        "Gemma-2B=google/gemma-2b",
    ]
    model_entries: Iterable[str] = args.model or default_models
    models: Dict[str, str] = {}
    for entry in model_entries:
        if "=" in entry:
            label, repo = entry.split("=", 1)
        else:
            repo = entry
            label = repo.split("/")[-1]
        models[label] = repo

    for label, repo in models.items():
        try:
            tpm = measure_model(
                repo,
                args.prompt,
                args.tokens,
                runs=args.runs,
                warmup=args.warmup,
            )
            print(f"{label}: {tpm:.2f} tokens/minute")
        except Exception as e:
            print(f"{label}: failed to benchmark ({e})")


if __name__ == "__main__":
    main()

import argparse
import time

from transformers import AutoModelForCausalLM, AutoTokenizer


def measure_model(model_name: str, prompt: str, num_tokens: int = 128) -> float:
    """Load a model and measure tokens per minute for a short generation."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start = time.perf_counter()
    output = model.generate(**inputs, max_new_tokens=num_tokens)
    end = time.perf_counter()

    generated = output.shape[-1] - inputs.input_ids.shape[-1]
    seconds = end - start
    if seconds == 0:
        return 0.0
    tokens_per_minute = (generated / seconds) * 60
    return tokens_per_minute


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
    args = parser.parse_args()

    models = {
        "Llama3-8B": "meta-llama/Meta-Llama-3.1-8B",
        "Qwen-2.5": "Qwen/Qwen2.5",
        "Gemma-2B": "google/gemma-2b",
    }

    for label, repo in models.items():
        try:
            tpm = measure_model(repo, args.prompt, args.tokens)
            print(f"{label}: {tpm:.2f} tokens/minute")
        except Exception as e:
            print(f"{label}: failed to benchmark ({e})")


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "transformers",
#     "accelerate",
# ]
# ///
"""Minimal HF Jobs experiment: load a model and generate.

Run on HF infra (resolves deps on the worker, no local install):
    hf jobs uv run --flavor l4x1 --timeout 30m --secrets HF_TOKEN -d job.py --model Qwen/Qwen3-1.7B

Smoke-test the logic cheaply on CPU first:
    hf jobs uv run --flavor cpu-basic --timeout 10m job.py --model sshleifer/tiny-gpt2
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--prompt", default="The key to fast iteration is")
    p.add_argument("--max-new-tokens", type=int, default=64)
    args = p.parse_args()

    # Verify the hardware the job actually landed on.
    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"loading {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map=device,
    )

    inputs = tok(args.prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    print("=== output ===")
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()

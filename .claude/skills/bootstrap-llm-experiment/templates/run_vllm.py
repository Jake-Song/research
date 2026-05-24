# /// script
# dependencies = [
#     "vllm",
# ]
# ///

"""{{NAME}}: local generation with vLLM."""

from __future__ import annotations

import argparse

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="{{MODEL_ID}}")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    llm = LLM(model=args.model)
    sampling = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    tokenizer = llm.get_tokenizer()
    if tokenizer.chat_template:
        outputs = llm.chat(
            [{"role": "user", "content": args.prompt}],
            sampling_params=sampling,
        )
    else:
        outputs = llm.generate([args.prompt], sampling_params=sampling)

    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()

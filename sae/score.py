import asyncio
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from delphi.clients import OpenRouter
from delphi.explainers import DefaultExplainer
from delphi.latents import (
    ActivatingExample,
    Latent,
    LatentRecord,
    NonActivatingExample,
)
from delphi.scorers import DetectionScorer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "/content/drive/MyDrive/huggingface/models/Qwen3.5-9B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float32,
).to(device)
model.eval()

sae_path = "/content/drive/MyDrive/huggingface/sae/"
LAYERS = range(8, 16)
TOPK = 50

POS = [
    "I love this movie, it was wonderful.",
    "What a fantastic meal, absolutely delicious.",
    "She is such a kind and generous person.",
    "The concert last night was amazing.",
    "I'm so happy with how things turned out.",
    "This book is brilliant and inspiring.",
    "The weather today is beautiful.",
    "He did an excellent job on the project.",
]

NEG = [
    "I hate this movie, it was terrible.",
    "What a disgusting meal, absolutely revolting.",
    "She is such a cruel and selfish person.",
    "The concert last night was awful.",
    "I'm so disappointed with how things turned out.",
    "This book is dreadful and boring.",
    "The weather today is miserable.",
    "He did a horrible job on the project.",
]


def analyze_layer(layer: int) -> dict:
    sae = torch.load(f"{sae_path}layer{layer}.sae.pt", map_location=device)
    W_enc = sae["W_enc"].to(torch.float32)
    b_enc = sae["b_enc"].to(torch.float32)

    captured = {}
    def _hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["residual"] = hidden.detach()
    handle = model.model.layers[layer].register_forward_hook(_hook)

    def last_token_feats(text: str) -> torch.Tensor:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
        residual = captured["residual"].to(torch.float32)
        pre_acts = residual @ W_enc.T + b_enc
        topk_vals, topk_idx = pre_acts.topk(TOPK, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk_idx, topk_vals)
        return acts[0, -1]

    pos = torch.stack([last_token_feats(t) for t in POS])
    neg = torch.stack([last_token_feats(t) for t in NEG])
    handle.remove()

    diff = pos.mean(0) - neg.mean(0)
    top_pos_idx = int(diff.argmax())
    return {
        "layer": layer,
        "top_pos_idx": top_pos_idx,
        "top_pos_diff": float(diff[top_pos_idx]),
    }


results = [analyze_layer(L) for L in LAYERS]
best = max(results, key=lambda r: r["top_pos_diff"])
BEST_LAYER = best["layer"]
FEATURE_IDX = best["top_pos_idx"]
print(f"Best POS-leaning feature: layer {BEST_LAYER}, feature {FEATURE_IDX}, diff {best['top_pos_diff']:+.4f}")

sae = torch.load(f"{sae_path}layer{BEST_LAYER}.sae.pt", map_location=device)
W_enc = sae["W_enc"].to(torch.float32)
b_enc = sae["b_enc"].to(torch.float32)

captured = {}
def _hook(module, input, output):
    hidden = output[0] if isinstance(output, tuple) else output
    captured["residual"] = hidden.detach()
handle = model.model.layers[BEST_LAYER].register_forward_hook(_hook)

def feat_acts_for(text: str):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)
    residual = captured["residual"].to(torch.float32)
    acts = (residual @ W_enc.T + b_enc)[0, :, FEATURE_IDX].cpu().clamp(min=0)
    return inputs["input_ids"][0].cpu(), acts

pos_data = [feat_acts_for(t) for t in POS]
neg_data = [feat_acts_for(t) for t in NEG]
handle.remove()

global_max = max(a.max().item() for _, a in pos_data) or 1.0

def make_activating(tokens, activations):
    normalized = (activations / global_max * 10).round().int()
    return ActivatingExample(
        tokens=tokens,
        activations=activations,
        normalized_activations=normalized,
        str_tokens=[tokenizer.decode([t]) for t in tokens.tolist()],
    )

activating = [make_activating(t, a) for t, a in pos_data if a.max() > 0]
activating.sort(key=lambda e: -e.activations.max().item())

split = max(1, len(activating) - 2)
train = activating[:split]
test = activating[split:]

not_active = [
    NonActivatingExample(
        tokens=tokens,
        activations=acts,
        str_tokens=[tokenizer.decode([t]) for t in tokens.tolist()],
    )
    for tokens, acts in neg_data
]

latent = Latent(module_name=f"layers.{BEST_LAYER}", latent_index=FEATURE_IDX)
record = LatentRecord(
    latent=latent,
    examples=train + test,
    train=train,
    test=test,
    not_active=not_active,
)

client = OpenRouter(
    model="anthropic/claude-sonnet-4.5",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

async def run():
    explainer = DefaultExplainer(client=client)
    explainer_result = await explainer(record)
    record.explanation = explainer_result.explanation
    print(f"\n--- Explanation ---\n{record.explanation}")

    scorer = DetectionScorer(client=client, n_examples_shown=1)
    scorer_result = await scorer(record)
    print(f"\n--- DetectionScorer score ({len(scorer_result.score)} samples) ---")
    for i, out in enumerate(scorer_result.score):
        print(f"  [{i}] prediction={out.prediction} probability={out.probability}")

asyncio.run(run())

import asyncio
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from delphi.clients import OpenRouter
from delphi.explainers import DefaultExplainer
from delphi.latents import ActivatingExample, Latent, LatentRecord

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1. Load base model ────────────────────────────────────────────────────────
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

# ── 2. Contrastive examples (positive vs negative sentiment) ─────────────────
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

# ── 3. Per-layer contrastive analysis ────────────────────────────────────────
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

results = []
for L in LAYERS:
    r = analyze_layer(L)
    results.append(r)
    print(f"layer {r['layer']:>2}  POS: feat={r['top_pos_idx']:>5} diff={r['top_pos_diff']:+.4f}")

best = max(results, key=lambda r: r["top_pos_diff"])
BEST_LAYER = best["layer"]
FEATURE_IDX = best["top_pos_idx"]
print(f"\nBest POS-leaning feature: layer {BEST_LAYER}, feature {FEATURE_IDX}, diff {best['top_pos_diff']:+.4f}")

# ── 4. Per-token activations of the chosen feature on POS+NEG ────────────────
sae = torch.load(f"{sae_path}layer{BEST_LAYER}.sae.pt", map_location=device)
W_enc = sae["W_enc"].to(torch.float32)
b_enc = sae["b_enc"].to(torch.float32)

captured = {}
def _hook(module, input, output):
    hidden = output[0] if isinstance(output, tuple) else output
    captured["residual"] = hidden.detach()
handle = model.model.layers[BEST_LAYER].register_forward_hook(_hook)

per_text = []
for text in POS + NEG:
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)
    residual = captured["residual"].to(torch.float32)
    feat_acts = (residual @ W_enc.T + b_enc)[0, :, FEATURE_IDX].cpu().clamp(min=0)
    per_text.append((inputs["input_ids"][0].cpu(), feat_acts))
handle.remove()

# ── 5. Build Delphi LatentRecord ─────────────────────────────────────────────
global_max = max(a.max().item() for _, a in per_text) or 1.0
examples = []
for tokens, activations in per_text:
    if activations.max() <= 0:
        continue
    normalized = (activations / global_max * 10).round().int()
    examples.append(ActivatingExample(
        tokens=tokens,
        activations=activations,
        normalized_activations=normalized,
        str_tokens=[tokenizer.decode([t]) for t in tokens.tolist()],
    ))
examples.sort(key=lambda e: -e.activations.max().item())

latent = Latent(module_name=f"layers.{BEST_LAYER}", latent_index=FEATURE_IDX)
record = LatentRecord(latent=latent, examples=examples, train=examples)

# ── 6. Run the Delphi explainer over OpenRouter ──────────────────────────────
client = OpenRouter(
    model="anthropic/claude-sonnet-4.5",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
explainer = DefaultExplainer(client=client)
result = asyncio.run(explainer(record))

print(f"\n--- Delphi explanation (layer={BEST_LAYER}, feature={FEATURE_IDX}) ---")
print(result.explanation)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
N_LAYERS = 24
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
    W_enc = sae["W_enc"].to(torch.float32)  # (n_features, d_in)
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
    top_neg_idx = int((-diff).argmax())
    return {
        "layer": layer,
        "top_pos_idx": top_pos_idx,
        "top_pos_diff": float(diff[top_pos_idx]),
        "top_neg_idx": top_neg_idx,
        "top_neg_diff": float(diff[top_neg_idx]),
    }

results = []
for L in LAYERS:
    r = analyze_layer(L)
    results.append(r)
    print(
        f"layer {r['layer']:>2}  "
        f"POS: feat={r['top_pos_idx']:>5} diff={r['top_pos_diff']:+.4f}  "
        f"NEG: feat={r['top_neg_idx']:>5} diff={r['top_neg_diff']:+.4f}"
    )

# ── 4. Pick the layer where sentiment is most separable ──────────────────────
best = max(results, key=lambda r: r["top_pos_diff"])
BEST_LAYER = best["layer"]
FEATURE_IDX = best["top_pos_idx"]
print(
    f"\nBest POS-leaning feature: layer {BEST_LAYER}, "
    f"feature {FEATURE_IDX}, diff {best['top_pos_diff']:+.4f}"
)

# ── 5. Steered generation at the best layer ──────────────────────────────────
STEER_ALPHA = 8.0
MAX_NEW_TOKENS = 60
PROMPT = "The movie I saw last night was"

sae = torch.load(f"{sae_path}layer{BEST_LAYER}.sae.pt", map_location=device)
W_dec = sae["W_dec"]  # (d_in, n_features)
steer_dir = W_dec[:, FEATURE_IDX].to(torch.float32)
steer_dir = steer_dir / steer_dir.norm()

def _steer_hook(module, input, output):
    hidden = output[0] if isinstance(output, tuple) else output
    delta = (STEER_ALPHA * steer_dir).to(hidden.dtype).to(hidden.device)
    new_hidden = hidden + delta
    if isinstance(output, tuple):
        return (new_hidden,) + output[1:]
    return new_hidden

def generate(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

baseline = generate(PROMPT)
steer_handle = model.model.layers[BEST_LAYER].register_forward_hook(_steer_hook)
try:
    steered = generate(PROMPT)
finally:
    steer_handle.remove()

print(f"\n--- Baseline ---\n{baseline}")
print(f"\n--- Steered (layer={BEST_LAYER}, feature={FEATURE_IDX}, alpha={STEER_ALPHA}) ---\n{steered}")

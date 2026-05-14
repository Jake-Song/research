import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "/content/drive/MyDrive/huggingface/models/Qwen3.5-9B"
sae_path = "/content/drive/MyDrive/huggingface/sae/"

N_LAYERS = 36
TOPK = 50
TOPK_FEATURES = 20
OUTPUT_PT = "/home/jake/research/sae/identify_features.pt"
OUTPUT_JSON = "/home/jake/research/sae/identify_features.top.json"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32).to(device)
model.eval()

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

texts = POS + NEG
n_pos = len(POS)

# ── Pass A: capture residuals at every layer for every example ───────────────
captured: dict[int, torch.Tensor] = {}

def make_hook(L: int):
    def _hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured[L] = h.detach()[0].to(torch.float32).cpu()
    return _hook

handles = [model.model.layers[L].register_forward_hook(make_hook(L)) for L in range(N_LAYERS)]

residuals: list[dict[int, torch.Tensor]] = []
for text in texts:
    captured.clear()
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)
    residuals.append(dict(captured))

for h in handles:
    h.remove()

# ── Pass B: per-layer SAE encode, max over positions ─────────────────────────
max_acts = None  # allocated after first SAE load once n_features is known
n_features = None

for L in range(N_LAYERS):
    print(f"layer {L}")
    sae = torch.load(f"{sae_path}layer{L}.sae.pt", map_location=device)
    W_enc = sae["W_enc"].to(torch.float32)
    b_enc = sae["b_enc"].to(torch.float32)

    if max_acts is None:
        n_features = W_enc.shape[0]
        max_acts = torch.zeros(len(texts), N_LAYERS, n_features)

    for i in range(len(texts)):
        resid = residuals[i][L].to(device)
        pre = resid @ W_enc.T + b_enc
        topv, topi = pre.topk(TOPK, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, topi, topv)
        max_acts[i, L] = acts.max(dim=0).values.cpu()

    del sae, W_enc, b_enc
    torch.cuda.empty_cache()

# ── Aggregate (Eq. 14: Δ_f = Σ_{y=1} h_{i,f} − Σ_{y=0} h_{i,f}) ──────────────
fired = (max_acts > 0).float()
pos_count = fired[:n_pos].sum(0)
neg_count = fired[n_pos:].sum(0)
diff = pos_count - neg_count

per_layer_topv, per_layer_topi = diff.topk(TOPK_FEATURES, dim=-1)

print(f"\n{'layer':>6} {'rank':>5} {'feature':>8} {'pos':>5} {'neg':>5} {'diff':>7}")
top_entries = []
for L in range(N_LAYERS):
    for k in range(TOPK_FEATURES):
        f = int(per_layer_topi[L, k])
        score = float(per_layer_topv[L, k])
        p = float(pos_count[L, f])
        n = float(neg_count[L, f])
        print(f"{L:>6} {k:>5} {f:>8} {p:>5.0f} {n:>5.0f} {score:+7.0f}")
        top_entries.append({"layer": L, "rank": k, "feature": f, "pos_count": p, "neg_count": n, "diff": score})

# ── Save ─────────────────────────────────────────────────────────────────────
torch.save({
    "pos_count": pos_count,
    "neg_count": neg_count,
    "diff": diff,
    "max_acts": max_acts,
    "pos_texts": POS,
    "neg_texts": NEG,
    "topk": TOPK,
    "topk_features": TOPK_FEATURES,
    "n_layers": N_LAYERS,
    "epsilon": 0.0,
}, OUTPUT_PT)

with open(OUTPUT_JSON, "w") as f:
    json.dump(top_entries, f, indent=2)



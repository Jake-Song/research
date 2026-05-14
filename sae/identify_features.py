import json

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "/content/drive/MyDrive/huggingface/models/Qwen3.5-9B"
sae_path = "/content/drive/MyDrive/huggingface/sae/"

N_LAYERS = 36
TOPK = 50
TOPK_FEATURES = 20
LANGUAGE = "en"
N_PER_CLASS = 2000  # paper §5.2: 2000 toxic + 2000 clean for feature discovery
OUTPUT_PT = "/home/jake/research/sae/identify_features.pt"
OUTPUT_JSON = "/home/jake/research/sae/identify_features.top.json"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32).to(device)
model.eval()

ds = load_dataset("textdetox/multilingual_toxicity_dataset", split=LANGUAGE)
TOXIC = [r["text"] for r in ds if r["toxic"] == 1][:N_PER_CLASS]
CLEAN = [r["text"] for r in ds if r["toxic"] == 0][:N_PER_CLASS]

texts = TOXIC + CLEAN
n_toxic = len(TOXIC)

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
toxic_count = fired[:n_toxic].sum(0)
clean_count = fired[n_toxic:].sum(0)
diff = toxic_count - clean_count

per_layer_topv, per_layer_topi = diff.topk(TOPK_FEATURES, dim=-1)

print(f"\n{'layer':>6} {'rank':>5} {'feature':>8} {'toxic':>6} {'clean':>6} {'diff':>7}")
top_entries = []
for L in range(N_LAYERS):
    for k in range(TOPK_FEATURES):
        f = int(per_layer_topi[L, k])
        score = float(per_layer_topv[L, k])
        p = float(toxic_count[L, f])
        n = float(clean_count[L, f])
        print(f"{L:>6} {k:>5} {f:>8} {p:>6.0f} {n:>6.0f} {score:+7.0f}")
        top_entries.append({"layer": L, "rank": k, "feature": f, "toxic_count": p, "clean_count": n, "diff": score})

# ── Save ─────────────────────────────────────────────────────────────────────
torch.save({
    "toxic_count": toxic_count,
    "clean_count": clean_count,
    "diff": diff,
    "max_acts": max_acts,
    "toxic_texts": TOXIC,
    "clean_texts": CLEAN,
    "language": LANGUAGE,
    "topk": TOPK,
    "topk_features": TOPK_FEATURES,
    "n_layers": N_LAYERS,
    "epsilon": 0.0,
}, OUTPUT_PT)

with open(OUTPUT_JSON, "w") as f:
    json.dump(top_entries, f, indent=2)



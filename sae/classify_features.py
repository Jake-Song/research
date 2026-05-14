import json

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "/content/drive/MyDrive/huggingface/models/Qwen3.5-9B"
sae_path = "/content/drive/MyDrive/huggingface/sae/"

N_LAYERS = 36
TOPK = 50
EPSILON = 0.0
LANGUAGE = "en"
N_PER_CLASS = 500  # paper §5.2: 500 toxic + 500 clean for evaluation
DISCOVERY_PER_CLASS = 2000  # must match identify_features.py N_PER_CLASS (used to skip the discovery split)
SELECTION_PT = "/home/jake/research/sae/identify_features.pt"
OUTPUT_JSON = "/home/jake/research/sae/classify_features.json"

# ── Test split (y=1 toxic, y=0 clean), held out from the discovery split ─────
ds = load_dataset("textdetox/multilingual_toxicity_dataset", split=LANGUAGE)
toxic_all = [r["text"] for r in ds if r["toxic"] == 1]
clean_all = [r["text"] for r in ds if r["toxic"] == 0]
TEST_TOXIC = toxic_all[DISCOVERY_PER_CLASS : DISCOVERY_PER_CLASS + N_PER_CLASS]
TEST_CLEAN = clean_all[DISCOVERY_PER_CLASS : DISCOVERY_PER_CLASS + N_PER_CLASS]

test_texts = TEST_TOXIC + TEST_CLEAN
y_true = torch.tensor([1] * len(TEST_TOXIC) + [0] * len(TEST_CLEAN))

# ── Load selected features S_ℓ per layer ─────────────────────────────────────
sel = torch.load(SELECTION_PT, map_location="cpu")
diff = sel["diff"]  # (N_LAYERS, n_features)
K = int(sel["topk_features"])
_, selected = diff.topk(K, dim=-1)  # (N_LAYERS, K)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32).to(device)
model.eval()

# ── Pass A: capture residuals at every layer for every test example ──────────
captured: dict[int, torch.Tensor] = {}

def make_hook(L: int):
    def _hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured[L] = h.detach()[0].to(torch.float32).cpu()
    return _hook

handles = [model.model.layers[L].register_forward_hook(make_hook(L)) for L in range(N_LAYERS)]

residuals: list[dict[int, torch.Tensor]] = []
for text in test_texts:
    captured.clear()
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)
    residuals.append(dict(captured))

for h in handles:
    h.remove()

# ── Pass B: per-layer SAE encode, OR-rule over S_ℓ (Eq. 15) ──────────────────
# y_hat[i, L] = 1[ max_{f ∈ S_ℓ} max_t a_{i,t,f} > ε ]
y_hat = torch.zeros(len(test_texts), N_LAYERS, dtype=torch.long)

for L in range(N_LAYERS):
    print(f"layer {L}")
    sae = torch.load(f"{sae_path}layer{L}.sae.pt", map_location=device)
    W_enc = sae["W_enc"].to(torch.float32)
    b_enc = sae["b_enc"].to(torch.float32)

    S = selected[L].to(device)  # (K,)

    for i in range(len(test_texts)):
        resid = residuals[i][L].to(device)
        pre = resid @ W_enc.T + b_enc
        topv, topi = pre.topk(TOPK, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, topi, topv)
        sel_acts = acts.index_select(-1, S)  # (n_tokens, K)
        fired = (sel_acts > EPSILON).any().item()
        y_hat[i, L] = int(fired)

    del sae, W_enc, b_enc
    torch.cuda.empty_cache()

# ── Report per-layer accuracy / precision / recall / F1 ──────────────────────
print(f"\n{'layer':>6} {'acc':>8} {'prec':>8} {'rec':>8} {'F1':>8} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}")
results = []
for L in range(N_LAYERS):
    pred = y_hat[:, L]
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / len(test_texts)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"{L:>6} {acc:>8.4f} {precision:>8.4f} {recall:>8.4f} {f1:>8.4f} {tp:>4} {fp:>4} {tn:>4} {fn:>4}")
    results.append({
        "layer": L,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    })

with open(OUTPUT_JSON, "w") as f:
    json.dump({
        "epsilon": EPSILON,
        "topk_features": K,
        "language": LANGUAGE,
        "test_toxic": TEST_TOXIC,
        "test_clean": TEST_CLEAN,
        "per_layer": results,
    }, f, indent=2)

"""Feature-coverage redundancy R̂(D) for GSM8K, per Qwen-Scope §4.2.

For each question x, F(x) = {j : SAE_j(x_last_token) > 0} under Top-k=50.
|F(D)| = |⋃_i F(x_i)|, n_f = #{i : f ∈ F(x_i)}.

Closed-form expected coverage:
    c_n = (1 / |F(D)|) · Σ_{f : n_f > 0} [1 - C(N - n_f, n) / C(N, n)]
Scalar: R̂(D) = (Σ_n c_n) / |F(D)|     (Eq. 9 of the paper)
"""
import math
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "/content/drive/MyDrive/huggingface/models/Qwen3.5-2B"
sae_path   = "/content/drive/MyDrive/huggingface/sae/"
out_path   = Path("/content/drive/MyDrive/huggingface/redundancy/gsm8k_layer12.npz")
out_path.parent.mkdir(parents=True, exist_ok=True)

LAYER = 12
TOPK  = 50

# ── 1. Load base model + SAE ──────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32).to(device)
model.eval()

sae = torch.load(f"{sae_path}layer{LAYER}.sae.pt", map_location=device)
W_enc = sae["W_enc"].to(torch.float32)  # (n_features, d_in)
b_enc = sae["b_enc"].to(torch.float32)
n_features = W_enc.shape[0]

# ── 2. Hook residual stream ───────────────────────────────────────────────────
captured = {}
def _hook(module, input, output):
    hidden = output[0] if isinstance(output, tuple) else output
    captured["residual"] = hidden.detach()
handle = model.model.layers[LAYER].register_forward_hook(_hook)

# ── 3. Encode every GSM8K question, record active feature set ────────────────
ds = load_dataset("gsm8k", "main", split="test")
N = len(ds)

support = torch.zeros(n_features, dtype=torch.int64, device="cpu")
for i, ex in enumerate(ds):
    inputs = tokenizer(ex["question"], return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)
    residual = captured["residual"][0, -1].to(torch.float32)  # (d_in,)
    pre = residual @ W_enc.T + b_enc                          # (n_features,)
    topk_vals, topk_idx = pre.topk(TOPK, dim=-1)
    active = topk_idx[topk_vals > 0].cpu()
    support[active] += 1

    if i == 0:
        print(f"[sanity] sample 0 active features (first 5): {active[:5].tolist()}")
        print(f"[sanity] sample 0 #active: {active.numel()} (expect ≤ {TOPK})")
    if (i + 1) % 100 == 0:
        print(f"  processed {i+1}/{N}")

handle.remove()

# ── 4. Aggregate ──────────────────────────────────────────────────────────────
n_f = support.numpy().astype(np.int64)          # (n_features,)
active_mask = n_f > 0
FD = int(active_mask.sum())                     # |F(D)|
print(f"\n|F(D)| = {FD}   N = {N}")

# ── 5. Closed-form coverage curve via log-binomials ──────────────────────────
# log C(a, b) = lgamma(a+1) - lgamma(b+1) - lgamma(a-b+1)
n_f_active = n_f[active_mask]                   # (FD,)
lg = np.array([math.lgamma(k + 1) for k in range(N + 1)])  # lgamma table

def log_comb(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    out = lg[a] - lg[b] - lg[a - b]
    out = np.where(b > a, -np.inf, out)
    out = np.where(b < 0, -np.inf, out)
    return out

c = np.empty(N + 1, dtype=np.float64)           # index 0..N (we use 1..N)
c[0] = 0.0
log_C_N_n_all = log_comb(N, np.arange(N + 1))   # (N+1,)
for n in range(1, N + 1):
    # Pr[f ∉ F(S)] = C(N - n_f, n) / C(N, n), valid when n_f ≤ N - n
    safe = n_f_active <= (N - n)
    log_num = log_comb(N - n_f_active[safe], n)
    pr_missing = np.zeros(FD, dtype=np.float64)
    pr_missing[safe] = np.exp(log_num - log_C_N_n_all[n])
    pr_present = 1.0 - pr_missing
    c[n] = pr_present.sum() / FD
c[N] = 1.0  # exact by construction (numerical guard)

R_hat = c[1:].sum() / FD
print(f"\nc_1   = {c[1]:.6f}   (expect ≈ {TOPK / FD:.6f})")
print(f"c_{N//10:<4} = {c[N//10]:.6f}")
print(f"c_{N//2:<4} = {c[N//2]:.6f}")
print(f"c_{N:<4} = {c[N]:.6f}   (expect 1.0)")
print(f"\nR̂(GSM8K) = {R_hat:.6f}")

# ── 6. Save ──────────────────────────────────────────────────────────────────
np.savez(out_path, n_f=n_f, c=c, R_hat=R_hat, N=N, FD=FD, layer=LAYER, topk=TOPK)
print(f"\nSaved to {out_path}")

"""Feature-coverage redundancy R̂(D) and inter-benchmark overlap, per Qwen-Scope §4.2–4.3.

For each question x, F(x) = {j : SAE_j(x_last_token) > 0} under Top-k=50.
|F(D)| = |⋃_i F(x_i)|, n_f = #{i : f ∈ F(x_i)}.

Per-benchmark redundancy (Eq. 9):
    c_n = (1 / |F(D)|) · Σ_{f : n_f > 0} [1 - C(N - n_f, n) / C(N, n)]
    R̂(D) = (Σ_n c_n) / |F(D)|

Inter-benchmark overlap (Eq. 10–11):
    overlap(D₁, D₂)     = |F(D₁) ∩ F(D₂)| / |F(D₁)|
    overlap_min(D₁, D₂) = |F(D₁) ∩ F(D₂)| / min(|F(D₁)|, |F(D₂)|)
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
out_path   = Path("/content/drive/MyDrive/huggingface/redundancy/gsm8k_vs_math_layer12.npz")
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


def compute_support(ds, text_field, tag):
    """Encode every example's last token; return n_f support over features."""
    N = len(ds)
    support = torch.zeros(n_features, dtype=torch.int64, device="cpu")
    for i, ex in enumerate(ds):
        inputs = tokenizer(ex[text_field], return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
        residual = captured["residual"][0, -1].to(torch.float32)  # (d_in,)
        pre = residual @ W_enc.T + b_enc                          # (n_features,)
        topk_vals, topk_idx = pre.topk(TOPK, dim=-1)
        active = topk_idx[topk_vals > 0].cpu()
        support[active] += 1

        if i == 0:
            print(f"[sanity:{tag}] sample 0 active features (first 5): {active[:5].tolist()}")
            print(f"[sanity:{tag}] sample 0 #active: {active.numel()} (expect ≤ {TOPK})")
        if (i + 1) % 100 == 0:
            print(f"  [{tag}] processed {i+1}/{N}")
    return support.numpy().astype(np.int64)


def redundancy_curve(n_f, N):
    """Closed-form coverage curve c[1..N] and scalar R̂ via log-binomials."""
    active_mask = n_f > 0
    FD = int(active_mask.sum())
    n_f_active = n_f[active_mask]
    lg = np.array([math.lgamma(k + 1) for k in range(N + 1)])

    def log_comb(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        out = lg[a] - lg[b] - lg[a - b]
        out = np.where(b > a, -np.inf, out)
        out = np.where(b < 0, -np.inf, out)
        return out

    c = np.empty(N + 1, dtype=np.float64)
    c[0] = 0.0
    log_C_N_n_all = log_comb(N, np.arange(N + 1))
    for n in range(1, N + 1):
        safe = n_f_active <= (N - n)
        log_num = log_comb(N - n_f_active[safe], n)
        pr_missing = np.zeros(FD, dtype=np.float64)
        pr_missing[safe] = np.exp(log_num - log_C_N_n_all[n])
        pr_present = 1.0 - pr_missing
        c[n] = pr_present.sum() / FD
    c[N] = 1.0
    R_hat = c[1:].sum() / FD
    return FD, c, R_hat


# ── 3. Encode both benchmarks ────────────────────────────────────────────────
ds_gsm  = load_dataset("gsm8k", "main", split="test")
ds_math = load_dataset("hendrycks/competition_math", split="test")
N_gsm, N_math = len(ds_gsm), len(ds_math)

nf_gsm  = compute_support(ds_gsm,  "question", "gsm8k")
nf_math = compute_support(ds_math, "problem",  "math")

handle.remove()

# ── 4. Per-benchmark redundancy ──────────────────────────────────────────────
FD_gsm,  c_gsm,  R_gsm  = redundancy_curve(nf_gsm,  N_gsm)
FD_math, c_math, R_math = redundancy_curve(nf_math, N_math)

print(f"\n|F(GSM8K)| = {FD_gsm}   N = {N_gsm}")
print(f"|F(MATH)|  = {FD_math}   N = {N_math}")
print(f"R̂(GSM8K) = {R_gsm:.6f}")
print(f"R̂(MATH)  = {R_math:.6f}")

# ── 5. Inter-benchmark overlap (Eq. 10–11) ───────────────────────────────────
F_gsm  = nf_gsm  > 0
F_math = nf_math > 0
inter  = int((F_gsm & F_math).sum())
ov_gsm_to_math = inter / FD_gsm
ov_math_to_gsm = inter / FD_math
ov_min         = inter / min(FD_gsm, FD_math)

print(f"\n|F(GSM8K) ∩ F(MATH)| = {inter}")
print(f"overlap(GSM8K, MATH) = {ov_gsm_to_math:.4f}   (Eq. 10)")
print(f"overlap(MATH, GSM8K) = {ov_math_to_gsm:.4f}   (Eq. 10)")
print(f"overlap_min          = {ov_min:.4f}   (Eq. 11)")

# ── 6. Save ──────────────────────────────────────────────────────────────────
np.savez(
    out_path,
    n_f_gsm=nf_gsm,   c_gsm=c_gsm,   R_gsm=R_gsm,   FD_gsm=FD_gsm,
    n_f_math=nf_math, c_math=c_math, R_math=R_math, FD_math=FD_math,
    overlap_gsm_to_math=ov_gsm_to_math,
    overlap_math_to_gsm=ov_math_to_gsm,
    overlap_min=ov_min,
    N_gsm=N_gsm, N_math=N_math,
    layer=LAYER, topk=TOPK,
)
print(f"\nSaved to {out_path}")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── 1. Load base model ────────────────────────────────────────────────────────
model_path = "/content/drive/MyDrive/huggingface/models/Qwen3.5-2B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float32
)
model.eval()

sae_path = "/content/drive/MyDrive/huggingface/sae/"

# ── 2. Load SAE for a target layer ───────────────────────────────────────────
LAYER = 1  # choose any layer in 0–23
sae = torch.load(f"{sae_path}layer{LAYER}.sae.pt", map_location="cpu")
W_enc = sae["W_enc"]  # (32768, 2048)
b_enc = sae["b_enc"]  # (32768,)

def get_feature_acts(residual: torch.Tensor) -> torch.Tensor:
    """residual: (..., 2048) → sparse feature activations (..., 32768)"""
    pre_acts = residual @ W_enc.T + b_enc
    topk_vals, topk_idx = pre_acts.topk(50, dim=-1)
    acts = torch.zeros_like(pre_acts)
    acts.scatter_(-1, topk_idx, topk_vals)
    return acts

# ── 3. Hook residual stream after the target transformer layer ────────────────
captured = {}

def _hook(module, input, output):
    hidden = output[0] if isinstance(output, tuple) else output
    captured["residual"] = hidden.detach().cpu()

hook = model.model.layers[LAYER].register_forward_hook(_hook)

# ── 4. Forward pass ───────────────────────────────────────────────────────────
text = "Hello安寧"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    model(**inputs)
hook.remove()

# ── 5. Extract feature activations ───────────────────────────────────────────
# Ensure residual is float32 to match SAE weights
residual = captured["residual"].to(torch.float32)               # (1, seq_len, 2048)
feature_acts = get_feature_acts(residual)     # (1, seq_len, 32768)

# Inspect active features for the last token
last_token_acts = feature_acts[0, -1]         # (32768,)
active_idx = last_token_acts.nonzero(as_tuple=True)[0]
print(f"Active features : {active_idx.tolist()}")
print(f"Feature values  : {last_token_acts[active_idx].tolist()}")
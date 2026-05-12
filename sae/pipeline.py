import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from delphi.clients import OpenRouter
from delphi.config import ConstructorConfig, SamplerConfig
from delphi.explainers import DefaultExplainer
from delphi.latents import LatentCache, LatentDataset
from delphi.pipeline import Pipe, Pipeline, process_wrapper
from delphi.scorers import DetectionScorer, FuzzingScorer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "/content/drive/MyDrive/huggingface/models/Qwen3.5-9B"
sae_path = "/content/drive/MyDrive/huggingface/sae/"
out_dir = Path("/content/drive/MyDrive/huggingface/delphi_run")
cache_dir = out_dir / "cache"
scores_dir = out_dir / "scores"
explanations_dir = out_dir / "explanations"
for d in (cache_dir, scores_dir, explanations_dir):
    d.mkdir(parents=True, exist_ok=True)

LAYERS = list(range(8, 16))
TOPK = 50
TOP_N_PER_LAYER = 5
N_TOKENS = 1_000_000
CTX_LEN = 256
BATCH_SIZE = 8

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

# ── 1. Load base model ────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32).to(device)
model.eval()

# ── 2. Load SAEs and build per-layer encode callables ────────────────────────
sae_weights = {}
for L in LAYERS:
    sae = torch.load(f"{sae_path}layer{L}.sae.pt", map_location=device)
    sae_weights[L] = (
        sae["W_enc"].to(torch.float32),
        sae["b_enc"].to(torch.float32),
    )

def make_encode(L):
    W_enc, b_enc = sae_weights[L]
    def encode(residual: torch.Tensor) -> torch.Tensor:
        pre_acts = residual.to(torch.float32) @ W_enc.T + b_enc
        topk_vals, topk_idx = pre_acts.topk(TOPK, dim=-1)
        out = torch.zeros_like(pre_acts)
        out.scatter_(-1, topk_idx, topk_vals)
        return out
    return encode

hookpoint_to_sparse_encode = {f"layers.{L}": make_encode(L) for L in LAYERS}

# ── 3. Pick top-N POS-leaning features per layer ─────────────────────────────
def topn_features_for_layer(layer: int, n: int) -> torch.Tensor:
    W_enc, b_enc = sae_weights[layer]
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
    idx = diff.topk(n).indices.cpu()
    print(f"layer {layer:>2}  top-{n} features: {idx.tolist()}")
    return idx

top_n_per_layer = {
    f"layers.{L}": topn_features_for_layer(L, TOP_N_PER_LAYER) for L in LAYERS
}

# ── 4. Tokenize corpus ───────────────────────────────────────────────────────
print(f"\nLoading corpus and tokenizing to ~{N_TOKENS:,} tokens at ctx={CTX_LEN}...")
raw = load_dataset(
    "togethercomputer/RedPajama-Data-1T-Sample", split="train", streaming=True
)
n_seqs = N_TOKENS // CTX_LEN
buf, seqs = [], []
for row in raw:
    ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
    buf.extend(ids)
    while len(buf) >= CTX_LEN and len(seqs) < n_seqs:
        seqs.append(buf[:CTX_LEN])
        buf = buf[CTX_LEN:]
    if len(seqs) >= n_seqs:
        break
tokens = torch.tensor(seqs, dtype=torch.long)
print(f"Tokenized: tokens.shape={tuple(tokens.shape)}")

# ── 5. Cache SAE activations ─────────────────────────────────────────────────
print("\nCaching SAE activations with LatentCache...")
cache = LatentCache(
    model,
    hookpoint_to_sparse_encode,
    batch_size=BATCH_SIZE,
    transcode=False,
    log_path=str(out_dir / "cache.log"),
)
cache.run(n_tokens=N_TOKENS, tokens=tokens)
cache.save_splits(n_splits=2, save_dir=str(cache_dir))
cache.save_config(save_dir=str(cache_dir), cfg=None, model_name=model_path)
print(f"Cache saved to {cache_dir}")

# ── 6. Build LatentDataset with stratified-decile sampling ────────────────────
sampler_cfg = SamplerConfig(
    n_examples_train=40,
    n_examples_test=50,
    n_quantiles=10,
    train_type="quantiles",
    test_type="quantiles",
)
constructor_cfg = ConstructorConfig(
    example_ctx_len=32,
    min_examples=200,
    n_non_activating=50,
    non_activating_source="random",
)
dataset = LatentDataset(
    raw_dir=str(cache_dir),
    sampler_cfg=sampler_cfg,
    constructor_cfg=constructor_cfg,
    tokenizer=tokenizer,
    modules=[f"layers.{L}" for L in LAYERS],
    latents=top_n_per_layer,
)

# ── 7. Run explainer + detection + fuzzing through Delphi Pipeline ───────────
client = OpenRouter(
    model="anthropic/claude-sonnet-4.5",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

def explainer_postprocess(result):
    latent = result.record.latent
    path = explanations_dir / f"{latent.module_name}_{latent.latent_index}.txt"
    path.write_text(result.explanation)
    return result

def scorer_preprocess(result):
    record = result.record
    record.explanation = result.explanation
    record.extra_examples = record.not_active
    return record

def scorer_postprocess(result, name):
    latent = result.record.latent
    outs = result.score
    row = {
        "module": latent.module_name,
        "feature": int(latent.latent_index),
        "n": len(outs),
        "correct": sum(1 for o in outs if o.correct),
    }
    with (scores_dir / f"{name}.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    return result

explainer = DefaultExplainer(client, tokenizer=tokenizer)
explainer_pipe = process_wrapper(explainer, postprocess=explainer_postprocess)

detection_pipe = process_wrapper(
    DetectionScorer(client, n_examples_shown=5),
    preprocess=scorer_preprocess,
    postprocess=lambda r: scorer_postprocess(r, "detection"),
)
fuzzing_pipe = process_wrapper(
    FuzzingScorer(client, n_examples_shown=5),
    preprocess=scorer_preprocess,
    postprocess=lambda r: scorer_postprocess(r, "fuzzing"),
)

pipeline = Pipeline(dataset, explainer_pipe, Pipe(detection_pipe, fuzzing_pipe))
print("\nRunning Delphi pipeline...")
asyncio.run(pipeline.run(num_processes=1))

# ── 8. Summary: detection vs fuzzing per feature ─────────────────────────────
def load_acc(name: str) -> dict[tuple[str, int], tuple[int, int]]:
    path = scores_dir / f"{name}.jsonl"
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        r = json.loads(line)
        out[(r["module"], r["feature"])] = (r["correct"], r["n"])
    return out

det = load_acc("detection")
fuz = load_acc("fuzzing")
keys = sorted(set(det) | set(fuz))

print(f"\n{'module':<14}{'feat':>8}{'detection':>14}{'fuzzing':>12}{'disagree':>12}")
for k in keys:
    dc, dn = det.get(k, (0, 0))
    fc, fn = fuz.get(k, (0, 0))
    da = dc / dn if dn else 0.0
    fa = fc / fn if fn else 0.0
    print(f"{k[0]:<14}{k[1]:>8}{f'{dc}/{dn}={da:.2f}':>14}{f'{fc}/{fn}={fa:.2f}':>12}{abs(da-fa):>12.2f}")

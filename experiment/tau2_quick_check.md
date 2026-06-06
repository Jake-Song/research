# Tau2 Qwen Quick Check

This standalone Colab smoke test runs five deterministic airline tasks from
Tau2 v1.0.0. It installs the benchmark, launches
`Qwen/Qwen3-4B-Instruct-2507` with vLLM, runs the evaluation, and stops vLLM.
`gpt-4.1` is the hosted user simulator.

## Run

Select a GPU runtime, put the script in Colab, and run:

```python
import os
os.environ["OPENAI_API_KEY"] = "..."
```

```bash
!pip install -q uv
!uv run run_tau2_quick_check_colab.py
```

To evaluate a trained checkpoint, pass its Hugging Face ID or local path:

Edit `MODEL` and `RUN_NAME` near the top of the script, then run the same
`uv run` command. Other quick-check settings are constants in the same block.

Results and full verbose transcripts are written under:

```text
/content/tau2_quick_check/tau2-bench/data/simulations/<run-name>/
```

The vLLM log is `/content/tau2_quick_check/vllm.log`. Reusing a run name with
an existing result is rejected to avoid mixing checkpoint outputs.

Tau2 v1.0.0 eagerly imports optional voice packages during text-mode startup,
so the script installs its documented `voice` extra and PortAudio system
dependency. The evaluation itself remains text-only.

This is a pipeline check, not a published-score reproduction or checkpoint
comparison.

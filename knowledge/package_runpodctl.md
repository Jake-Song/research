# runpodctl — basic usage

- **What it is:** CLI to manage RunPod GPU pods, serverless endpoints, templates, network volumes, and file transfers.
- **Version referenced:** `runpodctl 2.1.9-673143d` (verified via `--version` on 2026-05-24)
- **Docs:** https://github.com/runpod/runpodctl  ·  Console: https://www.runpod.io/console

## Install

Already installed in this environment. Verify with:

```sh
runpodctl version
```

## Auth / Setup

```sh
# get key at https://www.runpod.io/console/user/settings
runpodctl doctor                  # interactive: prompts for key, saves to ~/.runpod/config.toml
# or:
export RUNPOD_API_KEY=your-key
```

## Core concepts

- **`pod`** — GPU/CPU pods (create, list, get, start, stop, delete, restart, reset, update).
- **`serverless` (alias `sls`)** — manage serverless endpoints.
- **`template` (alias `tpl`)** — reusable pod images/configs; find with `template search`.
- **`network-volume` (alias `nv`)** — persistent storage attachable to pods.
- **`send` / `receive`** — peer-to-peer file transfer via croc, codephrase-based.
- **Global flag:** `-o json|yaml` (default `json`) — all commands emit structured output.

## Minimal example — spin up a pod

```sh
# 1. discover a GPU id
runpodctl gpu list

# 2. find an official template
runpodctl template search pytorch
runpodctl template list --type official

# 3. create a pod from a template
runpodctl pod create \
  --template-id runpod-torch-v21 \
  --gpu-id "NVIDIA GeForce RTX 4090"

# 4. list running pods, grab the id
runpodctl pod list

# 5. ssh in
runpodctl ssh info <pod-id>
```

## Common patterns

```sh
# Create with a custom image instead of a template
runpodctl pod create --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 \
  --gpu-id "NVIDIA GeForce RTX 4090"
```

```sh
# CPU-only pod
runpodctl pod create --compute-type cpu --image ubuntu:22.04
```

```sh
# Create with more disk, env vars, exposed ports, and a network volume
runpodctl pod create --template-id runpod-torch-v21 \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --gpu-count 1 \
  --container-disk-in-gb 50 \
  --volume-in-gb 100 --volume-mount-path /workspace \
  --ports "8888/http,22/tcp" \
  --env '{"HF_TOKEN":"hf_xxx"}' \
  --network-volume-id <nv-id> \
  --name my-pod
```

```sh
# Pod lifecycle
runpodctl pod list -a                       # include exited
runpodctl pod list --status RUNNING --since 7d
runpodctl pod get <pod-id> --include-machine --include-network-volume
runpodctl pod stop <pod-id>
runpodctl pod start <pod-id>
runpodctl pod delete <pod-id>               # aliases: rm, remove
```

```sh
# SSH keys
runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub
runpodctl ssh list-keys
runpodctl ssh info <pod-id>                 # prints ssh command + key
```

```sh
# File transfer (croc-based, works pod<->local or local<->local)
# sender:
runpodctl send ./model.safetensors
# -> prints a code like: 1234-abcd-efgh
# receiver:
runpodctl receive 1234-abcd-efgh
```

```sh
# Account / inventory
runpodctl user                 # alias: me — balance + account info
runpodctl gpu list             # gpu ids, stock status, secure/community availability
runpodctl datacenter           # alias: dc
runpodctl billing
```

## Gotchas

- `--gpu-id` expects the exact `gpuId` string from `runpodctl gpu list` (e.g. `"NVIDIA GeForce RTX 4090"`), not the `displayName`.
- `--cloud-type` defaults to `SECURE`; `--public-ip` is community-cloud only, `--global-networking` is secure-cloud only.
- `--env` must be a JSON object string, not `KEY=VAL` pairs.
- `pod list` shows only running pods by default — use `-a` or `--status` to see exited ones.
- Default `-o json` means commands are pipeable into `jq`; switch to `-o yaml` for human reading.
- Deprecated top-level verbs still exist (`get`, `create`, `remove`, `start`, `stop`, `exec`, `project`, `config`) — prefer the resource-scoped form (`runpodctl pod create`, etc.).
- `send`/`receive` use croc and need outbound network on both ends; the codephrase is single-use.

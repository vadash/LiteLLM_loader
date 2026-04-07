# LiteLLM Proxy

LiteLLM proxy that exposes multiple free/community LLM backends behind a single entry model. Provides ordered failover across models with automatic health management.

## Setup

```bash
install.cmd              # Windows: check uv, uv sync, create .env template
./install.sh            # Unix: same
```

Requires [uv](https://docs.astral.sh/uv/) in PATH and `src/.env` with `NVIDIA_API_BASE` and `NVIDIA_API_KEY`.

## Running

**Windows:**
```bash
start.cmd               # start hidden (no console window)
stop.cmd                # stop running instance
status.cmd              # check if running
restart.cmd             # restart
```

**Unix:**
```bash
./start.sh              # start in background
./stop.sh               # stop running instance
./status.sh             # check if running
./restart.sh            # restart
```

All scripts are thin wrappers around `litellm_ctl.py` which tracks the process via `src/.litellm.pid`.

## Architecture

### Latency-Based Fallback Routing (`src/config.yaml`)

Models are organized into **user-facing groups** (`FAST`, `SMART`) and **reusable fallback groups** (`qwen3x`, `kimi2`, `nvidia_glm`, `zai_glm47`). The router uses latency-based routing to pick the fastest deployment within each group.

**User-facing groups:**
- `FAST` — lowest latency of 2 Qwen models (qwen3x)
- `SMART` — alias for `zai_glm47` (Claude via z.ai)

**Fallback chain** (primary → last resort):
```
FAST → qwen3x → kimi2 → zai_glm47 → longcat → qwen-coder → cerebras
SMART → qwen3x → nvidia_glm → kimi2 → qwen-coder → cerebras → longcat
```

Each group that appears in a fallback list **must have its own fallback entry**. The chain terminates at `cerebras: []`.

### Empty Response Handler (`src/handler.py`)

Custom callback that detects garbage LLM responses (empty content, training data leakage, leaked HTML documents, missing JSON structure) and marks deployments as dead via `router.cooldown_cache`. Implements four hook points:
- `log_success_event` — sync completion path
- `async_log_success_event` — async completion path
- `async_log_stream_complete_event` — streaming completion path
- `async_post_call_success_hook` — proxy-level last-resort guard

**Garbage detection is intentionally loose for JSON** — only checks that `{` or `[` exists somewhere in the response. Client-side repair handles truncation/syntax issues. HTML detection only flags full leaked documents (`<!DOCTYPE`, `<html` at start of response), not code snippets.

### Fallback Chain Completeness

Every model group that appears in a fallback list **must have its own fallback entry** in `config.yaml`, even if it's an empty list. Missing entries cause unhandled exception loops when that model fails. The chain terminates at `cerebras: []`.

### Health Checks

Background health checks are enabled (`health_check_interval: 60`). Dead models are pinged periodically and restored to rotation early if they recover, rather than waiting the full cooldown.

## Adding a New Model

Add a new entry in `src/config.yaml` with a unique `model_name`, then append it to the `fallbacks` list in `router_settings`. **Important:** if the model appears as a fallback target, it must also have its own fallback entry to avoid crash loops.
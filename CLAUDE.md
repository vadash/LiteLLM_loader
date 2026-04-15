# LiteLLM Proxy

LiteLLM proxy that exposes multiple free/community LLM backends behind a single entry model. Provides ordered failover across models with automatic health management.

## Setup

```bash
litellm_install.cmd      # Windows: check uv, uv sync, create .env template
./litellm_install.sh     # Unix: same
```

Requires [uv](https://docs.astral.sh/uv/) in PATH and `src/.env` with API credentials.

**Required environment variables:**
- `NVIDIA_API_BASE`, `NVIDIA_API_KEY` — NVIDIA-hosted models (Qwen, Kimi, GLM, LongCat)
- `ZAI_API_KEY` — z.ai GLM models
- `ALIBABA_API_BASE`, `ALIBABA_API_KEY` — Alibaba models
- `LITELLM_MASTER_KEY` — Proxy authentication key

## Running

**Windows:**
```bash
litellm_start.cmd        # start hidden (no console window)
litellm_stop.cmd         # stop running instance
litellm_status.cmd       # check if running
litellm_restart.cmd      # restart
```

**Unix:**
```bash
./litellm_start.sh       # start in background
./litellm_stop.sh        # stop running instance
./litellm_status.sh      # check if running
./litellm_restart.sh     # restart
```

All scripts are thin wrappers around `litellm_ctl.py` which tracks the process via `src/.litellm.pid`.

**Proxy endpoint:** `http://localhost:4000` (default)

**Logs:** `src/litellm.log` — check for `[GarbageResponseHandler]` messages when debugging fallback behavior.

## Architecture

### Latency-Based Fallback Routing (`src/config.yaml`)

Models are organized into **user-facing groups** (`FAST`, `SMART`) and **reusable fallback groups** (`nvidia_qwen3x`, `nvidia_kimi2`, `nvidia_glm`, `zai`, `zai_free`, `gemma4`, `longcat`, `ali`).

The router uses **latency-based routing** with `lowest_latency_buffer: 0.3` to pick the fastest deployment within each group.

**User-facing groups:**
- `FAST` — prioritizes gemma4, then Qwen via NVIDIA
- `SMART` — prioritizes z.ai GLM models

**Fallback chain** (primary → last resort):
```
FAST → gemma4 → nvidia_qwen3x → longcat → nvidia_kimi2 → nvidia_glm → zai → ali
SMART → zai → ali → nvidia_qwen3x → gemma4 → longcat → nvidia_kimi2 → nvidia_glm
```

**Content policy fallbacks** (triggered by `ContentPolicyViolationError`):
```
FAST → gemma4 → zai_free → nvidia_qwen3x → longcat → nvidia_kimi2 → nvidia_glm → ali → zai
SMART → ali → nvidia_qwen3x → gemma4 → longcat → nvidia_kimi2 → nvidia_glm → zai
```

**Fallback resolution is FLAT (not recursive):** When a group fails, the router iterates that group's list directly without consulting each model's own fallback entry. Every group must still have an empty fallback entry (`[]`) to avoid crash loops.

### Empty Response Handler (`src/handler.py`)

Custom callback that detects garbage LLM responses (empty content, training data leakage, leaked HTML documents, missing JSON structure) and marks deployments as dead via `router.cooldown_cache`. Implements four hook points:
- `log_success_event` — sync completion path
- `async_log_success_event` — async completion path
- `async_log_stream_complete_event` — streaming completion path
- `async_post_call_success_hook` — proxy-level last-resort guard

**Garbage detection is intentionally loose for JSON** — only checks that `{` or `[` exists somewhere in the response. Client-side repair handles truncation/syntax issues. HTML detection only flags full leaked documents (`<!DOCTYPE`, `<html` at start of response), not code snippets.

### Fallback Chain Completeness

Every model group that appears in a fallback list **must have its own fallback entry** in `config.yaml`, even if it's an empty list (`[]`). Missing entries cause unhandled exception loops when that model fails.

### Cooldown and Failure Thresholds

- `cooldown_time: 600` — keep providers dead for 10 minutes
- `allowed_fails: 1` — mark as dead after 1 garbage response
- Per-error-type thresholds configured in `allowed_fails_policy` (BadRequest, BadGateway, RateLimit, Timeout, ContentPolicyViolation)

### Health Checks

Background health checks are **disabled** (`background_health_checks: false`). Models remain on cooldown for the full 10-minute duration.

## Adding a New Model

1. Add model entries in `src/config.yaml` under `model_list` with a unique `model_name`
2. Append to both `fallbacks` and `content_policy_fallbacks` lists in `router_settings`
3. Add an empty fallback entry (`- model_name: []`) at the end of `fallbacks`

**Important:** If the model appears as a fallback target, it must also have its own fallback entry (even if empty) to avoid crash loops.

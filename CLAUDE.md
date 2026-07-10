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

### Routing Architecture (`src/config.yaml`, `src/handler.py`)

**Two model types:**

1. **Virtual entry points** (aliases) — `FAST`, `SMART`, `CODE`, `GOON`, plus their legacy `*1` forms. Defined with LiteLLM's native `model_group_alias`; the handler does not rewrite model names. Each serves a use case:
   - `FAST` — high-volume, no limits, fastest available
   - `SMART` — planning/orchestration, prefers reasoning models
   - `CODE` — implementation tasks, code-optimized models
   - `GOON` — no-censor mode

2. **Provider model groups** — named `PROVIDERCODE_MODELNAME` (e.g., `nvidia/glm51`, `zai/glm52`, `google/gemma4`). Multiple entries with the same name form a load-balanced pool.

**Routing:** Least-busy across interchangeable proxy deployments. Transport fallbacks are **flat** — when a group fails, the router iterates that group's list directly. Every group must have a fallback entry (even `[]`) to avoid crash loops.

**Content policy fallbacks** are triggered separately when a provider rejects a prompt due to moderation.

### Empty Response Handler (`src/handler.py`)

Custom callback that sanitizes requests, classifies provider failures, validates responses, and manages per-deployment circuit breakers via `router.cooldown_cache`. Invalid non-streaming responses receive one bounded quality retry before reaching the client. Implements four hook points:
- `async_log_success_event` — async completion path
- `async_log_stream_complete_event` — streaming completion path
- `async_post_call_success_hook` — client-facing validation and quality retry boundary
- `async_log_failure_event` — provider failure classification

JSON response formats are parsed as JSON rather than guessed from punctuation. HTML detection only flags full leaked documents (`<!DOCTYPE`, `<html` at the start), not code snippets. Legitimate CJK responses are allowed; only explicit refusal patterns are rejected.

### Fallback Chain Completeness

Every model group that appears in a fallback list **must have its own fallback entry** in `config.yaml`, even if it's an empty list (`[]`). Missing entries cause unhandled exception loops when that model fails.

### Cooldown and Failure Thresholds

- `cooldown_time: 60` — short default circuit with real traffic acting as the recovery probe
- `allowed_fails: 2` — tolerate one isolated transient failure
- Per-error-type thresholds configured in `allowed_fails_policy` (BadRequest, BadGateway, RateLimit, Timeout, ContentPolicyViolation)

### Health Checks

Background health checks are **disabled** (`background_health_checks: false`). Short cooldown expiry provides a lightweight half-open recovery probe without background traffic.

## Adding a New Model

1. Add model entries in `src/config.yaml` under `model_list` with `model_name` following `PROVIDERCODE_MODELNAME` convention (e.g., `zai/glm52`, `nvidia/kimik26`)
2. Append to both `fallbacks` and `content_policy_fallbacks` lists in `router_settings`
3. Add an empty fallback entry (`- model_name: []`) at the end of `fallbacks`
4. Update `router_settings.model_group_alias` if the model is a virtual entry point target

**Important:** If the model appears as a fallback target, it must also have its own fallback entry (even if empty) to avoid crash loops.

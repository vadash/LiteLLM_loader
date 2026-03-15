# LiteLLM Proxy

LiteLLM proxy that exposes multiple free/community LLM backends behind a single entry model. Provides ordered failover across models with automatic health management.

## Setup

```
install.cmd              # check Python, install litellm, create .env template
```

Requires Python in PATH and a `.env` file with `NVIDIA_API_BASE` and `NVIDIA_API_KEY`.

## Running

```
litellm_start.vbs        # start hidden (kills previous instance via PID file)
litellm_stop.vbs         # stop running instance via PID file
litellm_start.cmd        # start in console (normal)
litellm_start_debug.cmd  # start in console (--debug)
```

The VBS scripts use `.litellm.pid` to track the running process — only the exact process tree is killed, never other Python processes.

## Architecture

### Ordered Fallback Routing (`config.yaml`)

Each model has a **unique `model_name`** to enable deterministic top-to-bottom fallback — not random selection among same-name deployments.

Fallback chain (primary → last resort):
`qwen-80b` → `qwen-122b` → `kimi-k2-new` → `kimi-k2-old` → `cerebras` → `longcat`

The router tries the primary model first. On failure it cascades through fallbacks in order. Models that exceed `allowed_fails` are put in cooldown for `cooldown_time` seconds and skipped via `enable_pre_call_checks`.

### Empty Response Handler (`handler.py`)

Custom callback that detects empty LLM responses (no `content` and no `reasoning`/`reasoning_content`) and raises to trigger retry/fallback. Implements three hook points for coverage across sync, async, and proxy-level paths:
- `log_success_event` — sync completion path
- `async_log_success_event` — async completion path
- `async_post_call_success_hook` — proxy-level last-resort guard

## Adding a New Model

Add a new entry in `config.yaml` with a unique `model_name`, then append it to the `fallbacks` list in `router_settings`.

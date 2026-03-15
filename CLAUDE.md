# LiteLLM Proxy

LiteLLM proxy that exposes multiple free/community LLM backends behind a single `claude-opus-4-6` model name. Provides ordered failover across models with automatic health management.

## Running

```
litellm_start.cmd        # normal
litellm_start_debug.cmd  # with --debug
```

Both scripts load `.env` (contains `LITELLM_API_BASE` and `LITELLM_API_KEY`) then launch the proxy.

## Architecture

### Ordered Fallback Routing (`config.yaml`)

Each model has a **unique `model_name`** to enable deterministic top-to-bottom fallback — not random selection among same-name deployments.

Fallback chain order:
1. `claude-opus-4-6` — qwen3-next-80b (primary, what clients request)
2. `fallback-1` — qwen3.5-122b
3. `fallback-2` — kimi-k2-instruct
4. `fallback-3` — kimi-k2-instruct-0905
5. `fallback-4` — LongCat-Flash-Chat
6. `fallback-5` — cerebras qwen-3-235b

The router tries the primary model first. On failure it cascades through fallbacks in order. Models that exceed `allowed_fails` are put in cooldown for `cooldown_time` seconds and skipped via `enable_pre_call_checks`.

### Empty Response Handler (`handler.py`)

Custom callback that detects empty LLM responses (no `content` and no `reasoning`/`reasoning_content`) and raises to trigger retry/fallback. Implements three hook points for coverage across sync, async, and proxy-level paths:
- `log_success_event` — sync completion path
- `async_log_success_event` — async completion path
- `async_post_call_success_hook` — proxy-level last-resort guard

## Adding a New Model

Add a new entry in `config.yaml` with a unique `model_name` (e.g., `fallback-6`), then append it to the `fallbacks` list in `router_settings`.

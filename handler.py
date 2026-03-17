"""
LiteLLM Garbage Response Handler
================================

WHAT THIS DOES
--------------
Detects garbage responses from low-quality LLMs (training data leakage, HTML
templates, empty responses, system prompt leaks) and marks the deployment as
"dead" for 10 minutes. Router will skip dead deployments and use fallbacks.

WHAT WORKS (as of 2026-03-17)
-----------------------------
✅ Garbage detection: PHP code, leaked HTML documents, Weibo patterns, Chinese system
   prompts, generic greetings, empty responses, missing JSON structure
✅ Cooldown marking: Writes to router.cooldown_cache directly
✅ Router respects cooldown: Dead deployments are not selected again
✅ Fallback chains: FASTER → qwen3x → kimi2 → ds3x → cerebras → longcat → qwen-coder
✅ Graceful degradation: Returns 200 OK even when garbage detected
✅ Streaming support: async_log_stream_complete_event for stream: true requests

DEBUGGING JOURNEY (for future reference)
-----------------------------------------
Attempt 1: Used _set_cooldown_deployments() → FAILED
  - Function is gated by fail counter, needs threshold to be exceeded first
  - Returns False on first call, never writes to cooldown cache

Attempt 2: Write to router.cache with key "deployment:{id}:cooldown" → FAILED
  - Wrong cache! Router has TWO caches:
    - router.cache (response/latency cache) ❌
    - router.cooldown_cache (CooldownCache instance) ✅
  - Selection logic reads from cooldown_cache, not response cache

Attempt 3: Direct cooldown_cache.add_deployment_to_cooldown() → SUCCESS ✅
  - Bypasses counter-based gating
  - Writes to correct cache that router checks during selection
  - Use SHA256 deployment ID from kwargs["litellm_params"]["model_info"]["id"]

REQUIREMENTS
------------
- LiteLLM proxy config must have:
  - router_settings.cooldown_time: 600 (matches COOLDOWN_SECONDS here)
  - router_settings.allowed_fails: 1
  - router_settings.allowed_fails_policy.InternalServerErrorAllowedFails: 1
  - litellm_settings.callbacks: ["handler.custom_handler"]
- litellm_start.cmd must redirect output to litellm.log for debugging

CONFIG FILE LOCATION
--------------------
C:\portable\_scripts\LiteLLM\config.yaml
"""

import litellm
from litellm.integrations.custom_logger import CustomLogger
import re


class GarbageResponseHandler(CustomLogger):
    """
    Detects garbage responses and marks deployments as dead.

    Patterns matched:
    - PHP code (<?php, namespace App\\, Illuminate\\)
    - Leaked HTML documents (<!DOCTYPE at start, <html at start)
    - System prompt leaks (<system>, 你是一个.*?AI.*?助手)
    - Empty responses (no content, no reasoning)
    - Generic greetings ("Hi! How can I help you today")
    - Missing JSON structure (when JSON format was requested but no brackets found)
    - Streaming responses (via async_log_stream_complete_event)
    """

    GARBAGE_PATTERNS = [
        r'<\?php',
        r'<!--\s*BEGIN\s+WEIBO',
        r'namespace\s+App\\',
        r'class\s+\w+Controller',
        r'Illuminate\\',
        r'use\s+Illuminate\\',
        r'<system>',
        r'</system>',
        r'你是一个专业的AI助手',
        r'你是一个.*?AI.*?助手',
        r'共\s+\d+\s+条',
        r'Hi there! How can I help you today?',
        r'Hi! How can I help you today',
    ]

    MIN_CONTENT_LENGTH = 15
    COOLDOWN_SECONDS = 600

    def _is_empty(self, response_obj):
        if not hasattr(response_obj, "choices") or not response_obj.choices:
            return True
        message = response_obj.choices[0].message
        content = getattr(message, "content", None) or ""
        reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None) or ""
        return not content.strip() and not reasoning.strip()

    def _looks_like_garbage(self, content: str, expect_json: bool = False) -> tuple[bool, str]:
        if not content or len(content.strip()) < self.MIN_CONTENT_LENGTH:
            return True, "too_short"
        content_lower = content.lower()
        for pattern in self.GARBAGE_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                return True, f"pattern_match:{pattern}"
        # Only flag full leaked HTML documents, not code snippets
        if re.match(r'^\s*(?:<!DOCTYPE|<html|<\?xml|<body)', content, re.IGNORECASE):
            return True, "leaked_html_document"
        # Loose JSON check: only flag if model completely failed to produce any
        # JSON structure. Client-side repair handles syntax/truncation issues.
        if expect_json:
            if '{' not in content and '[' not in content:
                return True, "missing_json_structure"
        return False, ""

    def _expects_json(self, kwargs: dict) -> bool:
        litellm_params = kwargs.get("litellm_params", {})
        response_format = litellm_params.get("response_format") or kwargs.get("response_format", {})
        if isinstance(response_format, dict):
            return response_format.get("type") == "json_schema"
        return False

    def _get_content(self, response_obj) -> str:
        if not hasattr(response_obj, "choices") or not response_obj.choices:
            return ""
        message = response_obj.choices[0].message
        content = getattr(message, "content", None) or ""
        reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None) or ""
        return (content or "") + " " + (reasoning or "")

    def _get_deployment_id(self, kwargs) -> str:
        model_id = kwargs.get("litellm_params", {}).get("model_info", {}).get("id", "")
        if model_id and len(model_id) >= 20:
            return model_id
        print("[GarbageResponseHandler] Warning: No valid model_info.id found, skipping cooldown")
        return ""

    def _mark_deployment_dead(self, deployment_id: str, reason: str):
        print(f"[GarbageResponseHandler] Marking deployment {deployment_id[:12]}... as DEAD (reason: {reason})")
        try:
            from litellm.proxy.proxy_server import llm_router
            if llm_router is None or not hasattr(llm_router, 'cooldown_cache'):
                print("[GarbageResponseHandler] Warning: router.cooldown_cache not available")
                return
            fake_exception = litellm.InternalServerError(
                message=f"Garbage response: {reason}",
                model=deployment_id,
                llm_provider="",
            )
            try:
                llm_router.cooldown_cache.add_deployment_to_cooldown(
                    model_id=deployment_id,
                    original_exception=fake_exception,
                    exception_status=500,
                    cooldown_time=float(self.COOLDOWN_SECONDS),
                )
            except TypeError:
                llm_router.cooldown_cache.add_deployment_to_cooldown(
                    deployment_id=deployment_id,
                    original_exception=fake_exception,
                    exception_status=500,
                    cooldown_time=float(self.COOLDOWN_SECONDS),
                )
            print(f"[GarbageResponseHandler] Deployment marked dead for {self.COOLDOWN_SECONDS}s")
        except Exception as e:
            print(f"[GarbageResponseHandler] Warning: Could not mark deployment dead: {e}")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        deployment_id = self._get_deployment_id(kwargs)
        if not deployment_id:
            return

        if self._is_empty(response_obj):
            self._mark_deployment_dead(deployment_id, "empty_response")
            return

        content = self._get_content(response_obj)
        is_garbage, reason = self._looks_like_garbage(content, self._expects_json(kwargs))
        if is_garbage:
            self._mark_deployment_dead(deployment_id, reason)

    async def async_log_stream_complete_event(self, kwargs, response_obj, start_time, end_time):
        """Hook for streaming requests. response_obj contains the aggregated response."""
        deployment_id = self._get_deployment_id(kwargs)
        if not deployment_id:
            return

        content = self._get_content(response_obj)

        if not content.strip():
            self._mark_deployment_dead(deployment_id, "empty_stream_response")
            return

        is_garbage, reason = self._looks_like_garbage(content, self._expects_json(kwargs))
        if is_garbage:
            self._mark_deployment_dead(deployment_id, f"stream_{reason}")


custom_handler = GarbageResponseHandler()

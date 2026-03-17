import litellm
from litellm.integrations.custom_logger import CustomLogger
import re
import json
import time


class GarbageResponseHandler(CustomLogger):
    """Detects garbage responses (empty, HTML, PHP, non-JSON) and triggers provider cooldown.

    Strategy: When garbage is detected, mark the deployment as dead in router cache
    and return an empty-but-valid response. This avoids 500 errors and allows the
    router to select a different model on retry.
    """

    # Patterns that indicate garbage/training data leakage
    GARBAGE_PATTERNS = [
        r'<\?php',                      # PHP code
        r'<!--\s*BEGIN\s+WEIBO',         # Weibo HTML template
        r'<!DOCTYPE\s+html',             # HTML documents
        r'<html[^>]*>',                  # HTML tags
        r'<\s*head\s*>',                 # HTML head
        r'namespace\s+App\\',            # Laravel/PHP namespace
        r'class\s+\w+Controller',        # PHP controller class
        r'Illuminate\\',                 # Laravel framework
        r'use\s+Illuminate\\',           # Laravel imports
        r'<system>',                     # System prompt leak (LLM returning its own instructions)
        r'</system>',                    # System prompt leak closing tag
        r'你是一个专业的AI助手',          # Chinese system prompt template
        r'你是一个.*?AI.*?助手',          # Generic Chinese "you are an AI assistant" pattern
        r'共\s+\d+\s+条',                # Weibo-style count ("共 0 条")
        r'Hi there! How can I help you today?',  # Generic chat bot greeting (not following instructions)
        r'Hi! How can I help you today',         # Generic chat bot greeting variant
    ]

    # Responses shorter than this are likely garbage (unless empty is valid)
    MIN_CONTENT_LENGTH = 15

    # Cooldown duration in seconds (must match router_settings.cooldown_time)
    COOLDOWN_SECONDS = 600

    def _is_empty(self, response_obj):
        """Check if response has no meaningful content."""
        if not hasattr(response_obj, "choices") or not response_obj.choices:
            return True
        message = response_obj.choices[0].message
        content = getattr(message, "content", None) or ""
        reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None) or ""
        return not content.strip() and not reasoning.strip()

    def _looks_like_garbage(self, content: str, expect_json: bool = False) -> tuple[bool, str]:
        """
        Check if content looks like garbage/training data leakage.
        Returns (is_garbage, reason).
        """
        if not content or len(content.strip()) < self.MIN_CONTENT_LENGTH:
            return True, "too_short"

        content_lower = content.lower()

        # Check for known garbage patterns
        for pattern in self.GARBAGE_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                return True, f"pattern_match:{pattern}"

        # Check for HTML-like content (unescaped tags, not markdown)
        # Allow common markdown like <thought> but reject raw HTML
        if re.search(r'<\s*\/?[a-z][a-z0-9]*(?:\s[^>]*)?>', content):
            # Exclude allowed tags (thinking tags, markdown-like structures)
            allowed_tags = {'thought', 'think', 'thinking', 'reasoning'}
            if not any(f'<{tag}' in content_lower for tag in allowed_tags):
                # Look for HTML attribute patterns (more likely to be real HTML)
                if re.search(r'<\w+\s+\w+\s*=', content):
                    return True, "html_content"

        # Check for invalid JSON when JSON format was requested
        if expect_json:
            content_stripped = content.strip()
            if content_stripped.startswith(('{', '[')):
                try:
                    json.loads(content)
                except json.JSONDecodeError:
                    return True, "invalid_json"

        return False, ""

    def _expects_json(self, kwargs: dict) -> bool:
        """Check if the request expected JSON response format."""
        litellm_params = kwargs.get("litellm_params", {})
        response_format = litellm_params.get("response_format") or kwargs.get("response_format", {})
        if isinstance(response_format, dict):
            return response_format.get("type") == "json_schema"
        return False

    def _get_content(self, response_obj) -> str:
        """Extract content from response object."""
        if not hasattr(response_obj, "choices") or not response_obj.choices:
            return ""
        message = response_obj.choices[0].message
        content = getattr(message, "content", None) or ""
        reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None) or ""
        return (content or "") + " " + (reasoning or "")

    def _mark_deployment_dead(self, deployment_id: str, reason: str):
        """Mark a deployment as dead using LiteLLM's cooldown cache."""
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

            llm_router.cooldown_cache.add_deployment_to_cooldown(
                model_id=deployment_id,
                original_exception=fake_exception,
                exception_status=500,
                cooldown_time=float(self.COOLDOWN_SECONDS),
            )
            print(f"[GarbageResponseHandler] Deployment marked dead for {self.COOLDOWN_SECONDS}s")

        except Exception as e:
            print(f"[GarbageResponseHandler] Warning: Could not mark deployment dead: {e}")

    def _get_deployment_id(self, kwargs, response_obj) -> str:
        """Extract deployment SHA256 ID from kwargs."""
        model_id = kwargs.get("litellm_params", {}).get("model_info", {}).get("id", "")
        # Only return if it's a real SHA256 hex string (64 chars)
        if model_id and len(model_id) >= 20:
            return model_id
        print(f"[GarbageResponseHandler] Warning: No valid model_info.id found, skipping cooldown")
        return ""

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Async path: check response quality, mark deployment dead if garbage."""
        if self._is_empty(response_obj):
            deployment_id = self._get_deployment_id(kwargs, response_obj)
            if deployment_id:
                self._mark_deployment_dead(deployment_id, "empty_response")
            return

        content = self._get_content(response_obj)
        expect_json = self._expects_json(kwargs)
        is_garbage, reason = self._looks_like_garbage(content, expect_json)

        if is_garbage:
            deployment_id = self._get_deployment_id(kwargs, response_obj)
            if deployment_id:
                self._mark_deployment_dead(deployment_id, reason)

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        """Proxy-level guard: marks garbage responses as dead before they reach the client."""
        if not isinstance(data, dict):
            return response

        if self._is_empty(response):
            deployment_id = self._get_deployment_id(data, response)
            if deployment_id:
                self._mark_deployment_dead(deployment_id, "empty_response")
            return response

        content = self._get_content(response)
        expect_json = self._expects_json(data)
        is_garbage, reason = self._looks_like_garbage(content, expect_json)

        if is_garbage:
            deployment_id = self._get_deployment_id(data, response)
            if deployment_id:
                self._mark_deployment_dead(deployment_id, reason)

        return response


# Create handler instance
custom_handler = GarbageResponseHandler()

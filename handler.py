import litellm
from litellm.integrations.custom_logger import CustomLogger
import re
import json


class GarbageResponseHandler(CustomLogger):
    """Detects garbage responses (empty, HTML, PHP, non-JSON) and triggers provider cooldown."""

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
    ]

    # Responses shorter than this are likely garbage (unless empty is valid)
    MIN_CONTENT_LENGTH = 15

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

    def _trigger_cooldown(self, model: str, reason: str):
        """
        Raise litellm InternalServerError to trigger router cooldown.
        The router will automatically add this model to cooldown based on
        allowed_fails_policy and cooldown_time settings.
        """
        print(f"[GarbageResponseHandler] Marking {model} as DEAD (reason: {reason})")

        # Raise InternalServerError to trigger router's cooldown mechanism
        raise litellm.InternalServerError(
            message=f"Garbage response detected: {reason}",
            model=model,
            llm_provider="unknown",  # Will be filled by router
        )

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Sync path: check response quality, trigger cooldown if garbage."""
        if self._is_empty(response_obj):
            model = kwargs.get("model", "unknown")
            self._trigger_cooldown(model, "empty_response")

        content = self._get_content(response_obj)
        expect_json = self._expects_json(kwargs)
        is_garbage, reason = self._looks_like_garbage(content, expect_json)
        if is_garbage:
            model = kwargs.get("model", "unknown")
            self._trigger_cooldown(model, reason)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Async path: same checks for async completion."""
        if self._is_empty(response_obj):
            model = kwargs.get("model", "unknown")
            self._trigger_cooldown(model, "empty_response")

        content = self._get_content(response_obj)
        expect_json = self._expects_json(kwargs)
        is_garbage, reason = self._looks_like_garbage(content, expect_json)
        if is_garbage:
            model = kwargs.get("model", "unknown")
            self._trigger_cooldown(model, reason)

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        """Proxy-level guard: rejects garbage before it reaches the client."""
        # DEBUG: Log that we're checking
        model = data.get("model", "unknown") if isinstance(data, dict) else "unknown"
        print(f"[GarbageResponseHandler] Checking response from {model}")

        if self._is_empty(response):
            self._trigger_cooldown(model, "empty_response")

        content = self._get_content(response)
        expect_json = self._expects_json(data) if isinstance(data, dict) else False

        # DEBUG: Log content preview
        preview = content[:200] if content else "<empty>"
        print(f"[GarbageResponseHandler] Content preview: {preview}")

        is_garbage, reason = self._looks_like_garbage(content, expect_json)
        if is_garbage:
            print(f"[GarbageResponseHandler] GARBAGE DETECTED: {reason}")
            self._trigger_cooldown(model, reason)
        else:
            print(f"[GarbageResponseHandler] Content OK")

        return response


# Create handler instance
custom_handler = GarbageResponseHandler()

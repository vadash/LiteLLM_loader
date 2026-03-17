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

    def _looks_like_garbage(self, content: str) -> tuple[bool, str]:
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

        # Check for non-JSON when response format is set to JSON
        # This catches models that return plain text when JSON is expected
        if 'response_format' in litellm.get_current_llm_module_vars():
            # Try to parse as JSON - if it fails and doesn't look like markdown,
            # it's likely garbage
            content_stripped = content.strip()
            if content_stripped.startswith(('{', '[')):
                try:
                    json.loads(content)
                except json.JSONDecodeError:
                    return True, "invalid_json"

        return False, ""

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
        is_garbage, reason = self._looks_like_garbage(content)
        if is_garbage:
            model = kwargs.get("model", "unknown")
            self._trigger_cooldown(model, reason)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Async path: same checks for async completion."""
        if self._is_empty(response_obj):
            model = kwargs.get("model", "unknown")
            self._trigger_cooldown(model, "empty_response")

        content = self._get_content(response_obj)
        is_garbage, reason = self._looks_like_garbage(content)
        if is_garbage:
            model = kwargs.get("model", "unknown")
            self._trigger_cooldown(model, reason)

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        """Proxy-level guard: rejects garbage before it reaches the client."""
        if self._is_empty(response):
            model = data.get("model", "unknown") if isinstance(data, dict) else "unknown"
            self._trigger_cooldown(model, "empty_response")

        content = self._get_content(response)
        is_garbage, reason = self._looks_like_garbage(content)
        if is_garbage:
            model = data.get("model", "unknown") if isinstance(data, dict) else "unknown"
            self._trigger_cooldown(model, reason)

        return response


# Create handler instance
custom_handler = GarbageResponseHandler()

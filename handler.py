import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_utils.cooldown_handlers import _set_cooldown_deployments
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
        """Mark a deployment as dead using LiteLLM's cooldown mechanism."""
        print(f"[GarbageResponseHandler] Marking deployment {deployment_id[:12]}... as DEAD (reason: {reason})")

        try:
            from litellm.proxy.proxy_server import llm_router

            if llm_router is None:
                print("[GarbageResponseHandler] Warning: llm_router not available")
                return

            # Create a synthetic exception that maps to InternalServerError
            # This triggers InternalServerErrorAllowedFails policy (set to 1 in config)
            fake_exception = litellm.InternalServerError(
                message=f"Garbage response detected: {reason}",
                model=deployment_id,
                llm_provider="",
            )

            result = _set_cooldown_deployments(
                litellm_router_instance=llm_router,
                original_exception=fake_exception,
                exception_status=500,
                deployment=deployment_id,
                time_to_cooldown=float(self.COOLDOWN_SECONDS),
            )

            if result:
                print(f"[GarbageResponseHandler] Deployment marked dead for {self.COOLDOWN_SECONDS}s")
            else:
                print(f"[GarbageResponseHandler] Cooldown not applied (allowed_fails threshold not yet exceeded)")

        except ImportError as e:
            print(f"[GarbageResponseHandler] Warning: Could not import llm_router: {e}")
        except Exception as e:
            print(f"[GarbageResponseHandler] Warning: Could not mark deployment dead: {e}")

    def _get_deployment_id(self, kwargs, response_obj) -> str:
        """Extract deployment ID from kwargs (SHA256 hash) or fallback to model name."""
        # litellm_params contains the internal model_id assigned by the router
        litellm_params = kwargs.get("litellm_params", {})
        model_id = litellm_params.get("model_info", {}).get("id", "")
        if model_id:
            return model_id
        # Fallback to model name from response
        return getattr(response_obj, 'model', '')

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Async path: check response quality, mark deployment dead if garbage."""
        model = kwargs.get("model", "unknown")

        if self._is_empty(response_obj):
            deployment_id = self._get_deployment_id(kwargs, response_obj) or model
            self._mark_deployment_dead(deployment_id, "empty_response")
            return

        content = self._get_content(response_obj)
        expect_json = self._expects_json(kwargs)
        is_garbage, reason = self._looks_like_garbage(content, expect_json)

        if is_garbage:
            deployment_id = self._get_deployment_id(kwargs, response_obj) or model
            self._mark_deployment_dead(deployment_id, reason)

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        """Proxy-level guard: marks garbage responses as dead before they reach the client."""
        # data here is kwargs dict in proxy context
        if not isinstance(data, dict):
            return response

        model = data.get("model", "unknown")

        if self._is_empty(response):
            deployment_id = self._get_deployment_id(data, response) or model
            self._mark_deployment_dead(deployment_id, "empty_response")
            return response

        content = self._get_content(response)
        expect_json = self._expects_json(data)
        is_garbage, reason = self._looks_like_garbage(content, expect_json)

        if is_garbage:
            deployment_id = self._get_deployment_id(data, response) or model
            self._mark_deployment_dead(deployment_id, reason)

        return response


# Create handler instance
custom_handler = GarbageResponseHandler()

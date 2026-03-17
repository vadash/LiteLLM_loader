import litellm
from litellm.integrations.custom_logger import CustomLogger
import re
import json


class GarbageResponseHandler(CustomLogger):

    GARBAGE_PATTERNS = [
        r'<\?php',
        r'<!--\s*BEGIN\s+WEIBO',
        r'<!DOCTYPE\s+html',
        r'<html[^>]*>',
        r'<\s*head\s*>',
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
        if re.search(r'<\s*\/?[a-z][a-z0-9]*(?:\s[^>]*)?>', content):
            allowed_tags = {'thought', 'think', 'thinking', 'reasoning'}
            if not any(f'<{tag}' in content_lower for tag in allowed_tags):
                if re.search(r'<\w+\s+\w+\s*=', content):
                    return True, "html_content"
        if expect_json:
            content_stripped = content.strip()
            if content_stripped.startswith(('{', '[')):
                try:
                    json.loads(content)
                except json.JSONDecodeError:
                    return True, "invalid_json"
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


custom_handler = GarbageResponseHandler()

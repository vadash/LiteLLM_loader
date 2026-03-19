"""
LiteLLM Garbage Response Handler
================================

WHAT THIS DOES
--------------
Detects garbage responses from low-quality LLMs (training data leakage, HTML
templates, empty responses, system prompt leaks, AI refusals) and marks the 
deployment as "dead" for 10 minutes. Router will skip dead deployments and use fallbacks.

REQUIREMENTS
------------
- LiteLLM proxy config must have:
  - router_settings.cooldown_time: 900 (matches COOLDOWN_SECONDS here)
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
    """

    # Patterns indicating raw training data, code leakage, or system leaks
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
        r'Hi there! How can I help you today\?',
        r'Hi! How can I help you today\?',
        
        # Useless responses: AI waiting for input (ignoring the provided prompt)
        r'Please provide the (?:text|context|details) you(?: would like|\'d like)',
        r'Пожалуйста, предоставьте (?:текст|контекст)',
        r'请提供(?:相关|更多)?的(?:文本|信息|上下文)',
    ]

    # Patterns indicating AI refusals and policy disclaimers.
    # Anchored tightly to AI-specific phrasing to avoid false positives on RP dialogue.
    REFUSAL_PATTERNS = [
        # English
        r"(?:As an|I am an) AI (?:language model|assistant)",
        r"I can(?:not|'t) (?:fulfill|comply with|process) (?:this|your) request",
        r"against my (?:programming|guidelines|safety|ethical|core) (?:principles|policies|guidelines)",
        r"violate(?:s)? (?:safety|OpenAI|Anthropic|guidelines|policies)",
        r"I can(?:not|'t) (?:generate|create|provide|write) (?:content|text|responses|stories|JSON) that (?:is|depicts|involves)",
        r"I(?:'m| am) sorry, (?:but )?I can(?:not|'t) (?:assist|help) with that",
        
        # Russian
        r"Как искусственный интеллект",
        r"Я не могу (?:выполнить|обработать) (?:этот )?запрос",
        r"противоречит (?:моим )?(?:правилам|политике|этическим)",
        r"нарушает (?:правила|политику|принципы) (?:безопасности|OpenAI|Anthropic)",
        r"Я не могу (?:создавать|генерировать|предоставлять) контент, который",
        r"Извините, но я не могу (?:помочь|выполнить|сгенерировать|предоставить)",
        
        # Chinese
        r"作为一个(?:人工智能|AI|语言模型)",
        r"抱歉，我无法(?:满足|处理)(?:您|你)的请求",
        r"我无法为(?:您|你)(?:生成|提供|创建)",
        r"违反(?:了)?(?:相关|使用)?(?:政策|规定|准则|法律|安全)",
        r"我不能(?:协助|提供)(?:此类|这方面)的",
    ]

    MIN_CONTENT_LENGTH = 15
    COOLDOWN_SECONDS = 900

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
            
        # Check standard garbage/leak patterns
        for pattern in self.GARBAGE_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                return True, f"garbage_match:{pattern[:20]}..."
                
        # Check AI refusal/disclaimer patterns
        for pattern in self.REFUSAL_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                return True, f"refusal_match:{pattern[:20]}..."

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
        litellm_params = kwargs.get("litellm_params") or {}
        model_info = litellm_params.get("model_info") or {}
        model_id = model_info.get("id", "")
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
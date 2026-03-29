import litellm
from litellm.integrations.custom_logger import CustomLogger
import re
import pathlib
from datetime import datetime

LOG_FILE = pathlib.Path(__file__).parent / "litellm.log"

def log_to_file(message: str):
    """Append a log message with timestamp to litellm.log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass


class UniversalGarbageHandler(CustomLogger):
    """
    A universal LiteLLM plugin that detects AI safety refusals, empty JSON, 
    hallucination loops, and raw training data leaks. It instantly marks the 
    offending deployment as 'dead' in the router to force an immediate fallback.
    """

    # 10 minutes (must match router_settings.cooldown_time in config.yaml.md)
    COOLDOWN_SECONDS = 600  
    MIN_CONTENT_LENGTH = 10

    # Universal patterns for models getting stuck in conversational loops or leaking raw data
    GARBAGE_PATTERNS = [
        r"<\?php",                                          # PHP code leak
        r"(?:分享到微博|新浪微博|转发微博)",                # Weibo scraping template leak
        r"(?i)^how can I (?:help|assist) you(?: today)?\?", # Assistant amnesia loop
        r"(?i)^please provide the text",                    # Assistant amnesia loop
        r"(?i)^I am ready\.",                               # Assistant amnesia loop
    ]

    # Universal patterns for AI safety/alignment refusals (En, Ru, Zh)
    REFUSAL_PATTERNS = [
        # English
        r"(?:As an|I am an) AI (?:language model|assistant)",
        r"I can(?:not|'t) (?:fulfill|comply with|process) (?:this|your) request",
        r"against my (?:programming|guidelines|safety|ethical|core) (?:principles|policies|guidelines)",
        r"violate(?:s)? (?:safety|OpenAI|Anthropic|guidelines|policies)",
        r"I can(?:not|'t) (?:generate|create|provide|write) (?:content|text|responses|stories|JSON) that (?:is|depicts|involves|contains?)",
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

    def _get_response_preview(self, response_obj, max_chars: int = 200) -> str:
        """Safely extract the first N chars of the response for logging."""
        try:
            if not hasattr(response_obj, "choices") or not response_obj.choices:
                return "<no_choices>"
            message = response_obj.choices[0].message
            content = getattr(message, "content", None) or ""
            reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None) or ""
            combined = content + reasoning
            preview = combined[:max_chars].replace('\n', '\\n').replace('\r', '\\r')
            if len(combined) > max_chars:
                preview += "..."
            return preview
        except Exception as e:
            return f"<error_extracting_preview: {e}>"

    def _expects_json(self, kwargs: dict) -> bool:
        """
        Universally detects if the API client expects a JSON response.
        Checks explicit API params (response_format) AND common prompt instructions.
        """
        # 1. Check strict API parameters (json_schema or json_object)
        litellm_params = kwargs.get("litellm_params", {})
        response_format = litellm_params.get("response_format") or kwargs.get("response_format", {})
        
        if isinstance(response_format, dict):
            req_type = response_format.get("type", "")
            if req_type in ["json_schema", "json_object"]:
                return True

        # 2. Universal Prompt Heuristics (Checks if the user/system explicitly demanded JSON)
        messages = kwargs.get("messages", [])
        for msg in messages:
            content = str(msg.get("content", "")).lower()
            if "output json" in content or "return json" in content or "```json" in content or "输出类型：json" in content:
                return True
                
        return False

    def _looks_like_garbage(self, response_obj, kwargs: dict) -> tuple[bool, str]:
        """
        Evaluates the response. Returns (True, reason) if it should be marked dead.
        """
        if not hasattr(response_obj, "choices") or not response_obj.choices:
            return True, "no_choices_in_response"
            
        message = response_obj.choices[0].message
        actual_content = getattr(message, "content", None) or ""
        reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None) or ""
        
        # We check both content and reasoning blocks together for refusals/garbage
        combined_text = actual_content + " " + reasoning

        # 1. Check for AI Refusals
        for pattern in self.REFUSAL_PATTERNS:
            if re.search(pattern, combined_text, re.IGNORECASE):
                return True, f"refusal_match:{pattern[:20]}..."

        # 2. Check for Hallucination/Garbage Loops
        for pattern in self.GARBAGE_PATTERNS:
            if re.search(pattern, combined_text, re.IGNORECASE):
                return True, f"garbage_match:{pattern[:20]}..."

        # 3. Check for HTML/Web-Scraping Leaks (Model broke and served a raw webpage)
        if re.match(r'^\s*(?:<!DOCTYPE|<html|<\?xml|<body)', combined_text, re.IGNORECASE):
            return True, "leaked_html_document"
            
        # 4. Strict JSON Structure Validation
        # If the app requested JSON, the actual_content MUST contain brackets. 
        # If a model trips a safety filter AFTER thinking, actual_content will be empty.
        if self._expects_json(kwargs):
            if not actual_content.strip():
                return True, "safety_trip_empty_json_content"
            if '{' not in actual_content and '[' not in actual_content:
                return True, "missing_json_structure_in_content"
        else:
            # Standard conversational length check
            if len(combined_text.strip()) < self.MIN_CONTENT_LENGTH:
                return True, "response_too_short_or_empty"
                
        return False, ""

    def _get_deployment_id(self, kwargs) -> str:
        """Extracts the exact LiteLLM router deployment hash."""
        litellm_params = kwargs.get("litellm_params") or {}
        model_info = litellm_params.get("model_info") or {}
        model_id = model_info.get("id", "")
        if model_id and len(model_id) >= 20:
            return model_id
        return ""

    def _mark_deployment_dead(self, deployment_id: str, reason: str):
        """
        Injects a synthetic HTTP 500 error directly into LiteLLM's cooldown cache.
        This forces the router to instantly failover to the next fallback model.
        """
        msg = f"[UniversalGarbageHandler] Marking deployment {deployment_id[:12]}... as DEAD (reason: {reason})"
        print(msg)
        log_to_file(f"[DEPLOYMENT_DEAD] deployment={deployment_id[:16]}... reason={reason}")
        
        try:
            from litellm.proxy.proxy_server import llm_router
            if llm_router is None or not hasattr(llm_router, 'cooldown_cache'):
                return
                
            fake_exception = litellm.InternalServerError(
                message=f"Garbage response detected: {reason}",
                model=deployment_id,
                llm_provider="",
            )
            
            # API changes across LiteLLM versions (model_id vs deployment_id)
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
        except Exception as e:
            log_to_file(f"[ERROR] Failed to mark deployment dead: {e}")

    # =========================================================================
    # LiteLLM Event Hooks
    # =========================================================================

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Hook for standard (non-streaming) completions."""
        deployment_id = self._get_deployment_id(kwargs)
        model_name = kwargs.get("model", "unknown")
        preview = self._get_response_preview(response_obj)
        
        log_to_file(f"[RESPONSE] model={model_name} deploy={deployment_id[:12]} preview={preview!r}")

        if not deployment_id:
            return

        is_garbage, reason = self._looks_like_garbage(response_obj, kwargs)
        if is_garbage:
            log_to_file(f"[GARBAGE_DETECTED] model={model_name} reason={reason}")
            self._mark_deployment_dead(deployment_id, reason)

    async def async_log_stream_complete_event(self, kwargs, response_obj, start_time, end_time):
        """Hook for streaming completions (evaluated after the stream finishes)."""
        deployment_id = self._get_deployment_id(kwargs)
        model_name = kwargs.get("model", "unknown")
        preview = self._get_response_preview(response_obj)
        
        log_to_file(f"[STREAM_COMPLETE] model={model_name} deploy={deployment_id[:12]} preview={preview!r}")

        if not deployment_id:
            return

        is_garbage, reason = self._looks_like_garbage(response_obj, kwargs)
        if is_garbage:
            log_to_file(f"[GARBAGE_DETECTED] model={model_name} reason={reason}")
            self._mark_deployment_dead(deployment_id, f"stream_{reason}")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Hook for logging natural provider failures (HTTP 400, 429, 502)."""
        deployment_id = self._get_deployment_id(kwargs)
        model_name = kwargs.get("model", "unknown")
        
        exception = response_obj
        exception_type = type(exception).__name__ if exception else "Unknown"
        exception_msg = str(exception)[:200] if exception else "no details"
        
        log_to_file(f"[PROVIDER_FAILURE] model={model_name} deploy={deployment_id[:12]} error={exception_type}: {exception_msg!r}")


# Register the plugin instance so LiteLLM can hook into it
custom_handler = UniversalGarbageHandler()

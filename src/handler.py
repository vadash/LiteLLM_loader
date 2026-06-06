import litellm
from litellm.integrations.custom_logger import CustomLogger
import re
import pathlib
import logging
from datetime import datetime

# =========================================================================
# SILENCE NOISY LITELLM TRACEBACKS IN THE CONSOLE
# =========================================================================
# By default, LiteLLM prints full traceback logs every time a request fails 
# and switches to a fallback. Under heavy load, this can clutter stdout.
# This filter suppresses redundant warnings during normal fallback operations.
# =========================================================================
class SuppressNoisyRouterErrors(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        # Suppress standard fallback warning tracebacks
        if "Error occurred while trying to do fallbacks" in msg:
            return False
        # Suppress orphaned tracebacks associated with gateway or format drops
        if "Traceback (most recent call last):" in msg and "OpenAIException" in msg:
            return False
        return True

# Attach the custom filter to LiteLLM's internal system loggers
logging.getLogger("LiteLLM Router").addFilter(SuppressNoisyRouterErrors())
logging.getLogger("LiteLLM").addFilter(SuppressNoisyRouterErrors())
logging.getLogger("litellm").addFilter(SuppressNoisyRouterErrors())
# =========================================================================

LOG_FILE = pathlib.Path(__file__).parent / "litellm.log"

def log_to_file(message: str):
    """Utility function to write timestamped log entries to litellm.log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass


class UniversalGarbageHandler(CustomLogger):
    """
    A LiteLLM custom callback class. 
    
    This class hooks into the router lifecycle to:
    1. Translate virtual models (FAST/SMART) into active endpoints before calling them.
    2. Inspect incoming LLM outputs (both standard completions and streams).
    3. Programmatically quarantine (cooldown) any deployment returning garbage/refusals.
    """

    # Cooldown duration: must match router_settings.cooldown_time in config.yaml
    COOLDOWN_SECONDS = 600  

    # regular expressions to identify common system alignment/moderation refusals
    REFUSAL_PATTERNS = [
        # English Refusal Patterns
        r"I can(?:not|'t) (?:fulfill|comply with|process) (?:this|your) request",
        r"against my (?:programming|guidelines|safety|ethical|core) (?:principles|policies|guidelines)",
        r"violate(?:s)? (?:safety|OpenAI|Anthropic|guidelines|policies)",
        r"I can(?:not|'t) (?:generate|create|provide|write) (?:content|text|responses|stories|JSON) that (?:is|depicts|involves|contains?)",
        r"I(?:'m| am) sorry, (?:but )?I can(?:not|'t) (?:assist|help) with that",
        
        # Russian Refusal Patterns
        r"Я не могу (?:выполнить|обработать) (?:этот )?запрос",
        r"противоречит (?:моим )?(?:правилам|политике|этическим)",
        r"нарушает (?:правила|политику|принципы) (?:безопасности|OpenAI|Anthropic)",
        r"Я не могу (?:создавать|генерировать|предоставлять) контент, который",
        r"Извините, но я не могу (?:помочь|выполнить|сгенерировать|предоставить)",
        
        # Chinese Refusal Patterns
        r"作为一个(?:人工智能|AI|语言模型)",
        r"抱歉，我无法(?:满足|处理)(?:您|你)的请求",
        r"我无法为(?:您|你)(?:生成|提供|创建)",
        r"违反(?:了)?(?:相关|使用)?(?:政策|规定|准则|法律|安全)",
        r"我不能(?:协助|提供)(?:此类|这方面)的",
    ]

    # Add loop detection / gibberish detection signatures here as needed
    GARBAGE_PATTERNS = []

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle Hook 1: Pre-Call Request Interception
    # ─────────────────────────────────────────────────────────────────────────
    async def async_pre_call_hook(self, user_api_key_dict: dict, data: dict, call_type: str, *args, **kwargs):
        """
        Pre-call hook to intercept virtual/entry model requests (FAST/SMART)
        and map them to active model groups and fallback arrays dynamically.
        This avoids hitting dummy deployments and triggering mock responses.
        
        Using *args and **kwargs makes this signature fully compatible with 
        varied versions of LiteLLM (which may pass 'cache' or 'cache_dict' keywords).
        """
        model = data.get("model")
        if model == "FAST":
            # Change the destination model name
            data["model"] = "gemma4"
            # Explicitly define fallback models for this specific request
            data["fallbacks"] = ["zai_free", "nvidia", "longcat", "zai"]
            log_to_file(f"[ROUTER_REWRITE] virtual_model=FAST -> target=gemma4 fallbacks={data['fallbacks']}")
        elif model == "SMART":
            data["model"] = "zai"
            data["fallbacks"] = ["nvidia", "gemma4", "longcat"]
            log_to_file(f"[ROUTER_REWRITE] virtual_model=SMART -> target=zai fallbacks={data['fallbacks']}")

    def _get_response_preview(self, response_obj, max_chars: int = 200) -> str:
        """Safely parses content and reasoning sections to return a concise log preview."""
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
        Heuristic function to check if the caller expects structural JSON.
        Evaluates system arguments and user prompts.
        """
        # Step 1: Check standard response_format parameters
        litellm_params = kwargs.get("litellm_params", {})
        response_format = litellm_params.get("response_format") or kwargs.get("response_format", {})
        
        if isinstance(response_format, dict):
            req_type = response_format.get("type", "")
            if req_type in ["json_schema", "json_object"]:
                return True

        # Step 2: Check for explicit prompts requesting JSON format
        messages = kwargs.get("messages", [])
        for msg in messages:
            content = str(msg.get("content", "")).lower()
            if "output json" in content or "return json" in content or "```json" in content or "输出类型：json" in content:
                return True
                
        return False

    def _looks_like_garbage(self, response_obj, kwargs: dict) -> tuple[bool, str]:
        """
        Inspects model responses. 
        Returns (True, reason) if output is detected as garbage/refusal.
        """
        if not hasattr(response_obj, "choices") or not response_obj.choices:
            return True, "no_choices_in_response"
            
        message = response_obj.choices[0].message
        actual_content = getattr(message, "content", None) or ""
        reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None) or ""
        combined_text = actual_content + " " + reasoning

        # Guard: Ignore dummy mock responses so they don't trigger unexpected errors
        if kwargs.get("model") in ["FAST", "SMART"] or actual_content.strip() == "error":
            return False, ""

        # Check 1: Refusal Matching
        for pattern in self.REFUSAL_PATTERNS:
            if re.search(pattern, combined_text, re.IGNORECASE):
                return True, f"refusal_match:{pattern[:20]}..."

        # Check 2: Loop/Garbage Pattern Matching
        for pattern in self.GARBAGE_PATTERNS:
            if re.search(pattern, combined_text, re.IGNORECASE):
                return True, f"garbage_match:{pattern[:20]}..."

        # Check 3: Web-Scraping / HTML Leaks
        if re.match(r'^\s*(?:<!DOCTYPE|<html|<\?xml|<body)', combined_text, re.IGNORECASE):
            return True, "leaked_html_document"
            
        # Check 4: JSON Validation Heuristic
        if self._expects_json(kwargs):
            if not actual_content.strip():
                return True, "safety_trip_empty_json_content"
            if '{' not in actual_content and '[' not in actual_content:
                return True, "missing_json_structure_in_content"
        else:
            # Check 5: Empty response detection
            if not combined_text.strip():
                return True, "response_is_empty"
                
        return False, ""

    def _get_deployment_id(self, kwargs) -> str:
        """Retrieves the unique, long-form identifier of the targeted model node."""
        litellm_params = kwargs.get("litellm_params") or {}
        model_info = litellm_params.get("model_info") or {}
        model_id = model_info.get("id", "")
        if model_id and len(model_id) >= 20:
            return model_id
        return ""

    def _mark_deployment_dead(self, deployment_id: str, reason: str):
        """
        Manually triggers a failover on a model node by flagging it in the 
        active cooldown cache. The node remains inactive for COOLDOWN_SECONDS.
        """
        msg = f"[UniversalGarbageHandler] Marking deployment {deployment_id[:12]}... as DEAD (reason: {reason})"
        print(msg)
        log_to_file(f"[DEPLOYMENT_DEAD] deployment={deployment_id[:16]}... reason={reason}")
        
        try:
            from litellm.proxy.proxy_server import llm_router
            if llm_router is None or not hasattr(llm_router, 'cooldown_cache'):
                return
                
            # Create a mock internal error exception to trigger LiteLLM's fallback logic
            fake_exception = litellm.InternalServerError(
                message=f"Garbage response detected: {reason}",
                model=deployment_id,
                llm_provider="",
            )
            
            # Accommodate variations across different LiteLLM package versions
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

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle Hook 2: Logging and Output Auditing (Non-Streaming)
    # ─────────────────────────────────────────────────────────────────────────
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Triggered upon any successful standard chat completion."""
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

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle Hook 3: Logging and Output Auditing (Streaming Completions)
    # ─────────────────────────────────────────────────────────────────────────
    async def async_log_stream_complete_event(self, kwargs, response_obj, start_time, end_time):
        """Triggered after an entire stream of tokens has completed."""
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

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle Hook 4: Logging Natural Failures (e.g., Timeout, 429 Rate Limit)
    # ─────────────────────────────────────────────────────────────────────────
    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Triggered if an API call fails due to standard provider errors."""
        deployment_id = self._get_deployment_id(kwargs)
        model_name = kwargs.get("model", "unknown")
        
        exception = kwargs.get("exception") or kwargs.get("original_exception") or response_obj
        exception_type = type(exception).__name__ if exception else "Unknown"
        exception_msg = str(exception)[:200] if exception else "no details"
        
        log_to_file(f"[PROVIDER_FAILURE] model={model_name} deploy={deployment_id[:12]} error={exception_type}: {exception_msg!r}")

# Instantiate class to automatically register callback within LiteLLM
custom_handler = UniversalGarbageHandler()

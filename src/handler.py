import litellm
from litellm.integrations.custom_logger import CustomLogger
import pathlib
import logging
import re
import time
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
for _logger_name in ("LiteLLM Router", "LiteLLM", "litellm"):
    logging.getLogger(_logger_name).addFilter(SuppressNoisyRouterErrors())
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
    1. Translate virtual models (FAST/SMART/CODE/GOON) into active endpoints before calling them.
    2. Inspect incoming LLM outputs (both standard completions and streams).
    3. Programmatically quarantine (cooldown) any deployment returning garbage/refusals.
    """

    COOLDOWN_SECONDS = 600
    EMPTY_RESPONSE_COOLDOWN_SECONDS = 120
    CONSECUTIVE_FAILURE_THRESHOLD = 3

    # Virtual entry points -> real model group. Each alias is rewritten in
    # async_pre_call_hook so the dummy config entries never actually fire.
    VIRTUAL_MODEL_MAP = {
        "FAST": "google/gemma4",
        "SMART": "nvidia/glm51",
        "CODE": "nvidia/glm51",
        "GOON": "nvidia/glm51",
    }

    # Regular expressions to identify common system alignment/moderation refusals.
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
    GARBAGE_PATTERNS = [
        r"Deferred tools? list",
        r"ToolSearch tools? not shown",
    ]

    # Pre-compiled regexes. Compiling once at class load avoids re-parsing the
    # patterns on every response check, which matters under heavy load.
    _COMPILED_REFUSAL = tuple(re.compile(p, re.IGNORECASE) for p in REFUSAL_PATTERNS)
    _COMPILED_GARBAGE = tuple(re.compile(p, re.IGNORECASE) for p in GARBAGE_PATTERNS)

    # HTML/XML document leak detector — only matches when a leaked document
    # appears at the very start of the response (not embedded code snippets).
    _HTML_LEAK_PATTERN = re.compile(r'^\s*(?:<!DOCTYPE|<html|<\?xml|<body)', re.IGNORECASE)

    def __init__(self):
        # Per-instance failure tracking. Defining these at class level would
        # silently share state across instances (and across re-imports), which
        # is not what we want.
        self._deployment_failures: dict[str, int] = {}
        self._deployment_last_failure: dict[str, float] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle Hook 1: Pre-Call Request Interception
    # ─────────────────────────────────────────────────────────────────────────
    async def async_pre_call_hook(self, user_api_key_dict: dict, data: dict, call_type: str, *args, **kwargs):
        """
        Pre-call hook to intercept virtual/entry model requests (FAST/SMART/...)
        and map them to real model groups. This avoids hitting dummy deployments
        defined in config.yaml.

        Using *args and **kwargs makes this signature fully compatible with
        varied versions of LiteLLM (which may pass 'cache' or 'cache_dict' keywords).
        """
        model = data.get("model")
        target = self.VIRTUAL_MODEL_MAP.get(model)
        if target:
            data["model"] = target
            log_to_file(f"[ROUTER_REWRITE] virtual_model={model} -> target={target}")

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers: response introspection
    # ─────────────────────────────────────────────────────────────────────────
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
        Checks if the caller explicitly expects structural JSON via response_format.
        Only checks the response_format parameter — not message content heuristics,
        which cause false positives when prompts discuss JSON without requesting it.
        """
        litellm_params = kwargs.get("litellm_params", {})
        response_format = litellm_params.get("response_format") or kwargs.get("response_format", {})
        if isinstance(response_format, dict):
            return response_format.get("type", "") in ("json_schema", "json_object")
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
        if kwargs.get("model") in ("FAST", "SMART") or actual_content.strip() == "error":
            return False, ""

        # Streaming requests (SSE) deliver content via chunk events, not in the
        # final response_obj wrapper that this hook inspects. The wrapper often
        # reports empty content even though the client received a full stream.
        # Audit those via async_log_stream_complete_event instead.
        litellm_params = kwargs.get("litellm_params", {}) or {}
        is_streaming = bool(litellm_params.get("stream") or kwargs.get("stream"))
        if is_streaming:
            return False, ""

        # Check 1: Refusal Matching
        for compiled in self._COMPILED_REFUSAL:
            if compiled.search(combined_text):
                return True, f"refusal_match:{compiled.pattern[:20]}..."

        # Check 2: Loop/Garbage Pattern Matching
        for compiled in self._COMPILED_GARBAGE:
            if compiled.search(combined_text):
                return True, f"garbage_match:{compiled.pattern[:20]}..."

        # Check 3: Web-Scraping / HTML Leaks
        if self._HTML_LEAK_PATTERN.match(combined_text):
            return True, "leaked_html_document"

        # Check 4: JSON Format Mismatch
        # When the client requests JSON but the model returns prose/empty,
        # log the mismatch but do NOT mark the deployment as dead.
        # The fallback chain will try the next model instead.
        if self._expects_json(kwargs):
            model_name = self._get_model_alias(kwargs)
            if not actual_content.strip():
                log_to_file(f"[JSON_FORMAT_MISMATCH] model={model_name} empty content but JSON expected in response_format/messages")
            elif '{' not in actual_content and '[' not in actual_content:
                log_to_file(f"[JSON_FORMAT_MISMATCH] model={model_name} no JSON structure in response (prose returned instead)")
        else:
            # Check 5: Empty response detection
            if not combined_text.strip():
                if reasoning.strip():
                    return False, ""
                return True, "response_is_empty"

        return False, ""

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers: deployment identification
    # ─────────────────────────────────────────────────────────────────────────
    def _get_model_alias(self, kwargs) -> str:
        """Returns the user-facing model alias (e.g. 'nvidia/glm51') instead of the provider model."""
        return (
            kwargs.get("litellm_params", {})
            .get("metadata", {})
            .get("model_group")
            or kwargs.get("model", "unknown")
        )

    def _get_deployment_id(self, kwargs) -> str:
        """Retrieves the unique, long-form identifier of the targeted model node."""
        litellm_params = kwargs.get("litellm_params") or {}
        model_info = litellm_params.get("model_info") or {}
        model_id = model_info.get("id", "")
        if model_id and len(model_id) >= 20:
            return model_id
        return ""

    def _is_retry_available(self, kwargs: dict) -> bool:
        """True when the router still has a retry budget for this request."""
        if not kwargs:
            return False
        litellm_params = kwargs.get("litellm_params", {})
        metadata = litellm_params.get("metadata", {})
        attempt = metadata.get("attempt", 1)
        retry_count = litellm_params.get("retry_count", 0)
        return attempt <= 1 and retry_count > 0

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers: failure tracking + cooldown
    # ─────────────────────────────────────────────────────────────────────────
    def _increment_failure(self, deployment_id: str, is_failure: bool):
        """Tracks consecutive failures per deployment; resets on any success."""
        if is_failure:
            self._deployment_failures[deployment_id] = self._deployment_failures.get(deployment_id, 0) + 1
            self._deployment_last_failure[deployment_id] = time.monotonic()
        else:
            self._deployment_failures.pop(deployment_id, None)
            self._deployment_last_failure.pop(deployment_id, None)

    def _should_mark_dead(self, deployment_id: str, reason: str, kwargs: dict) -> bool:
        """
        Decide whether a failure is severe enough to put the deployment on cooldown.

        - Empty responses need CONSECUTIVE_FAILURE_THRESHOLD strikes (often transient).
        - Anything else is marked dead immediately, unless a retry is still available.
        """
        base_reason = reason.removeprefix("stream_")

        if base_reason == "response_is_empty":
            # Reset counter if the last empty-response failure is older than the
            # empty-response cooldown window — don't punish for stale history.
            last = self._deployment_last_failure.get(deployment_id, 0)
            if last and (time.monotonic() - last > self.EMPTY_RESPONSE_COOLDOWN_SECONDS):
                self._deployment_failures.pop(deployment_id, None)
                self._deployment_last_failure.pop(deployment_id, None)
            consecutive = self._deployment_failures.get(deployment_id, 0) + 1
            if consecutive < self.CONSECUTIVE_FAILURE_THRESHOLD:
                log_to_file(
                    f"[FAILURE_ACCUMULATING] deployment={deployment_id[:16]}... "
                    f"reason={reason} count={consecutive}/{self.CONSECUTIVE_FAILURE_THRESHOLD}"
                )
                return False
            return True

        if self._is_retry_available(kwargs):
            log_to_file(
                f"[RETRY_PENDING] deployment={deployment_id[:16]}... "
                f"reason={reason} - skipping cooldown, retry may succeed"
            )
            return False

        return True

    def _mark_deployment_dead(self, deployment_id: str, reason: str, kwargs: dict):
        """
        Manually triggers a failover on a model node by flagging it in the
        active cooldown cache. The node remains inactive for COOLDOWN_SECONDS
        (or a shorter duration for empty responses).
        """
        if not self._should_mark_dead(deployment_id, reason, kwargs):
            self._increment_failure(deployment_id, True)
            return

        self._increment_failure(deployment_id, True)

        base_reason = reason.removeprefix("stream_")
        cooldown = (
            self.EMPTY_RESPONSE_COOLDOWN_SECONDS
            if base_reason == "response_is_empty"
            else self.COOLDOWN_SECONDS
        )

        msg = f"[UniversalGarbageHandler] Marking deployment {deployment_id[:12]}... as DEAD (reason: {reason}, cooldown: {cooldown}s)"
        print(msg)
        log_to_file(f"[DEPLOYMENT_DEAD] deployment={deployment_id[:16]}... reason={reason} cooldown={cooldown}s")

        try:
            from litellm.proxy.proxy_server import llm_router
            if llm_router is None or not hasattr(llm_router, 'cooldown_cache'):
                return

            fake_exception = litellm.InternalServerError(
                message=f"Garbage response detected: {reason}",
                model=deployment_id,
                llm_provider="",
            )

            try:
                llm_router.cooldown_cache.add_deployment_to_cooldown(
                    model_id=deployment_id,
                    original_exception=fake_exception,
                    exception_status=500,
                    cooldown_time=float(cooldown),
                )
            except TypeError:
                # Older LiteLLM versions use `deployment_id=` instead of `model_id=`.
                llm_router.cooldown_cache.add_deployment_to_cooldown(
                    deployment_id=deployment_id,
                    original_exception=fake_exception,
                    exception_status=500,
                    cooldown_time=float(cooldown),
                )
        except Exception as e:
            log_to_file(f"[ERROR] Failed to mark deployment dead: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle Hooks 2 & 3: Logging and Output Auditing
    # (Non-Streaming + Streaming Completions share the same audit pipeline)
    # ─────────────────────────────────────────────────────────────────────────
    async def _audit_response(self, kwargs, response_obj, *, stream: bool):
        """Shared audit pipeline used by both completion and streaming hooks."""
        deployment_id = self._get_deployment_id(kwargs)
        model_name = self._get_model_alias(kwargs)
        preview = self._get_response_preview(response_obj)

        tag = "STREAM_COMPLETE" if stream else "RESPONSE"
        log_to_file(f"[{tag}] model={model_name} deploy={deployment_id[:12]} preview={preview!r}")

        if not deployment_id:
            return

        is_garbage, reason = self._looks_like_garbage(response_obj, kwargs)
        if is_garbage:
            log_to_file(f"[GARBAGE_DETECTED] model={model_name} reason={reason}")
            prefixed_reason = f"stream_{reason}" if stream else reason
            self._mark_deployment_dead(deployment_id, prefixed_reason, kwargs)
        else:
            self._increment_failure(deployment_id, False)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Triggered upon any successful standard chat completion."""
        await self._audit_response(kwargs, response_obj, stream=False)

    async def async_log_stream_complete_event(self, kwargs, response_obj, start_time, end_time):
        """Triggered after an entire stream of tokens has completed."""
        await self._audit_response(kwargs, response_obj, stream=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle Hook 4: Logging Natural Failures (e.g., Timeout, 429 Rate Limit)
    # ─────────────────────────────────────────────────────────────────────────
    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Triggered if an API call fails due to standard provider errors."""
        deployment_id = self._get_deployment_id(kwargs)
        model_name = self._get_model_alias(kwargs)

        exception = kwargs.get("exception") or kwargs.get("original_exception") or response_obj
        exception_type = type(exception).__name__ if exception else "Unknown"
        exception_msg = str(exception)[:200] if exception else "no details"

        log_to_file(
            f"[PROVIDER_FAILURE] model={model_name} deploy={deployment_id[:12]} "
            f"error={exception_type}: {exception_msg!r}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle Hook 5: Post-Call Guard (runs BEFORE response reaches client)
    # ─────────────────────────────────────────────────────────────────────────
    async def async_post_call_success_deployment_hook(
        self, data, response_obj, call_type
    ):
        """
        Inspects the response after the upstream call completes but BEFORE it
        is returned to the client. If the response contains garbage fragments
        that the client will misinterpret as tool calls (e.g. "Deferred tools
        list", "ToolSearch"), mark the deployment dead and raise so the proxy's
        retry/fallback chain kicks in — the client never sees the garbage.
        """
        if not hasattr(response_obj, "choices") or not response_obj.choices:
            return response_obj

        message = response_obj.choices[0].message
        actual_content = getattr(message, "content", None) or ""
        reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None) or ""
        combined = actual_content + " " + reasoning

        for compiled in self._COMPILED_GARBAGE:
            if compiled.search(combined):
                model_name = (
                    data.get("litellm_params", {})
                    .get("metadata", {})
                    .get("model_group")
                    or data.get("model", "unknown")
                )
                log_to_file(
                    f"[POST_CALL_GARBAGE_BLOCKED] model={model_name} "
                    f"pattern={compiled.pattern!r}"
                )
                deployment_id = self._get_deployment_id(data)
                if deployment_id:
                    self._mark_deployment_dead(deployment_id, f"garbage:{compiled.pattern[:20]}", data)
                raise litellm.BadRequestError(
                    message=f"Garbage response blocked by proxy: {compiled.pattern[:30]}",
                    model=model_name,
                    llm_provider="",
                )

        return response_obj


# Instantiate class to automatically register callback within LiteLLM
custom_handler = UniversalGarbageHandler()

from __future__ import annotations

import atexit
import json
import logging
import logging.handlers
import pathlib
import queue
import re
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional

import litellm
import yaml
from litellm.integrations.custom_logger import CustomLogger


ROOT = pathlib.Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "config.yaml"
LOG_FILE = ROOT / "litellm.log"


def _build_logger() -> logging.Logger:
    """Write callback logs off the async request path and rotate them safely."""
    logger = logging.getLogger("openvault.handler")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    log_queue: queue.Queue[logging.LogRecord] = queue.Queue(-1)
    queue_handler = logging.handlers.QueueHandler(log_queue)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    listener = logging.handlers.QueueListener(log_queue, file_handler)
    listener.start()
    atexit.register(listener.stop)
    logger.addHandler(queue_handler)
    return logger


LOGGER = _build_logger()


class SuppressNoisyRouterErrors(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return "Error occurred while trying to do fallbacks" not in message


for _logger_name in ("LiteLLM Router", "LiteLLM", "litellm"):
    logging.getLogger(_logger_name).addFilter(SuppressNoisyRouterErrors())


class FailureCategory(str, Enum):
    AUTHENTICATION = "authentication"
    BAD_GATEWAY = "bad_gateway"
    CLIENT_ERROR = "client_error"
    CONNECTION = "connection"
    CONTENT_POLICY = "content_policy"
    EMPTY_RESPONSE = "empty_response"
    GARBAGE_RESPONSE = "garbage_response"
    INVALID_JSON = "invalid_json"
    RATE_LIMIT = "rate_limit"
    REFUSAL = "refusal"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class FailureScope(str, Enum):
    DEPLOYMENT = "deployment"
    MODEL = "model"
    REQUEST = "request"


@dataclass(frozen=True)
class FailureDecision:
    category: FailureCategory
    scope: FailureScope
    cooldown_seconds: int
    threshold: int = 1
    retry_current_request: bool = True


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    category: Optional[FailureCategory] = None
    reason: str = ""
    scope: FailureScope = FailureScope.DEPLOYMENT


class RequestSanitizer:
    DROP_PARAMS = frozenset(
        {
            "chat_template_kwargs",
            "do_sample",
            "frequency_penalty",
            "presence_penalty",
        }
    )
    INTEGER_PARAMS = ("max_tokens", "max_completion_tokens")
    REASONING_TOKEN_FLOORS = {"google/gemma4": 2000}

    def sanitize(self, data: dict[str, Any]) -> dict[str, Any]:
        removed: list[str] = []
        normalized: list[str] = []
        containers: list[tuple[str, dict[str, Any]]] = [("", data)]

        extra_body = data.get("extra_body")
        if isinstance(extra_body, dict):
            containers.append(("extra_body.", extra_body))
        elif extra_body is not None:
            data.pop("extra_body", None)
            removed.append("extra_body")

        for prefix, container in containers:
            for name in self.DROP_PARAMS:
                if name in container:
                    container.pop(name, None)
                    removed.append(prefix + name)
            for name in self.INTEGER_PARAMS:
                self._normalize_positive_int(container, name, prefix, removed, normalized)

        model = self.model_group(data)
        floor = self.REASONING_TOKEN_FLOORS.get(model)
        if floor:
            current = data.get("max_completion_tokens", data.get("max_tokens"))
            try:
                current_int = int(current) if current is not None and not isinstance(current, bool) else 0
            except (TypeError, ValueError):
                current_int = 0
            if current_int < floor:
                data["max_tokens"] = floor
                data.pop("max_completion_tokens", None)
                normalized.append(f"reasoning_token_floor={floor}")

        if removed or normalized:
            LOGGER.info(
                "event=request_sanitized model=%s removed=%s normalized=%s",
                model,
                sorted(set(removed)),
                sorted(set(normalized)),
            )
        return data

    @staticmethod
    def model_group(data: dict[str, Any]) -> str:
        params = data.get("litellm_params") or {}
        metadata = params.get("metadata") or data.get("metadata") or {}
        return str(metadata.get("model_group") or data.get("model") or "unknown")

    @staticmethod
    def _normalize_positive_int(
        container: dict[str, Any],
        name: str,
        prefix: str,
        removed: list[str],
        normalized: list[str],
    ) -> None:
        if name not in container:
            return
        value = container[name]
        clean: Optional[int] = None
        if isinstance(value, int) and not isinstance(value, bool):
            clean = value
        elif isinstance(value, float) and value.is_integer():
            clean = int(value)
        elif isinstance(value, str) and value.strip().isdecimal():
            clean = int(value.strip())

        if clean is None or clean <= 0:
            container.pop(name, None)
            removed.append(prefix + name)
        elif clean != value:
            container[name] = clean
            normalized.append(prefix + name)


class ResponseValidator:
    THOUGHT_PATTERN = re.compile(r"<thought>.*?</thought>\s*", re.IGNORECASE | re.DOTALL)
    HTML_DOCUMENT = re.compile(r"^\s*(?:<!DOCTYPE|<html|<\?xml|<body)", re.IGNORECASE)
    CJK = re.compile(
        r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
        r"\U00020000-\U0002A6DF\U0002A700-\U0002B73F]"
    )
    REFUSALS = tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"I can(?:not|'t) (?:fulfill|comply with|process) (?:this|your) request",
            r"against my (?:programming|guidelines|safety|ethical|core)",
            r"I(?:'m| am) sorry, (?:but )?I can(?:not|'t) (?:assist|help|provide)",
            r"Я не могу (?:выполнить|обработать|помочь)",
            r"Извините, но я не могу",
            r"抱歉，我无法",
            r"我不能(?:协助|提供)",
            r"违反(?:了)?(?:相关|使用)?(?:政策|规定|准则|法律|安全)",
        )
    )
    GARBAGE = tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"Deferred tools? list",
            r"ToolSearch tools? not shown",
            r"^\s*(?:error|null|undefined|none)\s*$",
        )
    )

    def validate(self, response: Any, request: dict[str, Any]) -> ValidationResult:
        message = self._first_message(response)
        if message is None:
            return ValidationResult(False, FailureCategory.EMPTY_RESPONSE, "no_choices", FailureScope.DEPLOYMENT)

        content = self._text(getattr(message, "content", None))
        reasoning = self._text(
            getattr(message, "reasoning", None)
            or getattr(message, "reasoning_content", None)
        )
        tool_calls = getattr(message, "tool_calls", None)
        combined = f"{content} {reasoning}".strip()

        if not combined and not tool_calls:
            return ValidationResult(False, FailureCategory.EMPTY_RESPONSE, "empty_response", FailureScope.DEPLOYMENT)

        for pattern in self.REFUSALS:
            if pattern.search(combined):
                return ValidationResult(False, FailureCategory.REFUSAL, f"refusal:{pattern.pattern[:32]}", FailureScope.MODEL)

        for pattern in self.GARBAGE:
            if pattern.search(combined):
                return ValidationResult(False, FailureCategory.GARBAGE_RESPONSE, f"garbage:{pattern.pattern[:32]}", FailureScope.DEPLOYMENT)

        if self.HTML_DOCUMENT.match(combined):
            return ValidationResult(False, FailureCategory.GARBAGE_RESPONSE, "html_document", FailureScope.DEPLOYMENT)

        if self._expects_json(request) and content and not self._valid_json(content):
            return ValidationResult(False, FailureCategory.INVALID_JSON, "invalid_json", FailureScope.MODEL)

        return ValidationResult(True)

    def strip_internal_reasoning(self, response: Any, model_group: str) -> None:
        if model_group != "google/gemma4":
            return
        message = self._first_message(response)
        if message is None:
            return
        content = self._text(getattr(message, "content", None))
        if "<thought" not in content.lower():
            return
        stripped = self.THOUGHT_PATTERN.sub("", content).strip()
        if stripped != content.strip():
            message.content = stripped
            LOGGER.info(
                "event=thought_stripped model=%s original_length=%d result_length=%d",
                model_group,
                len(content),
                len(stripped),
            )

    @staticmethod
    def _first_message(response: Any) -> Any:
        choices = getattr(response, "choices", None)
        if not choices:
            return None
        choice = choices[0]
        return getattr(choice, "message", None) or getattr(choice, "delta", None)

    @staticmethod
    def _text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif hasattr(item, "text"):
                    parts.append(str(item.text))
            return "".join(parts)
        return "" if value is None else str(value)

    @staticmethod
    def _expects_json(request: dict[str, Any]) -> bool:
        params = request.get("litellm_params") or {}
        response_format = request.get("response_format") or params.get("response_format") or {}
        return isinstance(response_format, dict) and response_format.get("type") in {
            "json_object",
            "json_schema",
        }

    @staticmethod
    def _valid_json(content: str) -> bool:
        candidate = content.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE)
        try:
            json.loads(candidate)
            return True
        except (TypeError, ValueError):
            return False


class ErrorClassifier:
    def classify(self, exception: Any) -> FailureDecision:
        status = self._status_code(exception)
        name = type(exception).__name__.lower() if exception is not None else ""
        message = str(exception).lower() if exception is not None else ""

        if status in (401, 403) or "authentication" in name:
            return FailureDecision(FailureCategory.AUTHENTICATION, FailureScope.DEPLOYMENT, 3600)
        if status == 429 or "ratelimit" in name or "rate limit" in message:
            return FailureDecision(FailureCategory.RATE_LIMIT, FailureScope.DEPLOYMENT, self._retry_after(exception, 60), threshold=2)
        if status in (502, 503, 504) or "badgateway" in name:
            return FailureDecision(FailureCategory.BAD_GATEWAY, FailureScope.DEPLOYMENT, 60)
        if "timeout" in name or "timed out" in message:
            return FailureDecision(FailureCategory.TIMEOUT, FailureScope.DEPLOYMENT, 45, threshold=2)
        if any(token in name or token in message for token in ("connection", "connecterror", "dns", "tls", "ssl")):
            return FailureDecision(FailureCategory.CONNECTION, FailureScope.DEPLOYMENT, 60)
        if "contentpolicy" in name or "moderation" in message:
            return FailureDecision(FailureCategory.CONTENT_POLICY, FailureScope.MODEL, 0)
        if status is not None and 400 <= status < 500:
            return FailureDecision(FailureCategory.CLIENT_ERROR, FailureScope.REQUEST, 0, retry_current_request=False)
        if status is not None and status >= 500:
            return FailureDecision(FailureCategory.SERVER_ERROR, FailureScope.DEPLOYMENT, 60)
        return FailureDecision(FailureCategory.UNKNOWN, FailureScope.DEPLOYMENT, 30, threshold=2)

    @staticmethod
    def _status_code(exception: Any) -> Optional[int]:
        for owner in (exception, getattr(exception, "response", None)):
            if owner is None:
                continue
            value = getattr(owner, "status_code", None) or getattr(owner, "status", None)
            if isinstance(value, int):
                return value
        return None

    @staticmethod
    def _retry_after(exception: Any, default: int) -> int:
        response = getattr(exception, "response", None)
        headers = getattr(response, "headers", None) or {}
        value = headers.get("retry-after") or headers.get("Retry-After")
        try:
            return max(1, min(int(value), 3600))
        except (TypeError, ValueError):
            return default


class LiteLLMCooldownAdapter:
    @staticmethod
    def router() -> Any:
        try:
            from litellm.proxy.proxy_server import llm_router

            return llm_router
        except Exception:
            return None

    def add(self, deployment_id: str, decision: FailureDecision, reason: str) -> bool:
        if not deployment_id or decision.cooldown_seconds <= 0:
            return False
        router = self.router()
        cache = getattr(router, "cooldown_cache", None)
        if cache is None:
            LOGGER.warning("event=cooldown_unavailable deployment=%s", deployment_id)
            return False

        exception = litellm.InternalServerError(
            message=f"Upstream deployment rejected: {reason}",
            model=deployment_id,
            llm_provider="openai",
        )
        arguments = {
            "original_exception": exception,
            "exception_status": 500,
            "cooldown_time": float(decision.cooldown_seconds),
        }
        try:
            try:
                cache.add_deployment_to_cooldown(model_id=deployment_id, **arguments)
            except TypeError:
                cache.add_deployment_to_cooldown(deployment_id=deployment_id, **arguments)
            return True
        except Exception:
            LOGGER.exception("event=cooldown_failed deployment=%s", deployment_id)
            return False


class CircuitBreaker:
    def __init__(self, cooldown: LiteLLMCooldownAdapter) -> None:
        self.cooldown = cooldown
        self._failures: dict[tuple[str, FailureCategory], tuple[int, float]] = {}
        self._lock = threading.Lock()

    def record_failure(self, deployment_id: str, decision: FailureDecision, reason: str) -> bool:
        if not deployment_id or decision.scope != FailureScope.DEPLOYMENT:
            return False
        key = (deployment_id, decision.category)
        now = time.monotonic()
        with self._lock:
            count, last = self._failures.get(key, (0, 0.0))
            if last and now - last > max(decision.cooldown_seconds, 60):
                count = 0
            count += 1
            self._failures[key] = (count, now)
            if count < decision.threshold:
                LOGGER.info(
                    "event=failure_accumulating deployment=%s category=%s count=%d threshold=%d",
                    deployment_id,
                    decision.category.value,
                    count,
                    decision.threshold,
                )
                return False
            self._failures.pop(key, None)

        marked = self.cooldown.add(deployment_id, decision, reason)
        LOGGER.warning(
            "event=circuit_open deployment=%s category=%s cooldown=%d marked=%s reason=%s",
            deployment_id,
            decision.category.value,
            decision.cooldown_seconds,
            marked,
            reason,
        )
        return marked

    def record_success(self, deployment_id: str) -> None:
        if not deployment_id:
            return
        with self._lock:
            for key in [key for key in self._failures if key[0] == deployment_id]:
                self._failures.pop(key, None)


class Configuration:
    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self.fallbacks: dict[str, list[str]] = {}
        self.aliases: dict[str, str] = {}
        self._load_and_validate()

    def _load_and_validate(self) -> None:
        with self.path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

        model_list = config.get("model_list") or []
        groups = {entry.get("model_name") for entry in model_list if isinstance(entry, dict)}
        ids: list[str] = []
        for entry in model_list:
            if not isinstance(entry, dict):
                continue
            model_info = entry.get("model_info") or {}
            deployment_id = model_info.get("id")
            if not deployment_id:
                raise RuntimeError(f"Deployment in group {entry.get('model_name')} has no model_info.id")
            ids.append(str(deployment_id))
        if len(ids) != len(set(ids)):
            raise RuntimeError("Duplicate model_info.id values in config.yaml")

        settings = config.get("router_settings") or {}
        aliases = settings.get("model_group_alias") or {}
        self.aliases = {
            str(alias): str(value.get("model") if isinstance(value, dict) else value)
            for alias, value in aliases.items()
        }
        fallback_items = settings.get("fallbacks") or []
        for item in fallback_items:
            if isinstance(item, dict):
                for source, targets in item.items():
                    self.fallbacks[str(source)] = [str(target) for target in (targets or [])]

        referenced = set(self.aliases.values())
        referenced.update(self.fallbacks)
        referenced.update(target for targets in self.fallbacks.values() for target in targets)
        missing = sorted(name for name in referenced if name not in groups)
        if missing:
            raise RuntimeError(f"Unknown model groups referenced by aliases/fallbacks: {missing}")
        missing_fallback_entries = sorted(name for name in groups if name not in self.fallbacks)
        if missing_fallback_entries:
            raise RuntimeError(f"Model groups missing fallback entries: {missing_fallback_entries}")

    def resolve_alias(self, model: str) -> str:
        return self.aliases.get(model, model)

    def next_model(self, model: str) -> Optional[str]:
        group = self.resolve_alias(model)
        targets = self.fallbacks.get(group) or []
        return targets[0] if targets else None


class UniversalServiceHandler(CustomLogger):
    """Request sanitation, response validation, bounded quality retry and health management."""

    RETRY_MARKER = "_openvault_quality_retry"
    RETRY_FIELDS = frozenset(
        {
            "messages",
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "stop",
            "response_format",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "seed",
            "user",
            "n",
        }
    )

    def __init__(self) -> None:
        self.sanitizer = RequestSanitizer()
        self.validator = ResponseValidator()
        self.classifier = ErrorClassifier()
        self.cooldown = LiteLLMCooldownAdapter()
        self.breaker = CircuitBreaker(self.cooldown)
        self.config = Configuration(CONFIG_FILE)
        LOGGER.info("event=handler_initialized aliases=%s", self.config.aliases)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: dict,
        cache: Any,
        data: dict[str, Any],
        call_type: str,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self.sanitizer.sanitize(data)

    async def async_pre_call_deployment_hook(
        self, kwargs: dict[str, Any], call_type: Optional[object]
    ) -> dict[str, Any]:
        return self.sanitizer.sanitize(kwargs)

    async def async_log_success_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        await self._audit(kwargs, response_obj, stream=False)

    async def async_log_stream_complete_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        await self._audit(kwargs, response_obj, stream=True)

    async def async_log_failure_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        exception = kwargs.get("exception") or kwargs.get("original_exception") or response_obj
        decision = self.classifier.classify(exception)
        deployment_id = self._deployment_id(kwargs, response_obj)
        LOGGER.warning(
            "event=provider_failure model=%s deployment=%s category=%s error=%s",
            self._model_group(kwargs),
            deployment_id,
            decision.category.value,
            self._safe_error(exception),
        )
        self.breaker.record_failure(deployment_id, decision, self._safe_error(exception))

    async def async_post_call_success_hook(
        self,
        data: dict[str, Any],
        user_api_key_dict: Any,
        response: Any,
    ) -> Any:
        model_group = self.config.resolve_alias(self._model_group(data))
        self.validator.strip_internal_reasoning(response, model_group)
        result = self.validator.validate(response, data)
        if result.valid:
            return response

        deployment_id = self._deployment_id(data, response)
        decision = self._validation_decision(result)
        self.breaker.record_failure(deployment_id, decision, result.reason)
        LOGGER.warning(
            "event=response_rejected model=%s deployment=%s category=%s scope=%s reason=%s",
            model_group,
            deployment_id,
            result.category.value if result.category else "unknown",
            result.scope.value,
            result.reason,
        )

        metadata = data.get("metadata") or {}
        if metadata.get(self.RETRY_MARKER):
            raise self._upstream_error(model_group, result.reason)

        retry_model = self.config.next_model(model_group) if result.scope == FailureScope.MODEL else model_group
        if not retry_model:
            raise self._upstream_error(model_group, result.reason)
        return await self._quality_retry(data, retry_model, result.reason)

    async def _quality_retry(self, data: dict[str, Any], model: str, reason: str) -> Any:
        router = self.cooldown.router()
        if router is None:
            raise self._upstream_error(model, reason)

        retry_data = {key: data[key] for key in self.RETRY_FIELDS if key in data}
        retry_data["model"] = model
        retry_data["stream"] = False
        metadata = dict(data.get("metadata") or {})
        metadata[self.RETRY_MARKER] = True
        metadata["quality_retry_reason"] = reason
        retry_data["metadata"] = metadata
        self.sanitizer.sanitize(retry_data)

        LOGGER.info("event=quality_retry target=%s reason=%s", model, reason)
        try:
            response = await router.acompletion(**retry_data)
        except Exception:
            LOGGER.exception("event=quality_retry_failed target=%s", model)
            raise

        self.validator.strip_internal_reasoning(response, self.config.resolve_alias(model))
        retry_result = self.validator.validate(response, retry_data)
        if not retry_result.valid:
            retry_deployment = self._deployment_id(retry_data, response)
            self.breaker.record_failure(
                retry_deployment,
                self._validation_decision(retry_result),
                retry_result.reason,
            )
            raise self._upstream_error(model, retry_result.reason)
        return response

    async def _audit(self, kwargs: dict[str, Any], response: Any, stream: bool) -> None:
        deployment_id = self._deployment_id(kwargs, response)
        model_group = self._model_group(kwargs)
        result = self.validator.validate(response, kwargs)
        if result.valid:
            self.breaker.record_success(deployment_id)
            LOGGER.info(
                "event=response_ok model=%s deployment=%s stream=%s",
                model_group,
                deployment_id,
                stream,
            )
            return
        LOGGER.warning(
            "event=response_invalid model=%s deployment=%s stream=%s category=%s reason=%s",
            model_group,
            deployment_id,
            stream,
            result.category.value if result.category else "unknown",
            result.reason,
        )
        # Non-streaming responses are handled again by the proxy post-call hook,
        # which can retry them. Counting here as well would double-count one bad
        # response and open a threshold-2 circuit immediately. Streams have no
        # post-call retry boundary, so their completed audit owns health updates.
        if stream:
            self.breaker.record_failure(deployment_id, self._validation_decision(result), result.reason)

    @staticmethod
    def _validation_decision(result: ValidationResult) -> FailureDecision:
        category = result.category or FailureCategory.UNKNOWN
        if category == FailureCategory.EMPTY_RESPONSE:
            return FailureDecision(category, result.scope, 45, threshold=2)
        if category in (FailureCategory.REFUSAL, FailureCategory.INVALID_JSON):
            return FailureDecision(category, FailureScope.MODEL, 0)
        return FailureDecision(category, result.scope, 120)

    @staticmethod
    def _model_group(data: dict[str, Any]) -> str:
        return RequestSanitizer.model_group(data)

    @staticmethod
    def _deployment_id(data: dict[str, Any], response: Any = None) -> str:
        params = data.get("litellm_params") or {}
        model_info = params.get("model_info") or data.get("model_info") or {}
        metadata = params.get("metadata") or data.get("metadata") or {}
        hidden = getattr(response, "_hidden_params", None) or {}
        for value in (
            model_info.get("id"),
            hidden.get("model_id"),
            hidden.get("deployment_id"),
            metadata.get("model_id"),
            metadata.get("deployment"),
        ):
            if value:
                return str(value)
        return ""

    @staticmethod
    def _safe_error(exception: Any) -> str:
        return re.sub(r"[\r\n]+", " ", str(exception or "unknown"))[:300]

    @staticmethod
    def _upstream_error(model: str, reason: str) -> Exception:
        return litellm.InternalServerError(
            message=f"All validated upstream attempts failed: {reason}",
            model=model,
            llm_provider="openai",
        )


custom_handler = UniversalServiceHandler()

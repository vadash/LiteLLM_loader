import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import handler  # noqa: E402


def response(content="ok", *, reasoning=None, tool_calls=None, deployment_id="proxy-01"):
    message = SimpleNamespace(
        content=content,
        reasoning=reasoning,
        reasoning_content=None,
        tool_calls=tool_calls,
    )
    result = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        _hidden_params={"model_id": deployment_id},
    )
    return result


class RequestSanitizerTests(unittest.TestCase):
    def setUp(self):
        self.sanitizer = handler.RequestSanitizer()

    def test_normalizes_and_drops_bad_parameters(self):
        data = {
            "model": "nvidia/cheap",
            "max_tokens": "128",
            "do_sample": True,
            "extra_body": {"presence_penalty": 1, "max_completion_tokens": -4},
        }
        self.sanitizer.sanitize(data)
        self.assertEqual(data["max_tokens"], 128)
        self.assertNotIn("do_sample", data)
        self.assertNotIn("presence_penalty", data["extra_body"])
        self.assertNotIn("max_completion_tokens", data["extra_body"])

    def test_reasoning_model_gets_token_floor(self):
        data = {"model": "google/gemma4", "max_tokens": 100}
        self.sanitizer.sanitize(data)
        self.assertEqual(data["max_tokens"], 2000)


class ResponseValidatorTests(unittest.TestCase):
    def setUp(self):
        self.validator = handler.ResponseValidator()

    def test_empty_response_is_rejected(self):
        result = self.validator.validate(response(""), {})
        self.assertFalse(result.valid)
        self.assertEqual(result.category, handler.FailureCategory.EMPTY_RESPONSE)

    def test_tool_call_without_text_is_valid(self):
        result = self.validator.validate(response("", tool_calls=[{"id": "call-1"}]), {})
        self.assertTrue(result.valid)

    def test_exact_error_is_rejected(self):
        result = self.validator.validate(response("error"), {})
        self.assertFalse(result.valid)
        self.assertEqual(result.category, handler.FailureCategory.GARBAGE_RESPONSE)

    def test_legitimate_chinese_is_not_rejected(self):
        result = self.validator.validate(response("这是一个完全正常的中文回答，包含足够多的汉字。"), {})
        self.assertTrue(result.valid)

    def test_chinese_refusal_is_rejected_as_model_failure(self):
        result = self.validator.validate(response("抱歉，我无法满足你的请求。"), {})
        self.assertFalse(result.valid)
        self.assertEqual(result.category, handler.FailureCategory.REFUSAL)
        self.assertEqual(result.scope, handler.FailureScope.MODEL)

    def test_json_response_is_parsed_not_guessed(self):
        request = {"response_format": {"type": "json_object"}}
        self.assertTrue(self.validator.validate(response('{"ok": true}'), request).valid)
        invalid = self.validator.validate(response("Here is {broken JSON"), request)
        self.assertFalse(invalid.valid)
        self.assertEqual(invalid.category, handler.FailureCategory.INVALID_JSON)

    def test_completed_stream_response_is_validated(self):
        request = {"stream": True, "litellm_params": {"stream": True}}
        result = self.validator.validate(response("Deferred tool list"), request)
        self.assertFalse(result.valid)

    def test_thought_blocks_are_removed_only_for_gemma(self):
        result = response("<thought>private</thought>public")
        self.validator.strip_internal_reasoning(result, "google/gemma4")
        self.assertEqual(result.choices[0].message.content, "public")


class ErrorClassifierTests(unittest.TestCase):
    def setUp(self):
        self.classifier = handler.ErrorClassifier()

    def test_gateway_switches_immediately(self):
        error = SimpleNamespace(status_code=502)
        decision = self.classifier.classify(error)
        self.assertEqual(decision.category, handler.FailureCategory.BAD_GATEWAY)
        self.assertEqual(decision.threshold, 1)

    def test_client_error_does_not_quarantine(self):
        error = SimpleNamespace(status_code=400)
        decision = self.classifier.classify(error)
        self.assertEqual(decision.scope, handler.FailureScope.REQUEST)
        self.assertEqual(decision.cooldown_seconds, 0)
        self.assertFalse(decision.retry_current_request)


class CircuitBreakerTests(unittest.TestCase):
    def test_threshold_is_respected(self):
        cooldown = SimpleNamespace(add=unittest.mock.Mock(return_value=True))
        breaker = handler.CircuitBreaker(cooldown)
        decision = handler.FailureDecision(
            handler.FailureCategory.EMPTY_RESPONSE,
            handler.FailureScope.DEPLOYMENT,
            45,
            threshold=2,
        )
        self.assertFalse(breaker.record_failure("proxy-1", decision, "empty"))
        self.assertTrue(breaker.record_failure("proxy-1", decision, "empty"))
        cooldown.add.assert_called_once()


class PostCallRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_failure_moves_to_first_fallback(self):
        service = handler.UniversalServiceHandler()
        service._quality_retry = AsyncMock(return_value=response("fallback ok"))
        request = {"model": "SMART", "messages": [{"role": "user", "content": "x"}]}
        bad = response("I'm sorry, but I can't help with that")

        result = await service.async_post_call_success_hook(request, None, bad)

        self.assertEqual(result.choices[0].message.content, "fallback ok")
        self.assertEqual(service._quality_retry.await_args.args[1], "zai/glm52")

    async def test_deployment_garbage_retries_same_group(self):
        service = handler.UniversalServiceHandler()
        service._quality_retry = AsyncMock(return_value=response("proxy retry ok"))
        request = {"model": "FAST", "messages": [{"role": "user", "content": "x"}]}

        await service.async_post_call_success_hook(request, None, response("error"))

        self.assertEqual(service._quality_retry.await_args.args[1], "nvidia/cheap")


class ConfigurationTests(unittest.TestCase):
    def test_config_has_unique_explicit_deployment_ids(self):
        with (ROOT / "src" / "config.yaml").open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        ids = [entry["model_info"]["id"] for entry in config["model_list"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(ids))

    def test_aliases_and_fallbacks_reference_real_groups(self):
        configuration = handler.Configuration(ROOT / "src" / "config.yaml")
        self.assertEqual(configuration.resolve_alias("FAST1"), "nvidia/cheap")
        self.assertEqual(configuration.next_model("SMART"), "zai/glm52")


if __name__ == "__main__":
    unittest.main()

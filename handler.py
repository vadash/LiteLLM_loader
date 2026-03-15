import litellm
from litellm.integrations.custom_logger import CustomLogger


class EmptyResponseHandler(CustomLogger):
    def _is_empty(self, response_obj):
        """Check if response has no meaningful content."""
        if not hasattr(response_obj, "choices") or not response_obj.choices:
            return True
        message = response_obj.choices[0].message
        content = getattr(message, "content", None) or ""
        reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None) or ""
        return not content.strip() and not reasoning.strip()

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Raises on empty response to trigger router-level retry/fallback."""
        if self._is_empty(response_obj):
            model = kwargs.get("model", "unknown")
            print(f"[EmptyResponseHandler] Empty response from {model}, triggering retry")
            raise Exception(f"Empty LLM response from {model}")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Async version — same check for async completion paths."""
        if self._is_empty(response_obj):
            model = kwargs.get("model", "unknown")
            print(f"[EmptyResponseHandler] Empty response from {model}, triggering retry (async)")
            raise Exception(f"Empty LLM response from {model}")

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        """Proxy-level guard: rejects empty responses before they reach the client."""
        if self._is_empty(response):
            model = data.get("model", "unknown") if isinstance(data, dict) else "unknown"
            print(f"[EmptyResponseHandler] Empty response from {model} caught at proxy level")
            raise Exception(f"Empty LLM response from {model}")
        return response


custom_handler = EmptyResponseHandler()

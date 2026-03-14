import litellm
from litellm.integrations.custom_logger import CustomLogger


class EmptyResponseHandler(CustomLogger):
    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        if hasattr(response_obj, "choices") and len(response_obj.choices) > 0:
            message = response_obj.choices[0].message
            content = getattr(message, "content", None) or ""
            reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None) or ""

            if not content.strip() and not reasoning.strip():
                model = kwargs.get("model", "unknown")
                print(f"[EmptyResponseHandler] Empty response from {model}, triggering retry")
                raise Exception(f"Empty LLM response from {model}")


custom_handler = EmptyResponseHandler()

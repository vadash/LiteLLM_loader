import litellm
from litellm.integrations.custom_logger import CustomLogger
import re


class GarbageResponseHandler(CustomLogger):
    """
    Detects garbage responses, AI refusals, and silent safety trips, 
    marking offending deployments as dead in the LiteLLM router cache.
    """

    # PATTERN REASONING:
    # These catch raw training data leakage, web scraping artifacts, and models 
    # that get stuck in "assistant" loops where they ask for a prompt you already gave.
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
        
        # Useless responses: AI waiting for input (ignoring the provided context)
        r'Please provide the (?:text|context|details) you(?: would like|\'d like)',
        r'Пожалуйста, предоставьте (?:текст|контекст)',
        r'请提供(?:相关|更多)?的(?:文本|信息|上下文)',
    ]

    # PATTERN REASONING:
    # These are specifically anchored to AI-isms ("As an AI", "guidelines", "policies").
    # We do NOT use generic phrases like "I can't do that" or "I'm sorry" because 
    # characters in RP scenarios say those things naturally.
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

    def _looks_like_garbage(self, response_obj, kwargs: dict) -> tuple[bool, str]:
        """
        Evaluates the response object for failures, refusals, and safety trips.
        Returns (True, reason) if it should be marked dead, (False, "") if healthy.
        """
        
        # 1. EXTRACT PIECES SEPARATELY
        # REASONING: We must isolate actual_content from reasoning. If we only evaluate
        # them together, a model reasoning about JSON but failing to output it will 
        # bypass our structural checks.
        if not hasattr(response_obj, "choices") or not response_obj.choices:
            return True, "no_choices_in_response"
            
        message = response_obj.choices[0].message
        actual_content = getattr(message, "content", None) or ""
        reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None) or ""
        combined_text = actual_content + " " + reasoning

        # 2. CHECK REFUSALS AND GARBAGE (using combined text)
        # REASONING: AI models might output their refusal entirely inside the <think> 
        # block ("I must decline this request..."), or in the actual content. Searching 
        # the combined string ensures we catch the refusal no matter where it landed.
        for pattern in self.GARBAGE_PATTERNS:
            if re.search(pattern, combined_text, re.IGNORECASE):
                return True, f"garbage_match:{pattern[:20]}..."
                
        for pattern in self.REFUSAL_PATTERNS:
            if re.search(pattern, combined_text, re.IGNORECASE):
                return True, f"refusal_match:{pattern[:20]}..."

        # 3. CHECK HTML LEAKS
        # REASONING: Some endpoints break and serve cloudflare/nginx error HTML pages 
        # instead of JSON. This catches them.
        if re.match(r'^\s*(?:<!DOCTYPE|<html|<\?xml|<body)', combined_text, re.IGNORECASE):
            return True, "leaked_html_document"
            
        # 4. THE SAFETY TRIP & JSON STRUCTURE CHECK (CRITICAL)
        # REASONING: If OpenVault requested JSON, we MUST verify that `actual_content` 
        # specifically (not the reasoning block) contains JSON structures. If the model 
        # triggered a safety filter after thinking, `actual_content` will be blank.
        expect_json = self._expects_json(kwargs)
        
        if expect_json:
            # If the filter wiped the output entirely:
            if not actual_content.strip():
                return True, "safety_trip_empty_json_content"
                
            # If the output exists but is just conversational filler without JSON:
            if '{' not in actual_content and '[' not in actual_content:
                return True, "missing_json_structure_in_content"
        else:
            # If we don't strictly expect JSON, just ensure the response isn't completely empty
            if len(combined_text.strip()) < self.MIN_CONTENT_LENGTH:
                return True, "too_short_or_empty"
                
        return False, ""

    def _expects_json(self, kwargs: dict) -> bool:
        """
        Determines if the client (OpenVault) requested structured JSON output.
        Checks both json_schema and json_object to handle different LiteLLM provider mappings.
        """
        litellm_params = kwargs.get("litellm_params", {})
        response_format = litellm_params.get("response_format") or kwargs.get("response_format", {})
        
        if isinstance(response_format, dict):
            req_type = response_format.get("type", "")
            return req_type in ["json_schema", "json_object"]
        return False

    def _get_deployment_id(self, kwargs) -> str:
        """
        Extracts the exact SHA256 deployment hash LiteLLM uses internally so we 
        can target the specific endpoint that failed.
        """
        litellm_params = kwargs.get("litellm_params") or {}
        model_info = litellm_params.get("model_info") or {}
        model_id = model_info.get("id", "")
        if model_id and len(model_id) >= 20:
            return model_id
        print("[GarbageResponseHandler] Warning: No valid model_info.id found, skipping cooldown")
        return ""

    def _mark_deployment_dead(self, deployment_id: str, reason: str):
        """
        Injects a synthetic 500 error directly into LiteLLM's cooldown cache.
        REASONING: Bypasses standard retry counters to instantly kill the deployment 
        for 15 minutes, forcing immediate failover to the next model in the chain.
        """
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
            
            # LiteLLM API changed argument names in recent versions. Try both to ensure compatibility.
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
        """Hook for standard (non-streaming) requests."""
        deployment_id = self._get_deployment_id(kwargs)
        if not deployment_id:
            return

        is_garbage, reason = self._looks_like_garbage(response_obj, kwargs)
        if is_garbage:
            self._mark_deployment_dead(deployment_id, reason)

    async def async_log_stream_complete_event(self, kwargs, response_obj, start_time, end_time):
        """Hook for streaming requests. response_obj contains the fully aggregated chunks."""
        deployment_id = self._get_deployment_id(kwargs)
        if not deployment_id:
            return

        is_garbage, reason = self._looks_like_garbage(response_obj, kwargs)
        if is_garbage:
            self._mark_deployment_dead(deployment_id, f"stream_{reason}")


# Register the plugin instance so LiteLLM can hook into it
custom_handler = GarbageResponseHandler()

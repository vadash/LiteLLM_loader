# Routing and callback lifecycle

- LiteLLM 1.82.3 invokes `async_post_call_success_hook` after the Router has completed its normal fallback loop. Raising from that hook rejects the client response but does not re-enter Router fallback; a response-quality retry must explicitly issue a bounded Router call.
- LiteLLM resolves `model_group_alias` for deployment selection but retains the client alias as the original model group during error fallback lookup. Every alias therefore needs its own `fallbacks` and `content_policy_fallbacks` entry mirroring its target group's chain.
- A completed streaming callback can update deployment health for future requests, but it cannot replace a stream whose bytes have already reached the client. Do not describe stream auditing as current-request failover.
- Classify model-level failures (refusal, requested JSON not produced) separately from deployment failures (gateway output, connection failure, empty response). Quarantining one proxy base for deterministic model behavior only wastes pool capacity.

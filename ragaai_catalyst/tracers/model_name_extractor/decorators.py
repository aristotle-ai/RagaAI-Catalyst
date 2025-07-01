from functools import wraps


def require_llm_span(func):
    @wraps(func)
    def wrapper(self, span, *args, **kwargs):
        if span.get("attributes", {}).get("openinference.span.kind") != "LLM":
            return ""
        return func(self, span, *args, **kwargs)
    return wrapper
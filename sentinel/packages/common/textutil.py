import re

_URL_CREDENTIALS = re.compile(r"://[^/@\s]+:[^/@\s]+@")
_API_KEY_HEADER = re.compile(r"(X-API-Key['\"]?\s*[:=]\s*['\"]?)[^'\",\s]+", re.IGNORECASE)


def redact_secrets(text: str) -> str:
    """Strip URL credentials and API-key-like values from a message."""
    text = _URL_CREDENTIALS.sub("://***:***@", str(text))
    return _API_KEY_HEADER.sub(r"\1***", text)

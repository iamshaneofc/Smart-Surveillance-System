from uuid import uuid4


def new_id(prefix: str = "") -> str:
    raw = uuid4().hex[:16]
    return f"{prefix}_{raw}" if prefix else raw

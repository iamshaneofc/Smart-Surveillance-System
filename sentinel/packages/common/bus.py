import json
import threading
from collections import defaultdict, deque
from typing import Callable, Protocol

from packages.common.logging import get_logger
from packages.common.timeutil import utcnow

log = get_logger(__name__)

Handler = Callable[[dict], None]


class MessageBus(Protocol):
    def publish(self, topic: str, payload: dict) -> None: ...

    def subscribe(self, topic: str, handler: Handler) -> None: ...

    def unsubscribe(self, topic: str, handler: Handler) -> None: ...

    def recent(self, topic: str, limit: int = 50) -> list[dict]: ...

    def close(self) -> None: ...


class InMemoryBus:
    def __init__(self, history_size: int = 1000) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._history: dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self._lock = threading.Lock()

    def publish(self, topic: str, payload: dict) -> None:
        envelope = {"topic": topic, "ts": utcnow().isoformat(), "payload": payload}
        with self._lock:
            self._history[topic].append(envelope)
            handlers = list(self._handlers.get(topic, []))
        for handler in handlers:
            try:
                handler(payload)
            except Exception:
                log.exception("bus handler failed", extra={"topic": topic})

    def subscribe(self, topic: str, handler: Handler) -> None:
        with self._lock:
            self._handlers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        with self._lock:
            handlers = self._handlers.get(topic)
            if not handlers:
                return
            try:
                handlers.remove(handler)
            except ValueError:
                pass

    def recent(self, topic: str, limit: int = 50) -> list[dict]:
        with self._lock:
            return list(self._history[topic])[-limit:]

    def healthcheck(self) -> bool:
        return True

    def close(self) -> None:
        with self._lock:
            self._handlers.clear()


class RedisBus:
    STREAM_PREFIX = "sentinel:"

    def __init__(self, url: str) -> None:
        import redis

        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, name="redis-bus", daemon=True)
        self._thread.start()

    def publish(self, topic: str, payload: dict) -> None:
        self._client.xadd(
            f"{self.STREAM_PREFIX}{topic}",
            {"data": json.dumps({"payload": payload}, default=str)},
        )

    def subscribe(self, topic: str, handler: Handler) -> None:
        with self._lock:
            self._handlers[topic].append(handler)
        stream = f"{self.STREAM_PREFIX}{topic}"
        if not self._client.exists(stream):
            self._client.xadd(stream, {"data": "{}"})

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        with self._lock:
            handlers = self._handlers.get(topic)
            if not handlers:
                return
            try:
                handlers.remove(handler)
            except ValueError:
                pass

    def _poll(self) -> None:
        last_ids: dict[str, str] = {}
        while not self._stop.is_set():
            with self._lock:
                topics = list(self._handlers)
            if not topics:
                self._stop.wait(0.2)
                continue
            try:
                streams = {f"{self.STREAM_PREFIX}{t}": last_ids.get(t, "0-0") for t in topics}
                results = self._client.xread(streams, count=50, block=200)
                for stream_name, entries in results:
                    topic = stream_name.removeprefix(self.STREAM_PREFIX)
                    for entry_id, fields in entries:
                        last_ids[topic] = entry_id
                        try:
                            data = json.loads(fields.get("data", "{}"))
                        except json.JSONDecodeError:
                            continue
                        for handler in list(self._handlers.get(topic, [])):
                            try:
                                handler(data.get("payload", {}))
                            except Exception:
                                log.exception("bus handler failed", extra={"topic": topic})
            except Exception:
                log.exception("redis bus poll error")
                self._stop.wait(1.0)

    def recent(self, topic: str, limit: int = 50) -> list[dict]:
        return []

    def healthcheck(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:
            return False

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self._client.close()
        except Exception:
            pass


def create_bus(url: str) -> MessageBus:
    if url.startswith("redis://") or url.startswith("rediss://"):
        return RedisBus(url)
    return InMemoryBus()

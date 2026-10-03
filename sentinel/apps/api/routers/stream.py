"""Server-Sent Events stream for operator UI real-time updates.

Transport-limited clients (browser fetch) read this with the normal
`X-API-Key` header. On connect the stream replays a small history window
from the bus so a late-joining UI still sees recent activity, then forwards
new messages until the client disconnects. Heartbeat comments keep
proxies/browsers from timing out.

Topics (closed set - anything else is rejected):
- `events.created`   pipeline confirmed a new event
- `events.updated`   event status transition via the API
- `camera.health`    periodic camera health flush
- `alerts.updated`   alert delivery attempt results
"""

import asyncio
import json
import queue
import time

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from apps.api.deps import get_bus, require_permission
from packages.schemas.auth import Principal

router = APIRouter(tags=["stream"])

STREAM_TOPICS = (
    "events.created",
    "events.updated",
    "camera.health",
    "alerts.updated",
)

HEARTBEAT_SECONDS = 15
REPLAY_LIMIT = 5

# Short poll so client disconnect (task cancellation) interrupts within
# ~250ms instead of waiting out a long blocking get.
POLL_SECONDS = 0.25


def _sse(topic: str, payload: dict) -> str:
    return f"event: message\ndata: {json.dumps({'topic': topic, 'payload': payload}, default=str)}\n\n"


@router.get("/stream")
def event_stream(
    topics: str | None = Query(default=None),
    principal: Principal = Depends(require_permission("events:read")),
    bus=Depends(get_bus),
):
    if topics:
        wanted = [t.strip() for t in topics.split(",") if t.strip() in STREAM_TOPICS]
    else:
        wanted = list(STREAM_TOPICS)
    if not wanted:
        wanted = list(STREAM_TOPICS)

    outbox: queue.Queue = queue.Queue(maxsize=256)

    def make_handler(topic: str):
        def handler(payload: dict) -> None:
            try:
                outbox.put_nowait({"topic": topic, "payload": payload})
            except queue.Full:
                # slow client: drop rather than back-pressure the pipeline
                pass

        return handler

    handlers = {topic: make_handler(topic) for topic in wanted}
    for topic, handler in handlers.items():
        bus.subscribe(topic, handler)

    async def generate():
        last_heartbeat = time.monotonic()
        try:
            yield ": connected\n\n"
            for topic in wanted:
                try:
                    for envelope in bus.recent(topic, REPLAY_LIMIT):
                        yield _sse(topic, envelope.get("payload", {}))
                except Exception:
                    continue
            while True:
                try:
                    message = await asyncio.to_thread(
                        outbox.get, True, POLL_SECONDS
                    )
                except queue.Empty:
                    if time.monotonic() - last_heartbeat >= HEARTBEAT_SECONDS:
                        last_heartbeat = time.monotonic()
                        yield ": heartbeat\n\n"
                    continue
                yield _sse(message["topic"], message["payload"])
        finally:
            for topic, handler in handlers.items():
                try:
                    bus.unsubscribe(topic, handler)
                except Exception:
                    pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )

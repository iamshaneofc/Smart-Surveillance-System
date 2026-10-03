from __future__ import annotations

import queue
import threading

from packages.common.logging import get_logger

log = get_logger(__name__)


class AlertDispatcher:
    """Bounded background queue between the video pipeline and alert delivery.

    submit() never blocks: when the queue is full the event is dropped and
    counted. The worker thread catches every exception, so alert failures
    can never crash the pipeline.
    """

    def __init__(self, router, queue_size: int = 100, bus=None) -> None:
        self._router = router
        self._bus = bus
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.dropped = 0
        self.processed = 0
        self.failed = 0

    @property
    def queue_size(self) -> int:
        return self._queue.maxsize

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="alert-dispatcher", daemon=True
        )
        self._thread.start()
        log.info("alert dispatcher started", extra={"queue_size": self.queue_size})

    def submit(self, event) -> bool:
        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:
            self.dropped += 1
            log.warning(
                "alert queue full, dropping event",
                extra={"event_id": getattr(event, "event_id", None)},
            )
            return False

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                event = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                results = self._router.dispatch(event)
            except Exception:
                self.failed += 1
                log.exception(
                    "alert dispatch failed",
                    extra={"event_id": getattr(event, "event_id", None)},
                )
            else:
                if self._bus is not None:
                    try:
                        self._bus.publish(
                            "alerts.updated",
                            {
                                "event_id": getattr(event, "event_id", None),
                                "results": [
                                    {
                                        "channel": r.channel,
                                        "status": r.status,
                                        "reason": r.reason,
                                        "alert_id": r.alert_id,
                                    }
                                    for r in results
                                ],
                            },
                        )
                    except Exception:
                        log.exception("alerts.updated publish failed")
            finally:
                self._queue.task_done()
                self.processed += 1

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                log.warning("alert dispatcher did not stop within timeout")
        self._thread = None
        log.info(
            "alert dispatcher stopped",
            extra={"processed": self.processed, "dropped": self.dropped, "failed": self.failed},
        )

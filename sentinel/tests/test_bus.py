from packages.common.bus import InMemoryBus, create_bus


def test_publish_subscribe_roundtrip():
    bus = InMemoryBus()
    received = []
    bus.subscribe("events.created", received.append)
    bus.publish("events.created", {"event_id": "evt1"})
    assert received == [{"event_id": "evt1"}]
    history = bus.recent("events.created")
    assert history[0]["topic"] == "events.created"
    assert history[0]["payload"] == {"event_id": "evt1"}


def test_handler_exception_does_not_break_others():
    bus = InMemoryBus()
    good = []

    def bad(_payload):
        raise RuntimeError("boom")

    bus.subscribe("t", bad)
    bus.subscribe("t", good.append)
    bus.publish("t", {"x": 1})
    assert good == [{"x": 1}]


def test_topic_isolation():
    bus = InMemoryBus()
    a, b = [], []
    bus.subscribe("a", a.append)
    bus.subscribe("b", b.append)
    bus.publish("a", {"n": 1})
    assert a == [{"n": 1}]
    assert b == []


def test_unsubscribe_stops_delivery():
    bus = InMemoryBus()
    received = []
    bus.subscribe("t", received.append)
    bus.publish("t", {"n": 1})
    bus.unsubscribe("t", received.append)
    bus.publish("t", {"n": 2})
    assert received == [{"n": 1}]
    assert bus._handlers["t"] == []


def test_unsubscribe_unknown_topic_is_noop():
    bus = InMemoryBus()
    bus.unsubscribe("never-subscribed", lambda _p: None)


def test_create_bus_memory_and_redis_urls():
    memory = create_bus("memory://")
    assert isinstance(memory, InMemoryBus)
    redis_bus = create_bus("redis://localhost:6379/0")
    assert redis_bus.__class__.__name__ == "RedisBus"
    redis_bus.close()

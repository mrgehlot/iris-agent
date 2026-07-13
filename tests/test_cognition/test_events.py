from iris_agent.cognition import CognitiveEventType, EventBus


def test_emit_and_receive():
    bus = EventBus()
    received = []

    def handler(event):
        received.append(event)

    bus.subscribe(CognitiveEventType.OBSERVATION_CREATED, handler)
    bus.emit(CognitiveEventType.OBSERVATION_CREATED, data="test")

    assert len(received) == 1
    assert received[0].type == CognitiveEventType.OBSERVATION_CREATED
    assert received[0].data["data"] == "test"


def test_multiple_handlers():
    bus = EventBus()
    results = []

    def h1(event):
        results.append("h1")

    def h2(event):
        results.append("h2")

    bus.subscribe(CognitiveEventType.BELIEF_UPDATED, h1)
    bus.subscribe(CognitiveEventType.BELIEF_UPDATED, h2)
    bus.emit(CognitiveEventType.BELIEF_UPDATED)

    assert results == ["h1", "h2"]


def test_unsubscribe():
    bus = EventBus()
    results = []

    def handler(event):
        results.append("got")

    bus.subscribe(CognitiveEventType.BELIEF_UPDATED, handler)
    bus.unsubscribe(CognitiveEventType.BELIEF_UPDATED, handler)
    bus.emit(CognitiveEventType.BELIEF_UPDATED)

    assert len(results) == 0


def test_no_handlers():
    bus = EventBus()
    bus.emit(CognitiveEventType.GOAL_ADDED)
    assert True


def test_clear():
    bus = EventBus()
    bus.subscribe(CognitiveEventType.GOAL_ADDED, lambda e: None)
    bus.clear()
    bus.emit(CognitiveEventType.GOAL_ADDED)
    assert True

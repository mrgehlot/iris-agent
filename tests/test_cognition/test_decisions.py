from iris_agent.cognition import DecisionLog


def test_record_decision():
    log = DecisionLog()
    d = log.record(
        goal="Test goal",
        options=["A", "B"],
        chosen="A",
        reasoning="A is better",
        confidence=0.8,
    )
    assert d.id != ""
    assert d.goal == "Test goal"
    assert d.chosen == "A"
    assert d.timestamp > 0


def test_get_decision():
    log = DecisionLog()
    d = log.record("goal", ["A"], "A", "reason")
    assert log.get(d.id) is d


def test_recent():
    log = DecisionLog()
    log.record("g1", ["A"], "A", "r1")
    log.record("g2", ["B"], "B", "r2")
    recent = log.recent(1)
    assert len(recent) == 1
    assert recent[0].goal == "g2"


def test_all():
    log = DecisionLog()
    log.record("g1", ["A"], "A", "r1")
    log.record("g2", ["B"], "B", "r2")
    assert len(log.all()) == 2


def test_clear():
    log = DecisionLog()
    log.record("g1", ["A"], "A", "r1")
    log.clear()
    assert len(log.all()) == 0

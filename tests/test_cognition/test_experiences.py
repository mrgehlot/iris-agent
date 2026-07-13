from iris_agent.cognition import ExperienceStore


def test_create_experience():
    store = ExperienceStore()
    exp = store.create(
        goal="Test goal",
        context="Test context",
        action="Test action",
        expected_outcome="Expected",
        actual_outcome="Actual",
    )
    assert exp.id != ""
    assert exp.goal == "Test goal"
    assert len(exp.lessons) == 0


def test_add_lessons():
    store = ExperienceStore()
    exp = store.create("goal", "context", "action", "expected", "actual")
    store.add_lesson(exp.id, "Lesson learned")
    assert "Lesson learned" in store.get(exp.id).lessons


def test_add_mistakes():
    store = ExperienceStore()
    exp = store.create("goal", "context", "action", "expected", "actual")
    store.add_mistake(exp.id, "Mistake made")
    assert "Mistake made" in store.get(exp.id).mistakes


def test_recent():
    store = ExperienceStore()
    store.create("g1", "c1", "a1", "e1", "a1")
    store.create("g2", "c2", "a2", "e2", "a2")
    recent = store.recent(1)
    assert len(recent) == 1
    assert recent[0].goal == "g2"


def test_all():
    store = ExperienceStore()
    store.create("g1", "c1", "a1", "e1", "a1")
    store.create("g2", "c2", "a2", "e2", "a2")
    assert len(store.all()) == 2


def test_clear():
    store = ExperienceStore()
    store.create("g1", "c1", "a1", "e1", "a1")
    store.clear()
    assert len(store.all()) == 0


def test_get_nonexistent():
    store = ExperienceStore()
    assert store.get("nonexistent") is None

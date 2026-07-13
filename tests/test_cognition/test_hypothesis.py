from iris_agent.cognition import HypothesisGenerator


def test_propose_hypothesis():
    hg = HypothesisGenerator()
    h = hg.propose("The system is slow due to database queries")
    assert h.statement == "The system is slow due to database queries"
    assert h.status == "proposed"
    assert h.confidence == 0.3


def test_test_hypothesis_supports():
    hg = HypothesisGenerator()
    h = hg.propose("Test hypothesis")
    hg.test(h.id, "Evidence supports it", supports=True)
    assert hg.get(h.id).confidence > 0.3


def test_test_hypothesis_contradicts():
    hg = HypothesisGenerator()
    h = hg.propose("Test hypothesis", predictions=["pred1"])
    hg.test(h.id, "Evidence contradicts", supports=False)
    assert hg.get(h.id).confidence < 0.3


def test_active_hypotheses():
    hg = HypothesisGenerator()
    hg.propose("Active hypothesis")
    assert len(hg.active()) == 1


def test_process_noop():
    hg = HypothesisGenerator()
    result = hg.process({"input": "test"})
    assert result["input"] == "test"

from iris_agent.cognition import BeliefSystem


def test_get_or_create():
    bs = BeliefSystem()
    belief = bs.get_or_create("The sky is blue")
    assert belief.statement == "The sky is blue"
    assert belief.confidence == 0.5


def test_get_or_create_reuses():
    bs = BeliefSystem()
    b1 = bs.get_or_create("Same statement")
    b2 = bs.get_or_create("Same statement")
    assert b1.id == b2.id


def test_update_supports():
    bs = BeliefSystem()
    bs.update("Python is fast", "Benchmark results show speed", supports=True)
    belief = bs.find("Python is fast")
    assert belief.confidence > 0.5


def test_update_contradicts():
    bs = BeliefSystem()
    bs.update("Python is fast", "Benchmark results show speed", supports=True)
    bs.update("Python is fast", "Other benchmarks show slowness", supports=False)
    belief = bs.find("Python is fast")
    assert belief.confidence < 1.0


def test_confident_threshold():
    bs = BeliefSystem()
    bs.update("True fact", "Evidence", supports=True, evidence_weight=0.5)
    confident = bs.confident(threshold=0.7)
    assert len(confident) == 1


def test_uncertain():
    bs = BeliefSystem()
    bs.update("Low confidence", "Weak evidence", supports=True, evidence_weight=0.05)
    uncertain = bs.uncertain(threshold=0.5)
    assert len(uncertain) == 0  # 0.5 + 0.05*0.5 = 0.525 >= 0.5

    bs.update("Very low", "Counter", supports=False, evidence_weight=0.5)
    very_low = bs.uncertain(threshold=0.5)
    assert len(very_low) >= 1

from iris_agent.cognition import Learning


def test_record_and_query():
    learn = Learning()
    learn.record_outcome("mental_model", "first_principles", success=True)
    learn.record_outcome("mental_model", "first_principles", success=True)
    learn.record_outcome("mental_model", "first_principles", success=False)
    rate = learn.success_rate("mental_model", "first_principles")
    assert 0.5 < rate < 0.8


def test_best_strategy():
    learn = Learning()
    learn.record_outcome("mental_model", "good_model", success=True)
    learn.record_outcome("mental_model", "good_model", success=True)
    learn.record_outcome("mental_model", "bad_model", success=False)
    best = learn.best_strategy("mental_model")
    assert best == "good_model"


def test_process():
    learn = Learning()
    context = {"selected_mental_model": "first_principles", "plan_score": 0.8}
    result = learn.process(context)
    assert result is context

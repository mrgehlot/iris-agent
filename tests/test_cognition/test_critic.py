from iris_agent.cognition import Critic


def test_critic_approves_good_plan():
    critic = Critic(min_score=0.5)
    context = {
        "plan": [
            {"action": "Step 1: Research"},
            {"action": "Step 2: Implement"},
            {"action": "Step 3: Test"},
        ]
    }
    result = critic.process(context)
    assert "plan_critique" in result
    assert result.get("plan_revision_needed") is False


def test_critic_rejects_bad_plan():
    critic = Critic(min_score=0.7)
    context = {
        "plan": [
            {"action": ""},
            {"action": ""},
        ]
    }
    result = critic.process(context)
    assert result.get("plan_revision_needed") is True


def test_critic_empty_plan():
    critic = Critic()
    context = {}
    result = critic.process(context)
    assert "plan_critique" not in result

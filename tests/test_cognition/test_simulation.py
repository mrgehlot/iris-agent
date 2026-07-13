from iris_agent.cognition import Simulation


def test_simulation_with_decision():
    sim = Simulation()
    context = {"plan_summary": "Implement this approach"}
    result = sim.process(context)
    assert "simulation_outcome" in result


def test_simulation_no_decision():
    sim = Simulation()
    result = sim.process({})
    assert "simulation_outcome" not in result

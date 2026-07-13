from iris_agent.cognition import Planner, WorldModel


def test_planner_creates_plan():
    planner = Planner()
    context = {"input": "Do something", "goal": "Do something"}
    result = planner.process(context)
    assert "plan" in result
    assert len(result["plan"]) > 0
    assert "plan_summary" in result


def test_planner_with_reasoning():
    planner = Planner()
    context = {
        "input": "Solve problem",
        "goal": "Solve problem",
        "reasoning_context": "First principles analysis",
    }
    result = planner.process(context)
    assert len(result["plan"]) > 0


def test_planner_empty_goal():
    planner = Planner()
    result = planner.process({"input": ""})
    assert "plan" in result


def test_planner_creates_tasks_in_world_model():
    wm = WorldModel()
    wm.set_goal("main goal")
    planner = Planner()
    planner.set_world_model(wm)

    context = {"input": "Do something", "goal": "main goal"}
    result = planner.process(context)

    tasks = wm.graph.find_entities(entity_type="task")
    assert len(tasks) >= 1
    for task in tasks:
        rels = wm.graph.get_relations(source_id=task.id)
        subgoals = [r for r in rels if r.type == "subgoal_of"]
        assert len(subgoals) >= 1
        assert subgoals[0].target_id == wm.current_goal_id


def test_planner_no_world_model_does_not_crash():
    planner = Planner()
    context = {"input": "Do something", "goal": "main goal"}
    result = planner.process(context)
    assert "plan" in result
    assert len(result["plan"]) > 0

from iris_agent.cognition import (
    CognitionEdge,
    CognitionNode,
    DictGraphStore,
    Entity,
    KnowledgeGraph,
    Relation,
)


# ---------------------------------------------------------------------------
# Legacy DictGraphStore tests (unchanged)
# ---------------------------------------------------------------------------


def test_add_node():
    store = DictGraphStore()
    node = CognitionNode(id="", type="test", data={"key": "value"})
    store.add_node(node)
    assert node.id != ""
    assert node.timestamp > 0


def test_get_node():
    store = DictGraphStore()
    node = CognitionNode(id="", type="test", data={"key": "value"})
    store.add_node(node)
    retrieved = store.get_node(node.id)
    assert retrieved is not None
    assert retrieved.data["key"] == "value"


def test_add_edge():
    store = DictGraphStore()
    n1 = CognitionNode(id="", type="a", data={})
    n2 = CognitionNode(id="", type="b", data={})
    store.add_node(n1)
    store.add_node(n2)
    edge = CognitionEdge(source_id=n1.id, target_id=n2.id, relation="leads_to")
    store.add_edge(edge)
    edges = store.get_edges(n1.id)
    assert len(edges) == 1
    assert edges[0].relation == "leads_to"


def test_get_nodes_by_type():
    store = DictGraphStore()
    store.add_node(CognitionNode(id="", type="foo", data={}))
    store.add_node(CognitionNode(id="", type="foo", data={}))
    store.add_node(CognitionNode(id="", type="bar", data={}))
    foos = store.get_nodes_by_type("foo")
    assert len(foos) == 2


def test_all_nodes():
    store = DictGraphStore()
    store.add_node(CognitionNode(id="", type="a", data={}))
    store.add_node(CognitionNode(id="", type="b", data={}))
    assert len(store.all_nodes()) == 2


def test_clear():
    store = DictGraphStore()
    store.add_node(CognitionNode(id="", type="a", data={}))
    store.clear()
    assert len(store.all_nodes()) == 0


# ---------------------------------------------------------------------------
# KnowledgeGraph tests
# ---------------------------------------------------------------------------


def test_kg_add_entity():
    kg = KnowledgeGraph()
    eid = kg.add_entity("goal", {"description": "test goal"})
    assert eid != ""
    entity = kg.get_entity(eid)
    assert entity is not None
    assert entity.type == "goal"
    assert entity.properties["description"] == "test goal"
    assert entity.created_at > 0
    assert entity.updated_at > 0


def test_kg_add_entity_with_id():
    kg = KnowledgeGraph()
    eid = kg.add_entity("task", {"name": "foo"}, entity_id="my-id")
    assert eid == "my-id"
    entity = kg.get_entity("my-id")
    assert entity is not None
    assert entity.type == "task"


def test_kg_update_entity():
    kg = KnowledgeGraph()
    eid = kg.add_entity("file", {"name": "foo.py", "size": 100})
    assert kg.update_entity(eid, {"size": 200}) is True
    entity = kg.get_entity(eid)
    assert entity.properties["size"] == 200
    assert entity.properties["name"] == "foo.py"


def test_kg_update_entity_missing():
    kg = KnowledgeGraph()
    assert kg.update_entity("nonexistent", {}) is False


def test_kg_delete_entity():
    kg = KnowledgeGraph()
    eid = kg.add_entity("goal", {})
    assert kg.delete_entity(eid) is True
    assert kg.get_entity(eid) is None


def test_kg_delete_entity_cascades_relations():
    kg = KnowledgeGraph()
    a = kg.add_entity("a", {})
    b = kg.add_entity("b", {})
    kg.add_relation(a, b, "relates_to")
    assert len(kg.get_relations()) == 1
    kg.delete_entity(a)
    assert len(kg.get_relations()) == 0


def test_kg_find_entities_by_type():
    kg = KnowledgeGraph()
    kg.add_entity("goal", {"desc": "g1"})
    kg.add_entity("goal", {"desc": "g2"})
    kg.add_entity("task", {"desc": "t1"})
    goals = kg.find_entities(entity_type="goal")
    assert len(goals) == 2
    tasks = kg.find_entities(entity_type="task")
    assert len(tasks) == 1


def test_kg_find_entities_by_property():
    kg = KnowledgeGraph()
    kg.add_entity("file", {"name": "a.py", "lang": "python"})
    kg.add_entity("file", {"name": "b.js", "lang": "javascript"})
    result = kg.find_entities(entity_type="file", query={"lang": "python"})
    assert len(result) == 1
    assert result[0].properties["name"] == "a.py"


def test_kg_add_relation():
    kg = KnowledgeGraph()
    a = kg.add_entity("goal", {"desc": "main"})
    b = kg.add_entity("task", {"desc": "sub"})
    rid = kg.add_relation(a, b, "subgoal_of")
    assert rid != ""
    rel = kg.get_relation(rid)
    assert rel is not None
    assert rel.type == "subgoal_of"
    assert rel.source_id == a
    assert rel.target_id == b


def test_kg_get_relations_filtered():
    kg = KnowledgeGraph()
    a = kg.add_entity("a", {})
    b = kg.add_entity("b", {})
    c = kg.add_entity("c", {})
    kg.add_relation(a, b, "relates_to")
    kg.add_relation(a, c, "blocks")
    assert len(kg.get_relations(source_id=a)) == 2
    assert len(kg.get_relations(source_id=a, rel_type="blocks")) == 1


def test_kg_get_related_outgoing():
    kg = KnowledgeGraph()
    a = kg.add_entity("a", {})
    b = kg.add_entity("b", {})
    c = kg.add_entity("c", {})
    kg.add_relation(a, b, "leads_to")
    kg.add_relation(a, c, "leads_to")
    related = kg.get_related(a, "leads_to", direction="outgoing")
    assert len(related) == 2


def test_kg_get_related_outgoing_and_incoming():
    kg = KnowledgeGraph()
    a = kg.add_entity("a", {})
    b = kg.add_entity("b", {})
    kg.add_relation(a, b, "depends_on")
    outgoing = kg.get_related(a, "depends_on", direction="outgoing")
    assert len(outgoing) == 1
    assert outgoing[0].id == b
    incoming = kg.get_related(b, "depends_on", direction="incoming")
    assert len(incoming) == 1
    assert incoming[0].id == a


def test_kg_traverse():
    kg = KnowledgeGraph()
    a = kg.add_entity("a", {"name": "root"})
    b = kg.add_entity("b", {"name": "child"})
    c = kg.add_entity("c", {"name": "grandchild"})
    kg.add_relation(a, b, "contains")
    kg.add_relation(b, c, "contains")
    result = kg.traverse(a, max_depth=3)
    ids = {e.id for e in result}
    assert a in ids
    assert b in ids
    assert c in ids


def test_kg_traverse_max_depth():
    kg = KnowledgeGraph()
    a = kg.add_entity("a", {})
    b = kg.add_entity("b", {})
    c = kg.add_entity("c", {})
    kg.add_relation(a, b, "contains")
    kg.add_relation(b, c, "contains")
    result = kg.traverse(a, max_depth=1)
    assert len(result) == 2  # a + b
    assert c not in {e.id for e in result}


def test_kg_traverse_with_type_filter():
    kg = KnowledgeGraph()
    a = kg.add_entity("a", {})
    b = kg.add_entity("b", {})
    c = kg.add_entity("c", {})
    kg.add_relation(a, b, "contains")
    kg.add_relation(b, c, "blocks")
    result = kg.traverse(a, rel_types=["contains"])
    assert len(result) == 2  # a + b
    assert c not in {e.id for e in result}


def test_kg_stats():
    kg = KnowledgeGraph()
    kg.add_entity("goal", {})
    kg.add_entity("goal", {})
    kg.add_entity("task", {})
    a = kg.add_entity("a", {})
    b = kg.add_entity("b", {})
    kg.add_relation(a, b, "depends_on")
    stats = kg.stats()
    assert stats["total_entities"] == 5
    assert stats["total_relations"] == 1
    assert stats["entity_types"]["goal"] == 2
    assert stats["entity_types"]["task"] == 1
    assert stats["relation_types"]["depends_on"] == 1


def test_kg_clear():
    kg = KnowledgeGraph()
    kg.add_entity("goal", {})
    kg.add_entity("task", {})
    kg.clear()
    assert kg.stats()["total_entities"] == 0
    assert kg.stats()["total_relations"] == 0


def test_kg_serialize_roundtrip():
    kg = KnowledgeGraph()
    a = kg.add_entity("goal", {"desc": "test"})
    b = kg.add_entity("task", {"desc": "sub"})
    kg.add_relation(a, b, "subgoal_of")
    data = kg.to_dict()
    assert len(data["entities"]) == 2
    assert len(data["relations"]) == 1

    kg2 = KnowledgeGraph.from_dict(data)
    assert len(kg2.find_entities()) == 2
    assert len(kg2.get_relations()) == 1
    assert kg2.get_entity(a).properties["desc"] == "test"


def test_entity_dataclass():
    e = Entity(id="e1", type="goal", properties={"desc": "x"})
    assert e.id == "e1"
    assert e.type == "goal"
    assert e.properties["desc"] == "x"


def test_relation_dataclass():
    r = Relation(id="r1", source_id="a", target_id="b", type="depends_on")
    assert r.id == "r1"
    assert r.source_id == "a"
    assert r.target_id == "b"
    assert r.type == "depends_on"

from neo.datastructures.knowledge import KnowledgeGraph

def test_add_and_query_nodes():
    kg = KnowledgeGraph()
    n1 = kg.add_node('User', {'name': 'alice'})
    n2 = kg.add_node('User', {'name': 'bob'})
    n3 = kg.add_node('Action', {'verb': 'open'})
    kg.add_edge(n1.id, n3.id, 'performsAction')
    users = kg.find_by_type('User')
    assert len(users) == 2
    alice = kg.find_by_property('name', 'alice')[0]
    assert alice.id == n1.id
    # traverse
    neighbors = list(kg.neighbors(n1.id))
    assert n3 in neighbors
    res = kg.query(type='User', property_eq=('name', 'alice'))
    assert len(res) == 1 and res[0].id == n1.id

def test_remove_node_cascades_edges():
    kg = KnowledgeGraph()
    a = kg.add_node('User', {'name': 'a'})
    b = kg.add_node('User', {'name': 'b'})
    kg.add_edge(a.id, b.id, 'knows')
    assert len(kg.edges) == 1
    kg.remove_node(a.id)
    assert len(kg.edges) == 0
    assert b.id in kg.nodes

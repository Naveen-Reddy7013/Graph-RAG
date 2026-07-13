import pytest
from unittest.mock import patch
from src.database import GraphDatabaseClient
from src.orchestrator import GraphRAGOrchestrator

def setup_test_graph(db: GraphDatabaseClient):
    """Populates the database with a clean, known multi-hop character/location network."""
    db.clear_database()
    db.create_constraints()

    # 1. Add characters and locations with their local embeddings
    # Tony Stark (associated with community 1)
    db.upsert_character(
        character_id="tony_stark",
        name="Tony Stark",
        description="A genius inventor and the hero Iron Man.",
        aliases=["Iron Man", "Tony"],
        embedding=[0.95] * 384  # High positive values for vector search matches
    )

    # Pepper Potts (associated with community 1)
    db.upsert_character(
        character_id="pepper_potts",
        name="Pepper Potts",
        description="The CEO of Stark Industries and Tony's partner.",
        aliases=["Pepper"],
        embedding=[0.45] * 384
    )

    # Obadiah Stane (associated with community 2)
    db.upsert_character(
        character_id="obadiah_stane",
        name="Obadiah Stane",
        description="A ruthless business rival plotting a corporate takeover.",
        aliases=["Stane"],
        embedding=[-0.85] * 384  # Negative values to distinguish semantically
    )

    # Stark Tower Location (associated with community 1)
    db.upsert_location(
        location_id="stark_tower",
        name="Stark Tower",
        description="A high-tech skyscraper in New York City.",
        aliases=["Stark HQ"],
        embedding=[0.70] * 384
    )

    # 2. Add multi-hop relationship links
    # Link: Tony Stark -[LOCATED_IN]-> Stark Tower
    db.upsert_relationship(
        source_id="tony_stark",
        source_label="Character",
        target_id="stark_tower",
        target_label="Location",
        rel_type="LOCATED_IN",
        properties={"description": "Tony resides and runs his research lab in Stark Tower.", "chunk_ids": ["ch1"]}
    )

    # Link: Pepper Potts -[LOCATED_IN]-> Stark Tower
    db.upsert_relationship(
        source_id="pepper_potts",
        source_label="Character",
        target_id="stark_tower",
        target_label="Location",
        rel_type="LOCATED_IN",
        properties={"description": "Pepper manages corporate operations from her office in Stark Tower.", "chunk_ids": ["ch1"]}
    )

    # Link: Obadiah Stane -[INTERACTED_WITH]-> Pepper Potts
    db.upsert_relationship(
        source_id="obadiah_stane",
        source_label="Character",
        target_id="pepper_potts",
        target_label="Character",
        rel_type="INTERACTED_WITH",
        properties={"description": "Obadiah confronts and threatens Pepper Potts in the Stark Tower lobby.", "chunk_ids": ["ch2"]}
    )

    # 3. Associate nodes to communities and create community summaries
    # Community 1: Stark Tower Allies
    db.add_node_to_community("tony_stark", "Character", "1")
    db.add_node_to_community("pepper_potts", "Character", "1")
    db.add_node_to_community("stark_tower", "Location", "1")
    db.upsert_community(
        community_id="1",
        summary="This community is centered around Stark Tower, where Tony Stark and Pepper Potts live and work as allies."
    )

    # Community 2: Obadiah's Faction
    db.add_node_to_community("obadiah_stane", "Character", "2")
    db.upsert_community(
        community_id="2",
        summary="This community features Obadiah Stane, who operates in the shadows plotting corporate espionage against Stark."
    )


# A mock helper for generate_completion that returns answers or evaluation scores
def mock_generate_completion(prompt, system_prompt="", temperature=0.1, max_tokens=2000):
    if "faithfulness_score" in prompt:
        # Returns evaluation JSON for the Evaluator agent
        return '{"faithfulness_score": 0.95, "factual_gaps": []}'
    
    # Returns a long narrative answer for tracer / thinker agents
    return (
        "This is a detailed mock grounded answer that fulfills the word count constraints "
        "and outlines the connections. Tony Stark operates inside Stark Tower located in New York City. "
        "He is joined by Pepper Potts, his partner and CEO. Obadiah Stane is a business rival who acts "
        "hostilely, confronting Pepper Potts in the Stark Tower lobby. Tony Stark intervenes to protect her."
    )

@patch('src.llm.generate_completion', side_effect=mock_generate_completion)
@patch('src.llm.get_embedding', return_value=[0.5] * 384)
def test_integration_queries(mock_emb, mock_comp):
    """
    Executes 3 sample queries against the database (2 local searches, 1 global search)
    and verifies that they run successfully and follow the correct structure.
    """
    db_client = GraphDatabaseClient()
    # Force mock mode for integration tests to ensure they execute offline/locally
    db_client.mode = "mock"
    
    setup_test_graph(db_client)

    orchestrator = GraphRAGOrchestrator(db_client)

    # --- Query 1: Local Multi-hop character connection query ---
    query_1 = "How is Tony Stark connected to Pepper Potts?"
    result_1 = orchestrator.query_pipeline(query_1, mode="local", max_hops=2)

    assert result_1["query"] == query_1
    # Check that starting node Tony Stark and hop node Pepper Potts were visited
    assert "tony_stark" in result_1["graph_context"]["nodes_visited"]
    assert "pepper_potts" in result_1["graph_context"]["nodes_visited"]
    # Check that their edge links through Stark Tower are registered
    assert len(result_1["graph_context"]["edges_traversed"]) > 0
    assert len(result_1["sources"]) > 0
    assert result_1["evaluation"]["faithfulness_score"] >= 0.0

    # --- Query 2: Local multi-hop adversary query ---
    query_2 = "What did Obadiah Stane do to Pepper Potts?"
    result_2 = orchestrator.query_pipeline(query_2, mode="local", max_hops=1)

    assert result_2["query"] == query_2
    assert "obadiah_stane" in result_2["graph_context"]["nodes_visited"]
    assert "pepper_potts" in result_2["graph_context"]["nodes_visited"]
    assert "obadiah_stane-INTERACTED_WITH-pepper_potts" in result_2["graph_context"]["edges_traversed"]

    # --- Query 3: Global thematic overview query ---
    query_3 = "What are the main conflicts and groups in the story?"
    result_3 = orchestrator.query_pipeline(query_3, mode="global")

    assert result_3["query"] == query_3
    # In global mode, we query community summaries
    assert "1" in result_3["graph_context"]["communities_traversed"]
    assert "2" in result_3["graph_context"]["communities_traversed"]
    assert len(result_3["sources"]) == 2  # Matches the 2 communities summary sources
    
    db_client.close()

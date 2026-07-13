import pytest
from src.database import GraphDatabaseClient
from src.agents.map_maker import MapMakerAgent

def test_id_normalization():
    """Verifies that entity names are correctly converted to clean snake_case IDs."""
    db_client = GraphDatabaseClient()
    agent = MapMakerAgent(db_client)
    
    assert agent._normalize_id("Tony Stark") == "tony_stark"
    assert agent._normalize_id("Stark HQ!!!") == "stark_hq"
    assert agent._normalize_id("  Pepper Potts  ") == "pepper_potts"
    db_client.close()

def test_text_chunking():
    """Verifies that our text splitter chunks long text with the specified overlap."""
    db_client = GraphDatabaseClient()
    agent = MapMakerAgent(db_client)
    
    text = "abcdefghijklmnopqrstuvwxyz"
    # Chunk size 10, overlap 2
    # Expect: 
    # Chunk 1: indices 0..10 ("abcdefghij")
    # Chunk 2: starts at 10-2=8, ends at 18 ("ijklmnopqr")
    # Chunk 3: starts at 18-2=16, ends at 26 ("qrstuvwxyz")
    chunks = agent._chunk_text(text, chunk_size=10, overlap=2)
    
    assert len(chunks) == 3
    assert chunks[0] == "abcdefghij"
    assert chunks[1] == "ijklmnopqr"
    assert chunks[2] == "qrstuvwxyz"
    db_client.close()

def test_mock_database_upserts():
    """Verifies that the mock database correctly stores, appends, and merges entities."""
    db_client = GraphDatabaseClient()
    # Force mock mode for clean local test
    db_client.mode = "mock"
    db_client.clear_database()

    # Ingest a character
    db_client.upsert_character(
        character_id="tony_stark",
        name="Tony Stark",
        description="A genius inventor",
        aliases=["Iron Man"],
        embedding=[0.1] * 384
    )

    # Ingest the same character again with details to test description appending and alias merging
    db_client.upsert_character(
        character_id="tony_stark",
        name="Tony Stark",
        description="Billionaire playboy",
        aliases=["Tony"],
        embedding=[0.2] * 384
    )

    nodes = db_client.get_nodes()
    assert len(nodes) == 1
    
    char_node = nodes[0]
    assert char_node["id"] == "tony_stark"
    # Description should be concatenated
    assert "A genius inventor" in char_node["description"]
    assert "Billionaire playboy" in char_node["description"]
    # Aliases should have both unique aliases
    assert "Iron Man" in char_node["aliases"]
    assert "Tony" in char_node["aliases"]

    db_client.close()

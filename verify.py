import os
import sys
import json
import uuid
import logging
import argparse
from src.database import GraphDatabaseClient

# Set up logging for validation reporter
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] verify: %(message)s")
logger = logging.getLogger("verify")

def validate_uuid(uuid_str: str) -> bool:
    """Verifies if a string is a valid UUID format."""
    try:
        uuid.UUID(uuid_str)
        return True
    except ValueError:
        return False

def run_verification(result_path: str = "output/query_result.json", database_name: str = None) -> bool:
    """
    Validates the pipeline output file against the schema requirements
    and checks node IDs against the database.
    """
    logger.info(f"Loading query result from: {result_path}")
    
    if not os.path.exists(result_path):
        logger.error(f"Error: Output file '{result_path}' does not exist! Please run the pipeline first.")
        return False

    try:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to parse JSON file: {e}")
        return False

    success = True

    # 1. JSON Schema Key Checks
    required_keys = ["query_id", "query", "answer", "graph_context", "sources", "evaluation", "metadata"]
    for key in required_keys:
        if key not in data:
            logger.error(f"[FAIL] Missing required root schema key: '{key}'")
            success = False
        else:
            logger.info(f"[PASS] Found key '{key}'")

    if not success:
        return False

    # 2. Check query_id UUID format
    if not validate_uuid(data["query_id"]):
        logger.error(f"[FAIL] 'query_id' is not a valid UUID string: {data['query_id']}")
        success = False
    else:
        logger.info("[PASS] 'query_id' is a valid UUID format.")

    # 3. Check answer length (500 to 2000 words)
    answer_text = data.get("answer", "")
    word_count = len(answer_text.split())
    if word_count < 500 or word_count > 2000:
        logger.warning(f"[WARNING] Word count is outside target range (500-2000 words). Found {word_count} words.")
    else:
        logger.info(f"[PASS] Answer word count is within range (Found {word_count} words).")

    # 4. Check faithfulness score (0.0 to 1.0)
    eval_data = data.get("evaluation", {})
    if "faithfulness_score" not in eval_data:
        logger.error("[FAIL] Missing 'faithfulness_score' inside 'evaluation'.")
        success = False
    else:
        score = eval_data["faithfulness_score"]
        if not isinstance(score, (int, float)) or score < 0.0 or score > 1.0:
            logger.error(f"[FAIL] 'faithfulness_score' must be a float between 0.0 and 1.0. Found: {score}")
            success = False
        else:
            logger.info(f"[PASS] 'faithfulness_score' is valid (Value: {score})")

    # 5. Check relevance score for sources (0.0 to 1.0)
    sources = data.get("sources", [])
    for idx, src in enumerate(sources):
        if "relevance_score" not in src:
            logger.error(f"[FAIL] Source index {idx} is missing 'relevance_score'.")
            success = False
        else:
            rel_score = src["relevance_score"]
            if not isinstance(rel_score, (int, float)) or rel_score < 0.0 or rel_score > 1.0:
                logger.error(f"[FAIL] Source index {idx} 'relevance_score' is invalid (Value: {rel_score}).")
                success = False

    # 6. Verify traversed node IDs with a live query to the database
    target_db_display = database_name if database_name else "default"
    logger.info(f"Connecting to database '{target_db_display}' to verify traversed nodes...")
    db_client = GraphDatabaseClient(database_name=database_name)
    try:
        db_nodes = db_client.get_nodes()
        valid_ids = {node["id"] for node in db_nodes}
        
        graph_context = data.get("graph_context", {})
        nodes_visited = graph_context.get("nodes_visited", [])
        
        logger.info(f"Database contains {len(valid_ids)} active nodes: {valid_ids}")
        logger.info(f"Output has {len(nodes_visited)} visited nodes: {nodes_visited}")
        
        missing_ids = []
        for node_id in nodes_visited:
            if node_id not in valid_ids:
                missing_ids.append(node_id)
                
        if missing_ids:
            logger.error(f"[FAIL] The following node IDs in 'nodes_visited' could not be found in the database: {missing_ids}")
            success = False
        else:
            logger.info("[PASS] All nodes visited in graph_context exist in the database!")
            
    except Exception as e:
        logger.error(f"Failed to query database for verification: {e}")
        success = False
    finally:
        db_client.close()

    if success:
        logger.info("="*50)
        logger.info("SUCCESS: All verification checks passed!")
        logger.info("="*50)
        return True
    else:
        logger.error("="*50)
        logger.error("FAILURE: Some verification checks failed.")
        logger.info("="*50)
        return False

def main():
    parser = argparse.ArgumentParser(description="Output JSON verification script")
    parser.add_argument("--database", type=str, default=None, help="Target database name to query")
    parser.add_argument("--result-path", type=str, default="output/query_result.json", help="Path to output JSON result file")
    args = parser.parse_args()
    
    success = run_verification(args.result_path, args.database)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

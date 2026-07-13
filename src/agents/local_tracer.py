import logging
import math
from typing import List, Dict, Any, Tuple
from src.database import GraphDatabaseClient
from src import llm

# Set up logging for tracer activities
logger = logging.getLogger("graph_rag.local_tracer")

class LocalCharacterTracerAgent:
    """
    Agent responsible for answering specific entity-based queries.
    It uses semantic vector search to find starting node(s),
    traverses the graph up to N hops, and synthesizes a localized path-based answer.
    """
    def __init__(self, db_client: GraphDatabaseClient):
        self.db = db_client

    def answer_query(self, query: str, max_hops: int = 2) -> Tuple[str, Dict[str, List[str]], List[Dict[str, Any]]]:
        """
        Answers a local query by finding starting nodes, traversing the graph,
        and generating a grounded response.
        
        Returns:
        - answer: The grounded narrative text (500 - 2000 words).
        - graph_context: Dictionary with communities_traversed, nodes_visited, and edges_traversed.
        - sources: List of source records used for grounding.
        """
        logger.info(f"Local Tracer processing query: '{query}' with max_hops: {max_hops}")

        # Step 1: Compute query embedding
        query_vector = llm.get_embedding(query)

        # Step 2: Find starting nodes using cosine similarity
        start_nodes = self._find_start_nodes_vector(query_vector, top_k=2, similarity_threshold=0.45)
        
        if not start_nodes:
            logger.warning("No matching starting nodes found via vector search. Falling back to default search.")
            # If vector search fails to match, let's do a substring/fallback search on all nodes
            start_nodes = self._find_start_nodes_fallback(query)
            
        if not start_nodes:
            return (
                f"I could not identify any characters or locations mentioned in your query '{query}' "
                "in our graph database. Please verify the spelling or check if the book has been ingested.",
                {"communities_traversed": [], "nodes_visited": [], "edges_traversed": []},
                []
            )

        logger.info(f"Identified starting node(s) for query: {[n['name'] for n in start_nodes]}")

        # Step 3: Traverse graph out to N hops from each starting node
        all_visited_nodes = set()
        all_traversed_edges = set()
        all_sources = []

        # Collect visited nodes and relationships by traversing from starting nodes
        for node in start_nodes:
            all_visited_nodes.add(node["id"])

            # Traverse paths from this starting node
            visited, edges, path_sources = self.db.traverse_paths(node["id"], max_hops)
            
            for v_id in visited:
                all_visited_nodes.add(v_id)
            for e_str in edges:
                all_traversed_edges.add(e_str)
            all_sources.extend(path_sources)

        logger.info(f"Traversal complete. Visited {len(all_visited_nodes)} nodes, "
                    f"traversed {len(all_traversed_edges)} edges.")

        # Step 4: Retrieve full profile details for all visited nodes to build a rich context
        nodes_details = []
        db_nodes = self.db.get_nodes()
        # Create a lookup map of node_id -> node_properties
        nodes_map = {n["id"]: n for n in db_nodes}
        
        communities_traversed = set()
        for node_id in all_visited_nodes:
            if node_id in nodes_map:
                n = nodes_map[node_id]
                nodes_details.append({
                    "id": n["id"],
                    "name": n["name"],
                    "label": n.get("label", "Character"),
                    "description": n.get("description", "")
                })
                # Check if this node belongs to a community, to document communities_traversed
                comm_id = n.get("community_id")
                if comm_id:
                    communities_traversed.add(str(comm_id))
                
                # Add node profile description to grounding sources so the Evaluator can verify it
                all_sources.append({
                    "source_id": f"node_{n['id']}",
                    "text_chunk": f"Entity Profile: {n['name']} ({n.get('label', 'Character')}) - Description: {n.get('description', '')}",
                    "relevance_score": 0.95
                })

        # Deduplicate sources by source_id now that all node profiles and relationship links are in all_sources
        unique_sources = []
        seen_source_ids = set()
        for src in all_sources:
            if src["source_id"] not in seen_source_ids:
                seen_source_ids.add(src["source_id"])
                unique_sources.append(src)

        # Step 5: Ask the LLM to write a detailed, path-grounded answer
        system_prompt = (
            "You are a helpful assistant and a Story Analyst. Your job is to answer specific queries "
            "about character connections and events in the book based on the provided story context. "
            "Your answer must sound natural, user-friendly, and story-focused, as if written for a general "
            "reader asking about the book, not for someone analyzing a database or graph. "
            "It MUST be between 500 and 2000 words long and be fully grounded in the provided context. "
            "Do not hypothesize, extrapolate, or hallucinate."
        )

        prompt = self._prepare_tracer_prompt(query, nodes_details, unique_sources)
        
        try:
            logger.info("Sending graph context paths to LLM for local synthesis...")
            answer = llm.generate_completion(prompt, system_prompt, temperature=0.0, max_tokens=3000)
            
            word_count = len(answer.split())
            logger.info(f"Generated local answer. Word count: {word_count}.")

            graph_context = {
                "communities_traversed": list(communities_traversed),
                "nodes_visited": list(all_visited_nodes),
                "edges_traversed": list(all_traversed_edges)
            }

            return answer, graph_context, unique_sources

        except Exception as e:
            logger.error(f"Error generating local answer: {e}")
            return (
                f"An error occurred while tracing character paths: {e}",
                {"communities_traversed": [], "nodes_visited": list(all_visited_nodes), "edges_traversed": list(all_traversed_edges)},
                unique_sources
            )

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """
        Calculates the cosine similarity between two numeric vectors.
        Formula: (A . B) / (||A|| * ||B||)
        """
        dot_product = sum(a * b for a, b in zip(v1, v2))
        magnitude_v1 = math.sqrt(sum(a * a for a in v1))
        magnitude_v2 = math.sqrt(sum(b * b for b in v2))
        if magnitude_v1 == 0.0 or magnitude_v2 == 0.0:
            return 0.0
        return dot_product / (magnitude_v1 * magnitude_v2)

    def _find_start_nodes_vector(self, query_vector: List[float], top_k: int, similarity_threshold: float) -> List[Dict[str, Any]]:
        """
        Performs semantic vector search on character and location nodes in the database.
        Returns a list of nodes matching the query.
        """
        # Fetch embeddings for all characters and locations
        char_embeddings = self.db.get_node_embeddings("Character")
        loc_embeddings = self.db.get_node_embeddings("Location")
        all_embeddings = char_embeddings + loc_embeddings

        scored_nodes = []
        for node in all_embeddings:
            score = self._cosine_similarity(query_vector, node["embedding"])
            if score >= similarity_threshold:
                scored_nodes.append({
                    "id": node["id"],
                    "name": node["name"],
                    "score": score
                })

        # Sort by similarity score descending
        scored_nodes.sort(key=lambda x: x["score"], reverse=True)
        
        # Resolve full node details for top matches
        db_nodes = self.db.get_nodes()
        nodes_map = {n["id"]: n for n in db_nodes}
        
        top_matches = []
        for match in scored_nodes[:top_k]:
            n_id = match["id"]
            if n_id in nodes_map:
                top_matches.append(nodes_map[n_id])

        return top_matches

    def _find_start_nodes_fallback(self, query: str) -> List[Dict[str, Any]]:
        """
        Fallback keyword/substring matching to find starting nodes in case
        vector similarity didn't exceed the threshold.
        """
        db_nodes = self.db.get_nodes()
        matches = []
        query_lower = query.lower()
        
        for node in db_nodes:
            if node.get("label") == "Community":
                continue
            name = node.get("name", "").lower()
            aliases = [a.lower() for a in node.get("aliases", [])]
            
            # If query contains the character's name or any alias
            if name in query_lower or any(alias in query_lower for alias in aliases):
                matches.append(node)
                
        return matches[:2]

    def _prepare_tracer_prompt(self, query: str, nodes: List[Dict], sources: List[Dict]) -> str:
        """
        Formats node profile details and edge path links into a clear markdown
        context description for the LLM to write a path-based response.
        """
        prompt = f"User Query: {query}\n\n"
        prompt += "Below is the context containing details about characters, locations, and their relationships:\n\n"

        prompt += "### Profiles:\n"
        for node in nodes:
            prompt += f"- {node['name']} ({node['label']}): {node['description']}\n"

        prompt += "\n### Relationship Connections:\n"
        relations = [src for src in sources if src["source_id"].startswith("chunk_") or "-" in src["source_id"]]
        if not relations:
            prompt += "No direct relationship connections were found.\n"
        else:
            for rel in relations:
                prompt += f"- {rel['text_chunk']}\n"

        prompt += "\n"
        prompt += "Task:\n"
        prompt += f"Write a comprehensive response to the query: '{query}' based on the context above.\n\n"
        prompt += "Instructions:\n"
        prompt += "- Generate answers that sound like they were written for a reader asking about the book, NOT for someone analyzing a graph/database.\n"
        prompt += "- Do NOT mention GraphRAG or graph-specific concepts in your final answer (e.g. do not mention 'community', 'communities', 'neighborhoods', 'graph traversal', 'nodes', 'edges', 'knowledge graph', 'source chunks', or how the information was retrieved).\n"
        prompt += "- Do NOT use analytical headings like 'Community Analysis', 'Community Interaction', 'Topological Summary', 'Graph Analysis', 'Introduction to the Conflict', or 'The Intervention and Character Network'. If organizing with headings, use natural story-focused headers (e.g., Story Overview, Main Characters, Main Conflict, Resolution, Themes).\n"
        prompt += "- Use simple, engaging, and conversational language. Avoid academic or meta-textual phrases (e.g., do NOT write 'The narrative is woven...', 'The thematic structure...', 'The provided profiles show...', 'Based on the traversed path...', etc.). Instead, describe the story elements directly.\n"
        prompt += "- Stay fully grounded. Every statement must be supported by the provided context. Do NOT extrapolate or assume any details that are not explicitly stated in the context. If the context says 'Maya convinced Victor not to destroy the Heartstone', do NOT add that Ashvale depends on it for survival, that cracks spread through the mountain, or that Victor wanted to possess it, unless those specific details are explicitly written in the context. Describe ONLY the literal facts and actions exactly as they are written. Do NOT write meta-commentary, notes about what details are missing, or references to 'the context' or 'the sources' (e.g., do NOT write 'the context does not mention...'). Simply omit any details not present in the context, and write the story synopsis directly without commenting on what is missing.\n"
        prompt += "- Keep important details: preserve character names, location names, major events, relationships, the story's ending, and themes.\n"
        prompt += "- Ensure the answer is comprehensive, easy to read, and between 500 and 2000 words long.\n"
        
        return prompt

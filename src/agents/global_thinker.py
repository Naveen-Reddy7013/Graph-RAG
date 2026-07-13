import logging
from typing import List, Dict, Any, Tuple
from src.database import GraphDatabaseClient
from src import llm

# Set up logging for tracking query routing
logger = logging.getLogger("graph_rag.global_thinker")

class GlobalThinkerAgent:
    """
    Agent responsible for answering book-wide, holistic questions.
    It aggregates pre-computed community summaries from the database
    to build a synthesized, global answer.
    """
    def __init__(self, db_client: GraphDatabaseClient):
        self.db = db_client

    def answer_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Answers a global query using community summaries.
        
        Input/Output:
        - state: The shared AgentState dictionary.
        """
        query = state["query"]
        chat_history = state.get("chat_history", [])
        
        logger.info(f"Global Thinker processing query: '{query}'")

        # Step 1: Fetch all community summaries from the database
        community_summaries = self.db.get_community_summaries()

        if not community_summaries:
            logger.warning("No community summaries found in the database. Returning default response.")
            state["answer"] = (
                "I couldn't find any community summaries in the database to synthesize a global answer. "
                "Please run ingestion and clustering first."
            )
            state["graph_context"] = {"communities_traversed": [], "nodes_visited": [], "edges_traversed": []}
            state["sources"] = []
            return state

        logger.info(f"Retrieved {len(community_summaries)} community summaries from the database.")

        # Step 2: Build the sources list and gather graph context
        sources = []
        communities_traversed = []
        nodes_visited = set()
        edges_traversed = set()

        for comm_id, summary in community_summaries.items():
            communities_traversed.append(comm_id)
            
            # Map this community summary as a source chunk
            sources.append({
                "source_id": f"community_{comm_id}",
                "text_chunk": f"Community {comm_id} Summary: {summary}",
                "relevance_score": 0.85 # High relevance for general synthesis
            })

            # Retrieve nodes and edges inside this community to include in graph_context
            comm_nodes = self.db.get_nodes_in_community(comm_id)
            for node in comm_nodes:
                nodes_visited.add(node["id"])
                
            comm_edges = self.db.get_edges_in_community(comm_id)
            for edge in comm_edges:
                edges_traversed.add(f"{edge['source']}-{edge['type']}-{edge['target']}")

        # Step 3: Call LLM to synthesize the final answer
        system_prompt = (
            "You are a helpful assistant and a Book Summarizer. Your job is to answer the user's query "
            "about the book by synthesizing information from the provided text segments. "
            "Your answer must sound natural, user-friendly, and story-focused, as if written for a general "
            "reader asking about the book, not for someone analyzing a database or graph. "
            "It MUST be between 500 and 2000 words long and be fully grounded in the provided summaries. "
            "Do not hypothesize, extrapolate, or hallucinate."
        )

        prompt = self._prepare_synthesis_prompt(query, community_summaries, chat_history)
        
        try:
            logger.info("Sending community summaries to LLM for global synthesis...")
            answer = llm.generate_completion(prompt, system_prompt, temperature=0.0, max_tokens=3000)
            
            # Simple check to help enforce the word count limit
            word_count = len(answer.split())
            logger.info(f"Generated global answer. Word count: {word_count}.")
            
            graph_context = {
                "communities_traversed": communities_traversed,
                "nodes_visited": list(nodes_visited),
                "edges_traversed": list(edges_traversed)
            }
            
            state["answer"] = answer
            state["graph_context"] = graph_context
            state["sources"] = sources
            return state

        except Exception as e:
            logger.error(f"Error generating global answer: {e}")
            state["answer"] = f"An error occurred while synthesizing the global answer: {e}"
            state["graph_context"] = {"communities_traversed": [], "nodes_visited": [], "edges_traversed": []}
            state["sources"] = []
            return state

    def _prepare_synthesis_prompt(self, query: str, summaries: Dict[str, str], chat_history: List[Dict[str, str]] = None) -> str:
        """
        Formats all community summaries into a clear prompt for global synthesis.
        Supports conversation context injection.
        """
        prompt = ""
        if chat_history:
            prompt += "Here is the conversation history so far for dialogue continuity:\n"
            for turn in chat_history:
                prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
            prompt += "\n"
            
        prompt += f"User Query: {query}\n\n"
        prompt += "Below are the summaries of different chapters, characters, and events from the book:\n\n"

        for comm_id, summary in summaries.items():
            prompt += f"--- Source Summary Segment {comm_id} ---\n"
            prompt += f"{summary}\n\n"

        prompt += "---\n"
        prompt += "Task:\n"
        prompt += f"Write a comprehensive response to the query: '{query}' by synthesizing the summaries above.\n\n"
        prompt += "Instructions:\n"
        prompt += "- Generate answers that sound like they were written for a reader asking about the book, NOT for someone analyzing a graph or database.\n"
        prompt += "- Do NOT mention GraphRAG or graph-specific concepts in your final answer (e.g. do not mention 'community', 'communities', 'community reports', 'neighborhoods', 'graph traversal', 'nodes', 'edges', 'knowledge graph', or how the information was retrieved).\n"
        prompt += "- If the query is an overview of the story, organize the response using natural story-focused sections, such as:\n"
        prompt += "  * Story Overview\n"
        prompt += "  * Main Characters\n"
        prompt += "  * Main Conflict\n"
        prompt += "  * Resolution\n"
        prompt += "  * Themes\n"
        prompt += "- Do NOT use analytical headings like 'Community Analysis', 'Community Interaction', 'Topological Summary', 'Graph Analysis', 'Introduction to the Conflict', or 'The Intervention and Character Network'.\n"
        prompt += "- Use simple, engaging, and conversational language. Avoid academic or meta-textual phrases (e.g., do NOT write 'The narrative is woven...', 'The thematic structure...', 'The community reports indicate...', 'The provided summaries show...', etc.). Instead, describe the story elements directly.\n"
        prompt += "- Stay fully grounded. Every statement must be supported by the provided summaries. Do NOT extrapolate or assume any details that are not explicitly stated in the summaries. Describe ONLY the literal facts and actions exactly as they are written. Do NOT write meta-commentary, notes about what details are missing, or references to 'the context' or 'the sources' (e.g., do NOT write 'the context does not mention...'). Simply omit any details not present in the summaries, and write the story synopsis directly without commenting on what is missing.\n"
        prompt += "- Keep important details: preserve character names, location names, major events, relationships, the story's ending, and themes.\n"
        prompt += "- Ensure the answer is comprehensive, easy to read, and between 500 and 2000 words long.\n"
        
        return prompt

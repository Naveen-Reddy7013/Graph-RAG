import logging
import networkx as nx
from typing import List, Dict, Any
from src.database import GraphDatabaseClient
from src import llm

# Set up logging for clustering process
logger = logging.getLogger("graph_rag.cluster_grouper")

class ClusterGrouperAgent:
    """
    Agent that analyzes the graph structure, groups connected nodes into communities
    (neighborhoods) using the Louvain algorithm, writes a summary report for each community,
    and saves these summaries back into the database.
    """
    def __init__(self, db_client: GraphDatabaseClient):
        self.db = db_client

    def run_clustering(self):
        """
        Executes community detection, summaries generation, and database updates.
        """
        logger.info("Starting community clustering pipeline...")

        # Step 1: Fetch nodes and edges from the DB
        nodes = self.db.get_nodes()
        edges = self.db.get_edges()

        if not nodes:
            logger.warning("No nodes found in the database. Skipping clustering.")
            return

        # Step 2: Build a NetworkX graph in Python memory
        g = nx.Graph()
        
        # Add nodes
        for node in nodes:
            # We skip Community nodes if there are any remnants in the database
            if node.get("label") == "Community":
                continue
            g.add_node(node["id"], label=node.get("label", "Character"), name=node.get("name", ""))

        # Add edges
        for edge in edges:
            g.add_edge(edge["source"], edge["target"], type=edge.get("type", "INTERACTED_WITH"))

        logger.info(f"Built NetworkX graph with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges.")

        # Step 3: Run Louvain community detection
        # Louvain partitions the nodes into communities that maximize modularity
        # (meaning nodes are highly connected within a community, but sparsely connected between them).
        try:
            communities = nx.community.louvain_communities(g, seed=42)
            logger.info(f"Louvain algorithm detected {len(communities)} communities.")
        except Exception as e:
            logger.error(f"Error running Louvain algorithm: {e}. Falling back to default single community partition.")
            communities = [set(g.nodes)]

        # Step 4: Process each community
        for i, node_set in enumerate(communities):
            community_id = str(i + 1)
            logger.info(f"Processing community {community_id}/{len(communities)} ({len(node_set)} members)...")

            # 4.1: Associate each node in this community with a Community node in the DB
            for node_id in node_set:
                node_label = g.nodes[node_id].get("label", "Character")
                # Add link (Node)-[:BELONGS_TO]->(Community)
                self.db.add_node_to_community(node_id, node_label, community_id)

            # 4.2: Retrieve community specific nodes and relationships from the database
            # This fetches the localized subgraph of who is in this community and how they connect
            member_nodes = self.db.get_nodes_in_community(community_id)
            member_edges = self.db.get_edges_in_community(community_id)

            # 4.3: Generate a readable text summary of the community's subgraph
            summary_prompt = self._prepare_summary_prompt(community_id, member_nodes, member_edges)
            
            # Ask the LLM to write a neighborhood report
            system_prompt = (
                "You are an expert Story Analyst and Graph Summarizer. "
                "Write a concise, professional summary report for the provided neighborhood "
                "of characters and locations in a book. Describe their main identities and relationships."
            )
            
            try:
                summary = llm.generate_completion(summary_prompt, system_prompt, temperature=0.2)
                # 4.4: Write the summary back to the database on the Community node
                self.db.upsert_community(community_id, summary.strip())
                logger.info(f"Saved summary for Community {community_id}.")
            except Exception as e:
                logger.error(f"Error generating summary for Community {community_id}: {e}")

        logger.info("Community clustering and summarization complete.")

    def _prepare_summary_prompt(self, community_id: str, nodes: List[Dict], edges: List[Dict]) -> str:
        """
        Formats community nodes and edges into a clean markdown description
        for the LLM to read and summarize.
        """
        prompt = f"Summarize the story context for Community ID: {community_id}.\n\n"
        
        prompt += "### Members of this community:\n"
        for node in nodes:
            prompt += f"- {node['name']} ({node['label']}): {node['description']}\n"
            
        prompt += "\n### Relationships/Connections within this community:\n"
        if not edges:
            prompt += "No direct connections listed in this community.\n"
        else:
            for edge in edges:
                desc = edge.get("properties", {}).get("description", "interacts with")
                # Format: Node A -[type]-> Node B: description
                prompt += f"- {edge['source']} -[{edge['type']}]-> {edge['target']}: {desc}\n"

        prompt += "\n"
        prompt += "Based on the members and relationships above, write a concise report (1-3 paragraphs) that:\n"
        prompt += "1. Summarizes the key character groups, affiliations, or factions in this community.\n"
        prompt += "2. Describes the nature of their connections and interactions.\n"
        prompt += "3. Outlines the primary locations or settings relevant to them.\n"
        prompt += "Ensure the report is factually grounded ONLY on the provided context."

        return prompt

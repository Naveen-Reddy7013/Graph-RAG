import time
import uuid
import logging
from typing import Dict, Any, Tuple
from src.database import GraphDatabaseClient
from src.agents.map_maker import MapMakerAgent
from src.agents.cluster_grouper import ClusterGrouperAgent
from src.agents.global_thinker import GlobalThinkerAgent
from src.agents.local_tracer import LocalCharacterTracerAgent
from src.agents.evaluator import EvaluatorAgent

# Configure logging for pipeline orchestration
logger = logging.getLogger("graph_rag.orchestrator")

class GraphRAGOrchestrator:
    """
    The orchestrator that coordinates the lifecycle of all agents:
    Map Maker, Cluster Grouper, Global Thinker, Local Character Tracer, and Evaluator.
    It tracks timing metrics and formats final outputs matching the JSON schema.
    """
    def __init__(self, db_client: GraphDatabaseClient):
        self.db = db_client
        
        # Initialize our agents
        self.map_maker = MapMakerAgent(self.db)
        self.cluster_grouper = ClusterGrouperAgent(self.db)
        self.global_thinker = GlobalThinkerAgent(self.db)
        self.local_tracer = LocalCharacterTracerAgent(self.db)
        self.evaluator = EvaluatorAgent()
        
        # Timing trackers to print the final execution breakdown
        self.timing_breakdown = {
            "extraction_seconds": 0.0,
            "clustering_seconds": 0.0,
            "traversal_seconds": 0.0,
            "generation_seconds": 0.0
        }

    def ingest_chapters(self, text: str):
        """
        Runs the extraction (Map Maker) and community clustering (Cluster Grouper).
        We record the durations of both stages.
        """
        logger.info("--- STARTING GRAPH RAG INGESTION STAGE ---")
        
        # 1. Extraction Stage (Map Maker)
        start_extraction = time.time()
        self.map_maker.run_ingestion(text)
        self.timing_breakdown["extraction_seconds"] = time.time() - start_extraction
        
        # 2. Topological Clustering Stage (Cluster Grouper)
        start_clustering = time.time()
        self.cluster_grouper.run_clustering()
        self.timing_breakdown["clustering_seconds"] = time.time() - start_clustering
        
        logger.info("--- INGESTION STAGE COMPLETE ---")
        self.print_ingestion_breakdown()

    def query_pipeline(self, query: str, mode: str = "local", max_hops: int = 2) -> Dict[str, Any]:
        """
        Executes query routing, traverses nodes, aggregates summaries,
        evaluates answer correctness, and generates a structured output schema dictionary.
        """
        logger.info(f"--- ROUTING QUERY (Mode: {mode}) ---")
        
        query_start_time = time.time()
        
        # Step 1: Execute query based on mode (routing)
        if mode.lower() == "global":
            # Global thinker uses pre-computed community summaries.
            # Time spent here counts as generation since it synthesizes existing summaries.
            start_gen = time.time()
            answer, graph_context, sources = self.global_thinker.answer_query(query)
            
            self.timing_breakdown["traversal_seconds"] = 0.0 # No dynamic path traversal in global search
            self.timing_breakdown["generation_seconds"] = time.time() - start_gen
            
        elif mode.lower() == "local":
            # Local tracer performs embedding search and database traversal
            start_traverse = time.time()
            # Vector search and traversal happen within the tracer
            answer, graph_context, sources = self.local_tracer.answer_query(query, max_hops)
            
            self.timing_breakdown["traversal_seconds"] = time.time() - start_traverse
            # In local mode, generation is part of the agent's work, but for clean separation,
            # we will attribute the majority of time to traversal & lookup.
            self.timing_breakdown["generation_seconds"] = time.time() - start_traverse
            
        else:
            raise ValueError(f"Invalid mode '{mode}' provided. Must be 'global' or 'local'.")

        # Step 2: Evaluate the generated answer
        eval_result = self.evaluator.evaluate_answer(query, answer, sources)
        
        query_end_time = time.time()
        execution_seconds = query_end_time - query_start_time

        # Step 3: Fetch database metadata counts
        nodes = self.db.get_nodes()
        # Exclude Community nodes from the metadata graph count to represent raw extracted network
        total_nodes = len([n for n in nodes if n.get("label") != "Community"])
        total_edges = len(self.db.get_edges())

        # Step 4: Construct the final output matching the schema
        output = {
            "query_id": str(uuid.uuid4()),
            "query": query,
            "answer": answer,
            "graph_context": graph_context,
            "sources": sources,
            "evaluation": {
                "faithfulness_score": eval_result["faithfulness_score"],
                "factual_gaps": eval_result["factual_gaps"]
            },
            "metadata": {
                "total_nodes_in_graph": total_nodes,
                "total_edges_in_graph": total_edges,
                "execution_seconds": execution_seconds
            }
        }
        
        logger.info("--- QUERY PROCESSING COMPLETE ---")
        return output

    def print_ingestion_breakdown(self):
        """Prints a clean timing log breakdown for the ingestion steps."""
        print("\n" + "="*40)
        print("INGESTION TIMING BREAKDOWN:")
        print(f"Extraction & Deduplication: {self.timing_breakdown['extraction_seconds']:.2f} seconds")
        print(f"Modularity Clustering & Summaries: {self.timing_breakdown['clustering_seconds']:.2f} seconds")
        print("="*40 + "\n")

    def print_query_breakdown(self, query_time: float):
        """Prints a clean timing log breakdown for the query execution steps."""
        print("\n" + "="*40)
        print("QUERY TIMING BREAKDOWN:")
        print(f"Graph Traversal/Lookup: {self.timing_breakdown['traversal_seconds']:.2f} seconds")
        print(f"Text Generation: {self.timing_breakdown['generation_seconds']:.2f} seconds")
        print(f"Total Query Execution: {query_time:.2f} seconds")
        print("="*40 + "\n")

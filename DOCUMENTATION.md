# Technical Report: Hierarchical GraphRAG Book Summarizer Pipeline

## 1. System Architecture & Agent Data Flow

The system is designed as a multi-agent state-driven pipeline that systematically parses book chapters, structures them into a Knowledge Graph (KG) in Neo4j, detects communities, and resolves holistic or local multi-hop queries.

### Data Flow Diagram

```
                     +---------------------------+
                     |  Raw Text Book Chapters   |
                     +-------------+-------------+
                                   |
                                   v (Page-by-page text chunks)
                     +-------------+-------------+
                     |   Map Maker (Extractor)   | <--- LLM Entity Resolution
                     +-------------+-------------+      (Deduplication)
                                   |
                                   | (Cypher MATCH/MERGE Mutations)
                                   v
                     +-------------+-------------+
                     |    Local Neo4j Graph DB   | <--- [Character, Location, Chunks]
                     +------+--------------+-----+
                            |              ^
        (Pull Graph Struc.) |              | (Write Community IDs
                            v              |  and Summary Reports)
                     +------+--------------+-----+
                     |       Cluster Grouper     | <--- Louvain Community Detection
                     +---------------------------+      (NetworkX backend)
                                   |
                +------------------+------------------+
                | (Routing Query or Chatbot Loop)     |
                v                                     v
   +------------+------------+           +------------+------------+
   |      Global Thinker     |           |      Local Character     |
   |   (Synthesizes pre-     |           |       Tracer Agent      |
   |   computed summaries)   |           |    (Rewriter + Paths)   |
   +------------+------------+           +------------+------------+
                |                                     |
                +------------------+------------------+
                                   |
                                   v (Pass Ground Truth Sources)
   +------------+-------------------------------------+------------+
   |                           Evaluator Agent                        |
   +----------------------------------+-------------------------------+
                                      |
                                      v (Validate schema & verify node IDs)
                        +-------------+-------------+
                        |     Output JSON Schema     |
                        +---------------------------+
```

---

## 2. Explicit State-Passing Architecture

To align with modern agentic workflow conventions (like LangGraph), the pipeline routes a single, unified `AgentState` dictionary parameter through all agent executions. Each agent extracts its required variables, executes its logic, mutates the state, and returns it.

### The AgentState Schema
The shared state dictionary is defined and updated across the pipeline execution lifecycle:
```python
state = {
    "query_id": str,                  # Unique UUID v4 for the transaction
    "query": str,                     # The original user input question
    "mode": str,                      # Search mode: "local" or "global"
    "max_hops": int,                  # Max relationship connections to traverse (1-3)
    "answer": str,                    # Grounded text answer synthesized by the tracer/thinker
    "graph_context": {                # Subgraph context parsed during traversal
        "communities_traversed": list,
        "nodes_visited": list,
        "edges_traversed": list
    },
    "sources": list,                  # Traversed database source records used for grounding
    "evaluation": {                   # Faithfulness audit results
        "faithfulness_score": float,  # Numeric rating clamped between 0.0 and 1.0
        "factual_gaps": list          # Specific unsupported claims flagged by the LLM
    },
    "chat_history": list,             # Accumulated dialogue turns for multi-turn chatbot mode
    "metadata": {                     # Operational logging metrics
        "total_nodes_in_graph": int,
        "total_edges_in_graph": int,
        "execution_seconds": float
    }
}
```

---

## 3. Conversational Chatbot Mode & Query Rewriting

We added an interactive dialogue session option (`--chat`) to the pipeline. Because conversational queries often rely on pronouns (*"How is he connected to her?"* or *"Where do they go?"*), passing follow-up inputs directly to vector search leads to matching failures.

To resolve this, we implemented a **Two-Stage Conversational Query Routing** process:

1. **The Query Rewriting Step**:
   Before initiating a graph search, the `LocalCharacterTracerAgent` invokes a private `_rewrite_query` helper. This sends the current query and the `chat_history` list to the LLM, prompting it to resolve pronouns into standalone search queries containing canonical names:
   $$\text{"Where do they plan to go?"} \quad \xrightarrow{\text{chat\_history}} \quad \text{"Where do Tony Stark and Pepper Potts plan to go?"}$$
2. **Context Prepending**:
   The dialogue history turns are prepended to the final synthesis prompts of `local_tracer` and `global_thinker` to ensure narrative continuity.

---

## 4. Agent Definitions & Communication Contracts

1. **The Map Maker Agent** ([map_maker.py](file:///n:/My%20Work/HSBC/Graph%20RAG%20-%20Book%20Summarizer/src/agents/map_maker.py))
   * **Role**: Splits raw chapter texts into 1500-char overlapping blocks and extracts nodes and edge tuples.
   * **Write Contract**: Enforces unique database constraints and executes Cypher merge statements.
2. **The Cluster Grouper Agent** ([cluster_grouper.py](file:///n:/My%20Work/HSBC/Graph%20RAG%20-%20Book%20Summarizer/src/agents/cluster_grouper.py))
   * **Role**: Detects communities using Louvain modularity, groups nodes into communities, and saves LLM-generated community summary reports.
3. **The Local Character Tracer Agent** ([local_tracer.py](file:///n:/My%20Work/HSBC/Graph%20RAG%20-%20Book%20Summarizer/src/agents/local_tracer.py))
   * **Role**: Resolves local relationship queries. Rewrites pronouns using `chat_history`, executes vector searches to locate starting nodes, and runs Cypher path traversals.
4. **The Global Thinker Agent** ([global_thinker.py](file:///n:/My%20Work/HSBC/Graph%20RAG%20-%20Book%20Summarizer/src/agents/global_thinker.py))
   * **Role**: Resolves holistic queries. Aggregates pre-computed community summaries to generate book-wide reports.
5. **The Evaluator Agent** ([evaluator.py](file:///n:/My%20Work/HSBC/Graph%20RAG%20-%20Book%20Summarizer/src/agents/evaluator.py))
   * **Role**: Compares generated responses against the retrieved raw database records to calculate faithfulness scores.

---

## 5. Database Schema Specification

Nodes and edges are modeled in Neo4j (or NetworkX fallback) as follows:

### Node Properties
* **Character Node (`:Character`)**:
  * `id` (String, Unique Normalized Key): e.g. `tony_stark`
  * `name` (String): e.g. `Tony Stark`
  * `description` (String): Accumulated descriptions across chapters (joined by `|`).
  * `aliases` (List of Strings): Recognized synonyms (e.g. `["Iron Man", "Tony"]`).
  * `embedding` (List of 384 Floats): Vector embedding of the node name.
  * `community_id` (Integer): Assigned Louvain modularity group ID.
* **Location Node (`:Location`)**:
  * `id` (String, Unique Normalized Key): e.g. `stark_tower`
  * `name` (String): e.g. `Stark Tower`
  * `description` (String): Aggregated geographical details.
  * `embedding` (List of 384 Floats): Vector embedding of the location name.
  * `community_id` (Integer): Assigned Louvain modularity group ID.
* **Community Node (`:Community`)**:
  * `id` (String, Unique): Modularity group number (e.g. `1`).
  * `summary` (String): LLM community summary report.

### Relationship Edge Properties (`-[:INTERACTED_WITH]->`, `-[:LOCATED_IN]->`, `-[:BELONGS_TO]->`)
* `id` (String, Unique): Compound identifier (e.g. `pepper_potts-LOCATED_IN-main_laboratory_24841`).
* `description` (String): Specific text-grounded description of the connection.
* `chunk_ids` (List of Strings): Array of chunk source keys where this connection was found.

---

## 6. Output Verification & Schema Validation

The standalone validation script `verify.py` ensures query outputs adhere to strict constraints:
1. **JSON Schema Check**: Verifies all root keys are present.
2. **Format Checks**: Checks `query_id` is a valid UUID structure.
3. **Score Constraints**: Checks relevance and faithfulness scores are between `0.0` and `1.0`.
4. **Answer Word Count**: Confirms answers contain between **500 and 2000 words**.
5. **Database Grounding Check**: Cross-references every node ID in `graph_context.nodes_visited` with the active database, ensuring no hallucinated nodes are returned.

---

## 7. Robustness Enhancements

We implemented the following features to solve deployment issues:
* **Online Status Polling**: Neo4j database creations are asynchronous. The database client polls `SHOW DATABASE` to wait for a database to become online before running transactions.
* **Community Edition Fallback**: If multiple databases are not supported (e.g. Neo4j Community Edition), the client falls back to the default `"neo4j"` database instead of crashing.
* **Output Stream Reconfiguration**: Windows terminal default encoding (CP1252) crashes on Unicode elements in LLM reports. We reconfigured stdout/stderr streams to `UTF-8` at the start of `main()`.
* **API Rate-Limit Handling**: Rate-limiting 429 errors from Groq trigger backoffs and retries.

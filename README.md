# The "Big Picture" Book Summarizer (GraphRAG Pipeline)

Welcome to the **"Big Picture" Book Summarizer**! This project implements a hierarchical GraphRAG pipeline that parses unstructured book chapters, extracts an interconnected Knowledge Graph (KG) of characters and locations, partitions them into communities, and evaluates user queries using both Global Summarization and Local Search.

---

## 1. Agent Architecture & Data Flow

Our system uses 4 distinct agents designed for modular operations. Below is the data flow showing how graph data passes from the raw text to the final evaluated response:

```
                  +-------------------------+
                  |  Raw Text Book Chapters |
                  +------------+------------+
                               |
                               v (Page-by-page text chunks)
                  +------------+------------+
                  |  Map Maker (Extractor)  | <--- LLM Entity Resolution
                  +------------+------------+      (Deduplication)
                               |
                               | (Cypher MATCH/MERGE Mutations)
                               v
                  +------------+------------+
                  |   Local Neo4j Graph DB  | <--- [Character, Location, Chunks]
                  +------+-------------+----+
                         |             ^
     (Pull Graph Struc.) |             | (Write Community IDs
                         v             |  and Summary Reports)
                  +------+-------------+----+
                  |      Cluster Grouper    | <--- Louvain Community Detection
                  +-------------------------+      (NetworkX backend)
                               |
            +------------------+------------------+
            | (Routing Query)                     |
            v                                     v
+-----------+------------+            +-----------+------------+
|     Global Thinker     |            |     Local Character    |
|     (Query Router)     |            |      Tracer Agent      |
+-----------+------------+            +-----------+------------+
            |                                     |
            | (Read Community                     | (Cosine Similarity start node,
            |  summaries from DB)                 |  N-hop path traversal in DB)
            |                                     |
            v                                     v
+-----------+-------------------------------------+-----------+
|                          Evaluator Agent                    |
+---------------------------------+---------------------------+
                                  |
                                  v (Validate schema & verify node IDs)
                     +------------+------------+
                     |    Output JSON Schema   |
                     +-------------------------+
```

### Communication Protocol
- **Map Maker Agent**: Ingests raw chunks, requests entity extractions from the LLM, groups aliases (e.g. "Iron Man" to "Tony Stark"), generates and commits Cypher transactions (nodes, attributes, and relationships) to Neo4j.
- **Cluster Grouper Agent**: Retrieves the active graph nodes and relationships from Neo4j, constructs a graph in NetworkX, executes the Louvain algorithm to partition the graph, links nodes to new `Community` nodes in Neo4j, and writes a detailed summary report of each community back to the DB.
- **Global Thinker Agent**: Triggered for book-wide thematic queries (e.g., "What are the main conflicts?"). It queries the summaries of all community nodes from Neo4j, parallelly aggregates them using the LLM, and produces a synthesized narrative.
- **Local Character Tracer Agent**: Triggered for targeted entity queries (e.g., "How is character A linked to location C?"). It embeds the query using a local `sentence-transformers` model, searches for the starting nodes via cosine similarity, traverses paths in the DB up to `max_hops` (1-3), and writes a grounded connection narrative.
- **Evaluator Agent**: Evaluates the compiled response against the raw retrieved sources, calculating a faithfulness score and highlighting any factual gaps (hallucinations).

---

## 2. Entity Deduplication and Resolution

Entity resolution is vital in GraphRAG to prevent the graph from becoming fragmented (e.g. having separate disconnected nodes for "Tony Stark", "Stark", and "Iron Man").

### Our Two-Stage Deduplication Strategy
1. **Uniqueness Constraints**: We apply a `UNIQUE` constraint on node `id` fields in Neo4j. This enforces integrity at the database layer.
2. **LLM-Based Batch Resolution**: 
   - After extracting entities from the text chunks, we collect all distinct names.
   - We send the full list of extracted names to the LLM in a single batch.
   - The LLM groups names that refer to the same identity and returns a JSON mapping of `Alias -> Canonical Name` (e.g., mapping "Iron Man" to "Tony Stark").
   - In Python, we translate all original names into their canonical resolved names and generate a normalized ID (e.g. `tony_stark`).
   - We run Cypher `MERGE` queries in the database. When matching an existing node (`ON MATCH`), we concatenate new descriptions and merge alias arrays using standard Cypher scripts (avoiding external plugins like APOC for maximum compatibility):
     ```cypher
     MERGE (c:Character {id: $id})
     ON CREATE SET c.name = $name, c.description = $description, c.aliases = $aliases, c.embedding = $embedding
     ON MATCH SET c.description = c.description + " | " + $description,
                  c.aliases = REDUCE(s = c.aliases, x IN $aliases | CASE WHEN x IN s THEN s ELSE s + x END),
                  c.embedding = $embedding
     ```

---

## 3. Dual-Mode Database Fallback

For maximum ease of development, the database client (`src/database.py`) operates in two modes:
1. **Neo4j Mode**: Connects to your running local Neo4j instance at `bolt://localhost:7687`.
2. **Mock Mode (In-Memory)**: If the Neo4j database is offline or not configured, it logs a warning and automatically activates an in-memory database simulation using `NetworkX`. 
   - All queries, upserts, vector lookups, and multi-hop path traversals are fully functional in Mock mode, allowing you to run and verify the pipeline offline or without database setups!

---

## 4. Quick Start Instructions

### Prerequisites
1. **Python 3.10+** and the **`uv`** package manager.
2. **Groq API Key**: Get a free key from the [Groq Console](https://console.groq.com/).
3. **Neo4j Database** (Optional): A local instance running on port 7687.

### Configuration
Copy `.env.example` to `.env` and fill in your details:
```bash
cp .env.example .env
```
Ensure you set your `GROQ_API_KEY` and local `NEO4J_PASSWORD` in `.env`.

### Automated Commands
We have automated the pipeline steps using both a `Makefile` (cross-platform) and a native PowerShell script `run.ps1` (perfect for Windows):

**Using PowerShell (Windows):**
```powershell
powershell -File run.ps1
```
This single command will:
1. Set up a virtual environment and install packages using `uv`.
2. Run unit and integration tests.
3. Run the ingestion pipeline (Map Maker and Cluster Grouper) using the sample chapters in `data/chapters/`.
4. Run a local query: *"How is Tony Stark connected to Pepper Potts?"*
5. Execute the output verification script (`verify.py`) to confirm correctness.

**Using Make (Git Bash / Linux):**
- **Install dependencies**: `make install`
- **Run tests**: `make test`
- **Execute pipeline**: `make run`
- **Verify schema output**: `make verify`


# python run.py --mock-llm
# python verify.py
# python run.py --query "How is Tony Stark connected to Pepper Potts ?"
# python run.py --query "How is Tony Stark connected to Pepper Potts ?" --database storygraph --skip-ingest
# python run.py --query "how is maya connected to victor in this story" --database storygraph4 --skip-ingest
# python run.py --chat --database storygraph9 --skip-ingest
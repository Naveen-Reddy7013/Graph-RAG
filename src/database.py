import os
import json
import logging
import networkx as nx
from typing import List, Dict, Any, Tuple
from neo4j import GraphDatabase, exceptions
from src import config

# Configure logging so we can trace database changes step-by-step
logger = logging.getLogger("graph_rag.database")

class GraphDatabaseClient:
    """
    A unified database client that connects to a real Neo4j instance,
    or falls back to an in-memory Mock Graph Database if Neo4j is offline.
    """
    def __init__(self, database_name: str = None):
        self.mode = "mock"
        self.driver = None
        self.database_name = database_name or config.NEO4J_DATABASE
        self.mock_db_path = f"output/mock_db_{self.database_name}.json"
        
        # In-memory graph storage for the mock fallback
        # nodes: maps node_id -> {properties}
        # edges: list of tuples (source_id, target_id, type, {properties})
        self._mock_nodes: Dict[str, Dict[str, Any]] = {}
        self._mock_edges: List[Dict[str, Any]] = []

        # Load existing mock DB state if it exists from previous run processes
        self._load_mock_db()

        print(f"Connecting to Neo4j at {config.NEO4J_URI}...")
        try:
            # Attempt to initialize the official Neo4j driver
            self.driver = GraphDatabase.driver(
                config.NEO4J_URI,
                auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
            )
            # Verify the connection by running a simple query
            self.driver.verify_connectivity()
            self.mode = "neo4j"
            print(f"Connected successfully to Neo4j database (Target DB: {self.database_name})!")
        except Exception as e:
            print("\n" + "="*80)
            print("WARNING: Could not connect to local Neo4j database.")
            print(f"Details: {e}")
            print("Falling back to in-memory MOCK GRAPH DATABASE mode.")
            print(f"All graph operations will run locally and persist in: {self.mock_db_path}")
            print("="*80 + "\n")
            self.mode = "mock"

    def create_new_database(self, db_name: str):
        """Creates a new database inside Neo4j if it doesn't exist."""
        if self.mode == "neo4j":
            logger.info(f"Creating new database in Neo4j: {db_name}")
            # System commands like CREATE DATABASE must be executed on the 'system' database
            with self.driver.session(database="system") as session:
                try:
                    # Run the Cypher query to create the database if it doesn't exist
                    session.run(f"CREATE DATABASE {db_name} IF NOT EXISTS")
                    logger.info(f"Database '{db_name}' verified/created successfully. Waiting for it to become online...")
                    
                    import time
                    online = False
                    for _ in range(15):
                        res = session.run(f"SHOW DATABASE {db_name}")
                        record = res.single()
                        if record and record.get("currentStatus") == "online":
                            online = True
                            break
                        time.sleep(0.5)
                    
                    if online:
                        logger.info(f"Database '{db_name}' is now online.")
                    else:
                        logger.warning(f"Database '{db_name}' was created but status is not online yet.")
                    
                    # Update our target database name
                    self.database_name = db_name
                except Exception as e:
                    logger.warning(
                        f"Could not create database '{db_name}'. \n"
                        f"Note: Multiple databases (CREATE DATABASE query) require Neo4j Desktop or Enterprise. \n"
                        f"If you are running Neo4j Community Edition, it only supports the default 'neo4j' database. \n"
                        f"We will fall back to using your default database. Error details: {e}"
                    )
                    # Reset database name to default 'neo4j' to prevent subsequent DatabaseNotFound errors
                    self.database_name = "neo4j"


    def _save_mock_db(self):
        """Saves the mock database state to a local JSON file."""
        os.makedirs("output", exist_ok=True)
        with open(self.mock_db_path, "w", encoding="utf-8") as f:
            json.dump({
                "nodes": self._mock_nodes,
                "edges": self._mock_edges
            }, f, indent=2)
        logger.debug(f"Saved mock DB state to {self.mock_db_path}")

    def _load_mock_db(self):
        """Loads the mock database state from a local JSON file if it exists."""
        if os.path.exists(self.mock_db_path):
            try:
                with open(self.mock_db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._mock_nodes = data.get("nodes", {})
                    self._mock_edges = data.get("edges", [])
                logger.info(f"Loaded mock DB state from {self.mock_db_path} ({len(self._mock_nodes)} nodes, {len(self._mock_edges)} edges).")
            except Exception as e:
                logger.warning(f"Failed to load mock DB file: {e}")


    def close(self):
        """Closes the connection to the Neo4j database driver."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j driver connection closed.")

    def clear_database(self):
        """Wipes all nodes and edges from the graph database to start fresh."""
        logger.info(f"Clearing database (Mode: {self.mode})")
        if self.mode == "neo4j":
            # MATCH (n) DETACH DELETE n matches all nodes in the database,
            # detaches their relationship links, and deletes them.
            with self.driver.session(database=self.database_name) as session:
                session.run("MATCH (n) DETACH DELETE n")
        else:
            self._mock_nodes.clear()
            self._mock_edges.clear()
            self._save_mock_db()

    def create_constraints(self):
        """Sets up uniqueness constraints to prevent duplicate entity IDs."""
        logger.info(f"Creating database constraints (Mode: {self.mode})")
        if self.mode == "neo4j":
            # Neo4j constraints ensure that two nodes of the same label
            # cannot share the same 'id'. This handles deduplication natively.
            with self.driver.session(database=self.database_name) as session:
                try:
                    session.run("CREATE CONSTRAINT character_id_unique IF NOT EXISTS FOR (c:Character) REQUIRE c.id IS UNIQUE")
                    session.run("CREATE CONSTRAINT location_id_unique IF NOT EXISTS FOR (l:Location) REQUIRE l.id IS UNIQUE")
                    session.run("CREATE CONSTRAINT community_id_unique IF NOT EXISTS FOR (co:Community) REQUIRE co.id IS UNIQUE")
                except exceptions.ClientError as e:
                    # In some older Neo4j versions, 'IF NOT EXISTS' syntax might vary.
                    logger.warning(f"Constraint creation warning: {e}")
        else:
            # In mock mode, we enforce uniqueness programmatically in Python
            pass

    def upsert_character(self, character_id: str, name: str, description: str, aliases: List[str], embedding: List[float]):
        """
        Creates or updates a Character node in the graph.
        
        We log this at DEBUG level as required by the assignment rules
        to show how entities transform into database queries.
        """
        logger.debug(f"DB Upsert Character -> ID: {character_id}, Name: {name}, Aliases: {aliases}")
        
        if self.mode == "neo4j":
            # MERGE matches the node by ID.
            # If the node doesn't exist (ON CREATE), it initializes it.
            # If it already exists (ON MATCH), it appends new descriptions and merges aliases.
            query = """
            MERGE (c:Character {id: $id})
            ON CREATE SET 
                c.name = $name, 
                c.description = $description, 
                c.aliases = $aliases, 
                c.embedding = $embedding
            ON MATCH SET 
                c.description = c.description + " | " + $description,
                c.aliases = REDUCE(s = c.aliases, x IN $aliases | CASE WHEN x IN s THEN s ELSE s + x END),
                c.embedding = $embedding
            """
            logger.debug(f"Executing Cypher: {query.strip()} with parameters ID={character_id}")
            with self.driver.session(database=self.database_name) as session:
                session.run(
                    query, 
                    id=character_id, 
                    name=name, 
                    description=description, 
                    aliases=aliases, 
                    embedding=embedding
                )
        else:
            # Mock mode implementation
            if character_id in self._mock_nodes:
                node = self._mock_nodes[character_id]
                node["description"] += " | " + description
                # Merge unique aliases
                existing_aliases = set(node.get("aliases", []))
                for a in aliases:
                    existing_aliases.add(a)
                node["aliases"] = list(existing_aliases)
                node["embedding"] = embedding
            else:
                self._mock_nodes[character_id] = {
                    "id": character_id,
                    "label": "Character",
                    "name": name,
                    "description": description,
                    "aliases": aliases,
                    "embedding": embedding
                }
            self._save_mock_db()

    def upsert_location(self, location_id: str, name: str, description: str, aliases: List[str], embedding: List[float]):
        """Creates or updates a Location node in the graph."""
        logger.debug(f"DB Upsert Location -> ID: {location_id}, Name: {name}")
        
        if self.mode == "neo4j":
            query = """
            MERGE (l:Location {id: $id})
            ON CREATE SET 
                l.name = $name, 
                l.description = $description, 
                l.aliases = $aliases, 
                l.embedding = $embedding
            ON MATCH SET 
                l.description = l.description + " | " + $description,
                l.aliases = REDUCE(s = l.aliases, x IN $aliases | CASE WHEN x IN s THEN s ELSE s + x END),
                l.embedding = $embedding
            """
            logger.debug(f"Executing Cypher: {query.strip()} with parameters ID={location_id}")
            with self.driver.session(database=self.database_name) as session:
                session.run(
                    query, 
                    id=location_id, 
                    name=name, 
                    description=description, 
                    aliases=aliases, 
                    embedding=embedding
                )
        else:
            if location_id in self._mock_nodes:
                node = self._mock_nodes[location_id]
                node["description"] += " | " + description
                existing_aliases = set(node.get("aliases", []))
                for a in aliases:
                    existing_aliases.add(a)
                node["aliases"] = list(existing_aliases)
                node["embedding"] = embedding
            else:
                self._mock_nodes[location_id] = {
                    "id": location_id,
                    "label": "Location",
                    "name": name,
                    "description": description,
                    "aliases": aliases,
                    "embedding": embedding
                }
            self._save_mock_db()

    def upsert_relationship(
        self, 
        source_id: str, 
        source_label: str, 
        target_id: str, 
        target_label: str, 
        rel_type: str, 
        properties: Dict[str, Any]
    ):
        """
        Creates or updates a relationship link between two nodes in the database.
        
        Parameters:
        - source_id: ID of the starting node (e.g., 'tony_stark')
        - source_label: Label of the starting node ('Character' or 'Location')
        - target_id: ID of the target node
        - target_label: Label of the target node
        - rel_type: Type of relationship (e.g., 'INTERACTED_WITH' or 'LOCATED_IN')
        - properties: Attributes of the edge (e.g. {'description': 'co-conspirators in NYC'})
        """
        logger.debug(f"DB Upsert Relationship: ({source_id}:{source_label}) -[{rel_type}]-> ({target_id}:{target_label})")
        
        if self.mode == "neo4j":
            # We match the source and target nodes by their unique IDs, 
            # then merge the relationship link between them.
            # We dynamically set properties on the relationship.
            query = f"""
            MATCH (s:{source_label} {{id: $source_id}})
            MATCH (t:{target_label} {{id: $target_id}})
            MERGE (s)-[r:{rel_type}]->(t)
            ON CREATE SET r.description = $description, r.chunk_ids = $chunk_ids
            ON MATCH SET r.description = r.description + " | " + $description,
                         r.chunk_ids = REDUCE(s = r.chunk_ids, x IN $chunk_ids | CASE WHEN x IN s THEN s ELSE s + x END)
            """
            description = properties.get("description", "")
            chunk_ids = properties.get("chunk_ids", [])
            
            logger.debug(f"Executing Cypher: {query.strip()} with parameters source_id={source_id}, target_id={target_id}")
            with self.driver.session(database=self.database_name) as session:
                session.run(
                    query, 
                    source_id=source_id, 
                    target_id=target_id, 
                    description=description, 
                    chunk_ids=chunk_ids
                )
        else:
            # Mock mode implementation
            # Verify source and target exist
            if source_id not in self._mock_nodes or target_id not in self._mock_nodes:
                logger.warning(f"Mock DB: Cannot create relationship since source {source_id} or target {target_id} does not exist.")
                return
            
            # Check if edge already exists
            found = False
            for edge in self._mock_edges:
                if (edge["source"] == source_id and 
                    edge["target"] == target_id and 
                    edge["type"] == rel_type):
                    edge["properties"]["description"] += " | " + properties.get("description", "")
                    edge["properties"]["chunk_ids"] = list(set(edge["properties"].get("chunk_ids", []) + properties.get("chunk_ids", [])))
                    found = True
                    break
            
            if not found:
                self._mock_edges.append({
                    "source": source_id,
                    "target": target_id,
                    "type": rel_type,
                    "properties": {
                        "description": properties.get("description", ""),
                        "chunk_ids": properties.get("chunk_ids", [])
                    }
                })
            self._save_mock_db()

    def get_nodes(self) -> List[Dict[str, Any]]:
        """Returns all Character and Location nodes in the graph."""
        if self.mode == "neo4j":
            query = "MATCH (n) WHERE n:Character OR n:Location RETURN n.id as id, labels(n)[0] as label, n.name as name, n.description as description, n.aliases as aliases, n.embedding as embedding"
            with self.driver.session(database=self.database_name) as session:
                result = session.run(query)
                return [dict(record) for record in result]
        else:
            return list(self._mock_nodes.values())

    def get_edges(self) -> List[Dict[str, Any]]:
        """Returns all relationships in the graph."""
        if self.mode == "neo4j":
            query = "MATCH (s)-[r]->(t) WHERE NOT s:Community AND NOT t:Community RETURN s.id as source, t.id as target, type(r) as type, r.description as description, r.chunk_ids as chunk_ids"
            with self.driver.session(database=self.database_name) as session:
                result = session.run(query)
                edges = []
                for record in result:
                    edges.append({
                        "source": record["source"],
                        "target": record["target"],
                        "type": record["type"],
                        "properties": {
                            "description": record["description"],
                            "chunk_ids": record["chunk_ids"]
                        }
                    })
                return edges
        else:
            return self._mock_edges

    def add_node_to_community(self, node_id: str, label: str, community_id: str):
        """Links a node (Character/Location) to a specific community node."""
        logger.debug(f"DB Link Node {node_id} to Community {community_id}")
        if self.mode == "neo4j":
            # We MERGE the Community node first to make sure it exists, 
            # then link the Character/Location to it via a BELONGS_TO relationship.
            query = f"""
            MERGE (co:Community {{id: $community_id}})
            WITH co
            MATCH (n:{label} {{id: $node_id}})
            MERGE (n)-[:BELONGS_TO]->(co)
            """
            with self.driver.session(database=self.database_name) as session:
                session.run(query, community_id=community_id, node_id=node_id)
        else:
            # Set community_id on mock node
            if node_id in self._mock_nodes:
                self._mock_nodes[node_id]["community_id"] = community_id
            
            # Make sure Community node exists in our mock nodes map
            comm_node_id = f"comm_{community_id}"
            if comm_node_id not in self._mock_nodes:
                self._mock_nodes[comm_node_id] = {
                    "id": community_id,
                    "label": "Community",
                    "summary": ""
                }
            self._save_mock_db()

    def upsert_community(self, community_id: str, summary: str):
        """Creates or updates a Community node, setting its summary report."""
        logger.debug(f"DB Upsert Community {community_id} summary")
        if self.mode == "neo4j":
            query = """
            MERGE (co:Community {id: $id})
            SET co.summary = $summary
            """
            with self.driver.session(database=self.database_name) as session:
                session.run(query, id=community_id, summary=summary)
        else:
            comm_node_id = f"comm_{community_id}"
            if comm_node_id in self._mock_nodes:
                self._mock_nodes[comm_node_id]["summary"] = summary
            else:
                self._mock_nodes[comm_node_id] = {
                    "id": community_id,
                    "label": "Community",
                    "summary": summary
                }
            self._save_mock_db()

    def get_community_summaries(self) -> Dict[str, str]:
        """Returns a mapping of community_id -> summary for all communities."""
        summaries = {}
        if self.mode == "neo4j":
            query = "MATCH (co:Community) RETURN co.id as id, co.summary as summary"
            with self.driver.session(database=self.database_name) as session:
                result = session.run(query)
                for record in result:
                    if record["summary"]:
                        summaries[record["id"]] = record["summary"]
        else:
            for node in self._mock_nodes.values():
                if node.get("label") == "Community" and node.get("summary"):
                    summaries[node["id"]] = node["summary"]
        return summaries

    def get_nodes_in_community(self, community_id: str) -> List[Dict[str, Any]]:
        """Returns all character/location nodes that belong to a specific community."""
        if self.mode == "neo4j":
            query = """
            MATCH (n)-[:BELONGS_TO]->(co:Community {id: $community_id})
            RETURN n.id as id, labels(n)[0] as label, n.name as name, n.description as description
            """
            with self.driver.session(database=self.database_name) as session:
                result = session.run(query, community_id=community_id)
                return [dict(record) for record in result]
        else:
            members = []
            for node in self._mock_nodes.values():
                if node.get("community_id") == community_id:
                    members.append({
                        "id": node["id"],
                        "label": node["label"],
                        "name": node["name"],
                        "description": node["description"]
                    })
            return members

    def get_edges_in_community(self, community_id: str) -> List[Dict[str, Any]]:
        """Returns all relationships where both nodes belong to the specified community."""
        if self.mode == "neo4j":
            query = """
            MATCH (s)-[r]->(t)
            MATCH (s)-[:BELONGS_TO]->(co:Community {id: $community_id})
            MATCH (t)-[:BELONGS_TO]->(co:Community {id: $community_id})
            RETURN s.id as source, t.id as target, type(r) as type, r.description as description
            """
            with self.driver.session(database=self.database_name) as session:
                result = session.run(query, community_id=community_id)
                edges = []
                for record in result:
                    edges.append({
                        "source": record["source"],
                        "target": record["target"],
                        "type": record["type"],
                        "properties": {
                            "description": record["description"]
                        }
                    })
                return edges
        else:
            edges = []
            # Gather nodes in this community
            member_ids = {node["id"] for node in self._mock_nodes.values() if node.get("community_id") == community_id}
            for edge in self._mock_edges:
                if edge["source"] in member_ids and edge["target"] in member_ids:
                    edges.append(edge)
            return edges

    def get_node_embeddings(self, label: str) -> List[Dict[str, Any]]:
        """Retrieves node IDs, names, and embeddings for vector similarity search."""
        if self.mode == "neo4j":
            query = f"MATCH (n:{label}) RETURN n.id as id, n.name as name, n.embedding as embedding"
            with self.driver.session(database=self.database_name) as session:
                result = session.run(query)
                return [dict(record) for record in result if record["embedding"] is not None]
        else:
            nodes = []
            for node in self._mock_nodes.values():
                if node.get("label") == label and "embedding" in node:
                    nodes.append({
                        "id": node["id"],
                        "name": node["name"],
                        "embedding": node["embedding"]
                    })
            return nodes

    def traverse_paths(self, start_node_id: str, max_hops: int) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """
        Traverses paths from a starting node out to a given number of hops (1-3).
        Returns:
        - List of visited node IDs
        - List of traversed relationship ID strings (e.g. "A->B")
        - List of detailed source information dictionaries
        """
        logger.debug(f"DB Graph Traversal from {start_node_id} up to {max_hops} hops")
        
        visited_nodes = set([start_node_id])
        traversed_edges = set()
        sources = []

        if self.mode == "neo4j":
            # Cypher path query: matches paths from start node to other nodes.
            # *1..{max_hops} expands the path dynamically from 1 link to max_hops links.
            query = f"""
            MATCH path = (start {{id: $start_id}})-[r:INTERACTED_WITH|LOCATED_IN*1..{max_hops}]-(other)
            RETURN nodes(path) as nodes, relationships(path) as rels
            """
            with self.driver.session(database=self.database_name) as session:
                result = session.run(query, start_id=start_node_id)
                for record in result:
                    path_nodes = record["nodes"]
                    path_rels = record["rels"]
                    
                    # Accumulate visited nodes in this path
                    for node in path_nodes:
                        visited_nodes.add(node["id"])
                    
                    # Accumulate relationships and extract source details
                    for rel in path_rels:
                        source_id = rel.start_node["id"]
                        target_id = rel.end_node["id"]
                        rel_type = rel.type
                        edge_str = f"{source_id}-{rel_type}-{target_id}"
                        traversed_edges.add(edge_str)
                        
                        # Generate a unique source_id based on description to prevent deduplication of multiple interactions
                        desc_hash = abs(hash(rel.get("description", ""))) % 100000
                        unique_src_id = f"{edge_str}_{desc_hash}"
                        
                        # Add relationship details to source text contexts
                        sources.append({
                            "source_id": unique_src_id,
                            "text_chunk": f"Connection: {source_id} interacts with {target_id} (details: {rel.get('description', '')})",
                            "relevance_score": 0.9  # Set high relevance for direct graph path connections
                        })
        else:
            # Mock mode path traversal using NetworkX
            g = nx.Graph()
            # Build networkx graph from mock data
            for node_id, node in self._mock_nodes.items():
                if node.get("label") in ["Character", "Location"]:
                    g.add_node(node_id)
            for edge in self._mock_edges:
                g.add_edge(edge["source"], edge["target"], type=edge["type"], description=edge["properties"]["description"])
            
            if start_node_id in g:
                # Find all nodes within max_hops using ego_graph or single_source_shortest_path_length
                lengths = nx.single_source_shortest_path_length(g, start_node_id, cutoff=max_hops)
                for node_id in lengths:
                    visited_nodes.add(node_id)
                
                # Identify traversed edges within this visited subgraph
                for edge in self._mock_edges:
                    s, t = edge["source"], edge["target"]
                    if s in visited_nodes and t in visited_nodes:
                        edge_str = f"{s}-{edge['type']}-{t}"
                        traversed_edges.add(edge_str)
                        
                        desc_hash = abs(hash(edge['properties']['description'])) % 100000
                        unique_src_id = f"{edge_str}_{desc_hash}"
                        
                        sources.append({
                            "source_id": unique_src_id,
                            "text_chunk": f"Connection: {s} interacts with {t} (details: {edge['properties']['description']})",
                            "relevance_score": 0.9
                        })

        return list(visited_nodes), list(traversed_edges), sources

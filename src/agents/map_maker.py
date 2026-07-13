import json
import re
import logging
from typing import List, Dict, Any, Tuple
from src.database import GraphDatabaseClient
from src import llm

# Initialize logger for tracking extraction steps
logger = logging.getLogger("graph_rag.map_maker")

class MapMakerAgent:
    """
    Agent responsible for reading raw text, extracting entities and relationships,
    resolving duplicate names (deduplication), generating embeddings, and loading
    the data into the Neo4j graph database.
    """
    def __init__(self, db_client: GraphDatabaseClient):
        self.db = db_client

    def run_ingestion(self, text: str, chunk_size: int = 1500, overlap: int = 200):
        """
        Main entrypoint to ingest a chapter/book text.
        
        Steps:
        1. Split the text into overlapping page chunks.
        2. Extract characters, locations, and relationships page-by-page using Groq.
        3. Deduplicate extracted entity names globally.
        4. Generate embeddings and upsert nodes and edges into the database.
        """
        logger.info("Starting text ingestion pipeline...")
        
        # Step 1: Chunk text
        chunks = self._chunk_text(text, chunk_size, overlap)
        logger.info(f"Split raw text into {len(chunks)} chunks.")

        all_characters = []
        all_locations = []
        all_relationships = []

        # Step 2: Page-by-page extraction
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}"
            logger.info(f"Processing chunk {i+1}/{len(chunks)} ({chunk_id})...")
            
            chars, locs, rels = self._extract_entities_and_relations(chunk, chunk_id)
            
            all_characters.extend(chars)
            all_locations.extend(locs)
            all_relationships.extend(rels)

        logger.info(f"Extraction complete. Found {len(all_characters)} character references, "
                    f"{len(all_locations)} location references, and {len(all_relationships)} relationships.")

        # Step 3: Run Batch Entity Resolution (Deduplication)
        resolved_names = self._resolve_entities(all_characters, all_locations)
        
        # Step 4: Write to Graph Database
        self._load_to_database(all_characters, all_locations, all_relationships, resolved_names)
        logger.info("Ingestion pipeline successfully completed!")

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Splits a large string into overlapping smaller text chunks.
        Overlaps ensure that relationships spanning across boundaries aren't lost.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start += chunk_size - overlap
        return chunks

    def _extract_entities_and_relations(self, text_chunk: str, chunk_id: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Calls the Groq LLM to extract entities and relations in a clean JSON format.
        """
        system_prompt = (
            "You are a Graph AI Extractor. Read the text snippet and extract all characters, "
            "locations, and their relationships. Return the output STRICTLY as a valid JSON object. "
            "Do not include any chat prefix or markdown formatting outside of the JSON block."
        )

        prompt = f"""
        Extract characters, locations, and relationships from this text chunk:
        ---
        {text_chunk}
        ---

        Your output must follow this JSON schema:
        {{
          "characters": [
            {{
              "name": "Canonical Character Name",
              "description": "One sentence describing who they are in this snippet",
              "aliases": ["Alias 1", "Alias 2"]
            }}
          ],
          "locations": [
            {{
              "name": "Canonical Location Name",
              "description": "One sentence describing what this location is",
              "aliases": ["Alias 1"]
            }}
          ],
          "relationships": [
            {{
              "source": "Name of source character/location",
              "target": "Name of target character/location",
              "type": "INTERACTED_WITH" or "LOCATED_IN",
              "description": "One sentence describing how they relate or interact in this snippet"
            }}
          ]
        }}
        """

        try:
            # We call the Groq LLM using temperature=0.1 to get structured, factual extractions
            response_text = llm.generate_completion(prompt, system_prompt, temperature=0.1)
            
            # Clean up potential markdown formatting (e.g. ```json ... ```)
            cleaned_response = self._clean_json_string(response_text)
            data = json.loads(cleaned_response)

            characters = data.get("characters", [])
            locations = data.get("locations", [])
            relationships = data.get("relationships", [])

            # Attach the source chunk ID to each relationship for traceback/sourcing
            for rel in relationships:
                rel["chunk_id"] = chunk_id

            return characters, locations, relationships

        except Exception as e:
            logger.error(f"Error parsing LLM response for chunk {chunk_id}: {e}")
            logger.debug(f"Raw response: {response_text if 'response_text' in locals() else 'None'}")
            return [], [], []

    def _clean_json_string(self, text: str) -> str:
        """
        Extracts the raw JSON substring from the LLM's response, stripping away
        any code blocks or conversational wrappers.
        """
        # Search for text enclosed between ```json and ```
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Search for text enclosed between standard ``` and ```
        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _resolve_entities(self, characters: List[Dict], locations: List[Dict]) -> Dict[str, str]:
        """
        Resolves similar or duplicate names into a single canonical identifier.
        For example: "Tony Stark", "Stark", and "Iron Man" -> "Tony Stark".
        
        Returns a dictionary mapping: original_extracted_name -> canonical_name
        """
        logger.info("Running entity resolution and deduplication...")
        
        # Get unique names from all characters and locations
        unique_char_names = list(set(c["name"] for c in characters))
        unique_loc_names = list(set(l["name"] for l in locations))

        if not unique_char_names and not unique_loc_names:
            return {}

        system_prompt = (
            "You are a Graph Entity Resolution Agent. Your job is to analyze lists of entities "
            "extracted from a book and identify which names refer to the exact same real-world entity "
            "(e.g., 'Tony Stark' and 'Iron Man' and 'Stark' are the same character; 'Stark Tower' and "
            "'Stark HQ' are the same location). Return a JSON mapping of each alias to its canonical name."
        )

        prompt = f"""
        Given the following extracted characters and locations:
        
        Characters: {unique_char_names}
        Locations: {unique_loc_names}

        Group names that refer to the same identity. For each group, decide on a single canonical name 
        (usually the most complete or formal name, e.g., 'Tony Stark' instead of 'Stark').
        
        Return a JSON object where the keys are the names from the list above, and the values are their resolved canonical names.
        
        Example Output Schema:
        {{
          "Tony": "Tony Stark",
          "Iron Man": "Tony Stark",
          "Tony Stark": "Tony Stark",
          "Stark Tower": "Stark Tower",
          "Stark HQ": "Stark Tower"
        }}
        """

        try:
            response_text = llm.generate_completion(prompt, system_prompt, temperature=0.1)
            cleaned_response = self._clean_json_string(response_text)
            resolution_map = json.loads(cleaned_response)
            
            logger.info(f"Deduplication completed. Resolved {len(resolution_map)} entity names.")
            return resolution_map
        except Exception as e:
            logger.error(f"Error during entity resolution LLM call: {e}. Falling back to default identity mapping.")
            # Fallback: map each name to itself (no deduplication)
            fallback_map = {}
            for name in unique_char_names + unique_loc_names:
                fallback_map[name] = name
            return fallback_map

    def _normalize_id(self, name: str) -> str:
        """
        Converts an entity name into a standardized node ID (snake_case, lower).
        e.g., "Tony Stark" -> "tony_stark"
        """
        clean_name = re.sub(r"[^\w\s]", "", name).strip() # Remove special punctuation
        return re.sub(r"\s+", "_", clean_name).lower()     # Replace spaces with underscores

    def _load_to_database(
        self, 
        characters: List[Dict], 
        locations: List[Dict], 
        relationships: List[Dict], 
        resolved_names: Dict[str, str]
    ):
        """
        Iterates over the resolved entities and relationships, generates embeddings,
        and executes database writes.
        """
        # Step 4.1: Write Character Nodes
        # Keep track of unique canonical characters to avoid repeating embedding calls
        processed_chars = {}
        for char in characters:
            original_name = char["name"]
            # Look up the resolved canonical name
            canonical_name = resolved_names.get(original_name, original_name)
            char_id = self._normalize_id(canonical_name)

            if char_id not in processed_chars:
                processed_chars[char_id] = {
                    "name": canonical_name,
                    "description": char["description"],
                    "aliases": set(char.get("aliases", []) + [original_name])
                }
            else:
                # Merge description and alias lists
                processed_chars[char_id]["description"] += " | " + char["description"]
                processed_chars[char_id]["aliases"].add(original_name)
                for a in char.get("aliases", []):
                    processed_chars[char_id]["aliases"].add(a)

        # Upsert characters to database
        for char_id, data in processed_chars.items():
            name = data["name"]
            description = data["description"]
            aliases = list(data["aliases"])
            
            # Create a combined text representation of the character's profile for the embedding vector
            profile_text = f"Character: {name}. Description: {description}. Aliases: {', '.join(aliases)}."
            embedding = llm.get_embedding(profile_text)
            
            # Database upsert (will log detailed mutation queries at DEBUG/INFO levels inside database.py)
            self.db.upsert_character(char_id, name, description, aliases, embedding)

        # Step 4.2: Write Location Nodes
        processed_locs = {}
        for loc in locations:
            original_name = loc["name"]
            canonical_name = resolved_names.get(original_name, original_name)
            loc_id = self._normalize_id(canonical_name)

            if loc_id not in processed_locs:
                processed_locs[loc_id] = {
                    "name": canonical_name,
                    "description": loc["description"],
                    "aliases": set(loc.get("aliases", []) + [original_name])
                }
            else:
                processed_locs[loc_id]["description"] += " | " + loc["description"]
                processed_locs[loc_id]["aliases"].add(original_name)
                for a in loc.get("aliases", []):
                    processed_locs[loc_id]["aliases"].add(a)

        # Upsert locations to database
        for loc_id, data in processed_locs.items():
            name = data["name"]
            description = data["description"]
            aliases = list(data["aliases"])
            
            profile_text = f"Location: {name}. Description: {description}. Aliases: {', '.join(aliases)}."
            embedding = llm.get_embedding(profile_text)
            
            self.db.upsert_location(loc_id, name, description, aliases, embedding)

        # Step 4.3: Write Relationships (Edges)
        # Note: We must translate relationship source/target names into their resolved canonical IDs
        for rel in relationships:
            src_orig = rel["source"]
            tgt_orig = rel["target"]
            rel_type = rel["type"]
            description = rel["description"]
            chunk_id = rel["chunk_id"]

            src_canonical = resolved_names.get(src_orig, src_orig)
            tgt_canonical = resolved_names.get(tgt_orig, tgt_orig)

            src_id = self._normalize_id(src_canonical)
            tgt_id = self._normalize_id(tgt_canonical)

            # Determine whether the source and target are Characters or Locations
            src_label = "Character" if src_id in processed_chars else "Location"
            tgt_label = "Character" if tgt_id in processed_chars else "Location"

            # If an entity is not in either, default to Character
            if src_id not in processed_chars and src_id not in processed_locs:
                src_label = "Character"
            if tgt_id not in processed_chars and tgt_id not in processed_locs:
                tgt_label = "Character"

            # Write relationship to database
            properties = {
                "description": description,
                "chunk_ids": [chunk_id]
            }
            self.db.upsert_relationship(src_id, src_label, tgt_id, tgt_label, rel_type, properties)

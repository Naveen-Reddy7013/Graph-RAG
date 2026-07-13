import os
import sys
import json
import logging
import argparse
import re
from pypdf import PdfReader
from src.database import GraphDatabaseClient
from src.orchestrator import GraphRAGOrchestrator
from src import llm

def setup_logging():
    """
    Configures logging to print messages to the console (stderr).
    Matches Requirement 5: explicit logging at INFO and DEBUG levels showing 
    step-by-step how text transforms into database mutations.
    """
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.DEBUG, 
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def read_chapters(chapters_dir: str) -> str:
    """Reads and concatenates all chapter text files in alphabetical order."""
    if not os.path.exists(chapters_dir):
        os.makedirs(chapters_dir)
        logger = logging.getLogger("run")
        logger.warning(f"Chapters directory '{chapters_dir}' was empty/missing and has been created.")
        return ""

    chapter_files = [f for f in os.listdir(chapters_dir) if f.endswith(".txt")]
    chapter_files.sort()

    full_text = ""
    logger = logging.getLogger("run")
    
    for filename in chapter_files:
        filepath = os.path.join(chapters_dir, filename)
        logger.info(f"Reading source chapter: {filename}")
        with open(filepath, "r", encoding="utf-8") as f:
            full_text += f.read() + "\n\n"
            
    return full_text

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text page-by-page from a PDF file using pypdf."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    
    logger = logging.getLogger("run")
    logger.info(f"Extracting text from PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n\n"
    logger.info(f"Successfully extracted {len(text)} characters from {len(reader.pages)} PDF pages.")
    return text

# --- Mock LLM implementation for local demonstration without API keys ---

def extract_nouns(text: str) -> list:
    """Helper to extract proper nouns (capitalized words) for dynamic mock graphs."""
    words = re.findall(r'\b[A-Z][a-z]+\b', text)
    stopwords = {
        "The", "He", "She", "It", "They", "But", "Meanwhile", "Then", "In", "Under", "On", "At", 
        "A", "An", "And", "Or", "If", "Of", "To", "For", "With", "By", "As", "This", "That", 
        "Inside", "Outside", "In", "Out", "Stark", "Tony", "Pepper", "Obadiah"
    }
    candidates = [w for w in words if w not in stopwords]
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)
    return unique_candidates

def get_simulated_completion(prompt: str, system_prompt: str = "", temperature: float = 0.1, max_tokens: int = 2000) -> str:
    """Returns pre-defined JSON extractions or narrative text matching our sample chapters."""
    
    # 0. Query Synthesis check (differentiate query answering from extraction)
    if "Tracer Agent" in system_prompt or "Global Thinker" in system_prompt or "narrative response" in prompt.lower() or "answer this query" in prompt.lower():
        return (
            "### Introduction to the Conflict\n\n"
            "This is a detailed, structured narrative answer describing character connections, setting layouts, and thematic conflicts. "
            "The story takes place in New York City, a major metropolitan center that serves as the strategic backdrop for Stark Industries' technological operations. "
            "At the geographical and structural center of this narrative is Stark Tower, a massive state-of-the-art skyscraper built by Tony Stark. "
            "This tower is not just a corporate headquarters, but also represents a modern monument to clean energy, powered by an advanced arc reactor. "
            "Tony Stark stands on the high balcony of the tower, reflecting on his legacy and his responsibilities as a genius inventor. "
            "Inside, the main laboratory is managed by Pepper Potts, the highly capable CEO of Stark Industries who shares a deep personal and professional connection with Tony. "
            "Pepper Potts is reviewing flight logs for the Stark Industries jet when Tony enters to talk to her about their upcoming research trip to Malibu. "
            "Their conversation reveals the operational dynamics of the tower: Pepper reminds Tony that the core arc reactor inside Stark Tower requires urgent maintenance, "
            "warning him of potential system failures if it is neglected.\n\n"
            "### The Intrusion of Obadiah Stane\n\n"
            "The peace of Stark Tower is shattered by the actions of Obadiah Stane, a ruthless business executive and corporate rival. "
            "Obadiah Stane is deeply jealous of Tony Stark's genius and covets the technology housed inside Stark Tower. "
            "Seeking to force a hostile corporate takeover, Obadiah Stane decides to act under the cover of night, sneaks into Stark Tower, and bypasses the advanced security systems. "
            "In the lobby of Stark Tower, Obadiah Stane confronts Pepper Potts, who is carrying an encrypted USB drive containing vital corporate secrets. "
            "Obadiah Stane threatens Pepper Potts, demanding she hand over the encrypted files. "
            "This moment represents a high-stakes clash between the loyal defenders of Stark Industries and its external corporate threats.\n\n"
            "### The Intervention and Character Network\n\n"
            "The confrontation escalates quickly, but Tony Stark arrives in time to intervene and confront Obadiah Stane directly. "
            "Tony Stark forces Obadiah Stane to leave the premises, temporarily securing Stark Tower and protecting Pepper Potts from harm. "
            "This network of interactions forms a multi-hop character path: Obadiah Stane is connected to Pepper Potts through a hostile confrontation in the lobby. "
            "Pepper Potts is connected to Tony Stark through a close alliance and professional collaboration. "
            "Tony Stark is connected to Stark Tower as its builder, power source, and protector, and Pepper Potts is located in Stark Tower as its operational manager. "
            "Thus, Stark Tower acts as the central hub connecting all characters, and the corporate secrets serve as the primary item of value that drives the plot. "
            "By tracing the graph paths: Obadiah Stane interacts with Pepper Potts at Stark Tower, Pepper Potts is allied with Tony Stark, and Tony Stark confronts Obadiah Stane to resolve the intrusion. "
            "This forms a complete, closed-loop network showing the flow of conflict from Obadiah's jealousy to Tony's protective intervention, illustrating how a local search traversal of the graph "
            "recovers the complete narrative context."
        )

    # 1. Entity Resolution Mock (check this first to prevent overlap)
    if "canonical name" in prompt or "Entity Resolution" in prompt or "Group names" in prompt:
        # If the prompt contains Tony Stark, return the hardcoded map
        if "Tony Stark" in prompt:
            return json.dumps({
                "Tony Stark": "Tony Stark",
                "Iron Man": "Tony Stark",
                "Tony": "Tony Stark",
                "Pepper Potts": "Pepper Potts",
                "Pepper": "Pepper Potts",
                "Obadiah Stane": "Obadiah Stane",
                "Stane": "Obadiah Stane",
                "Stark Tower": "Stark Tower",
                "Stark HQ": "Stark Tower"
            })
        else:
            # Dynamic identity resolution map: map each extracted name to itself
            # Locate words enclosed in quotes or in lists
            extracted_names = re.findall(r"['\"]([A-Z][a-zA-Z\s]+)['\"]", prompt)
            res_map = {}
            for name in extracted_names:
                res_map[name] = name
            return json.dumps(res_map)

    # 2. Community Summarization Mock
    elif "Community ID" in prompt or "Summarize the story context" in prompt:
        if "obadiah_stane" in prompt:
            return "Community Report: Obadiah's Faction. Obadiah Stane is a rogue executive who sneaks into Stark Tower to confront Pepper Potts in his bid to steal secrets."
        elif "pepper_potts" in prompt:
            return "Community Report: Stark Tower Allies. This community contains Tony Stark and Pepper Potts working in Stark Tower. They cooperate closely on reactor maintenance."
        else:
            # Generic community summary report
            return "Community Report: Story Cohort. This group of connected entities operates together in the setting described in the ingested chapters."

    # 3. Faithfulness Evaluation Mock
    elif "faithfulness_score" in prompt:
        return json.dumps({
            "faithfulness_score": 0.98,
            "factual_gaps": []
        })

    # 4. Combined Extraction Mock (if both Chapter 1 and Chapter 2 are chunked together)
    elif ("stands on the high balcony" in prompt or "Malibu" in prompt) and ("Obadiah Stane" in prompt or "corporate secrets" in prompt):
        return json.dumps({
            "characters": [
                {"name": "Tony Stark", "description": "A genius inventor and the hero Iron Man.", "aliases": ["Iron Man", "Tony"]},
                {"name": "Pepper Potts", "description": "The CEO of Stark Industries.", "aliases": ["Pepper"]},
                {"name": "Obadiah Stane", "description": "A corporate rival planning a hostile takeover.", "aliases": ["Stane"]}
            ],
            "locations": [
                {"name": "Stark Tower", "description": "A modern clean energy skyscraper.", "aliases": ["Stark HQ"]}
            ],
            "relationships": [
                {"source": "Tony Stark", "target": "Pepper Potts", "type": "INTERACTED_WITH", "description": "Tony enters and speaks with Pepper Potts about a trip."},
                {"source": "Tony Stark", "target": "Stark Tower", "type": "LOCATED_IN", "description": "Tony stands on the balcony of Stark Tower."},
                {"source": "Pepper Potts", "target": "Stark Tower", "type": "LOCATED_IN", "description": "Pepper Potts reviews logs inside Stark Tower."},
                {"source": "Obadiah Stane", "target": "Pepper Potts", "type": "INTERACTED_WITH", "description": "Obadiah confronts and threatens Pepper Potts in the Stark Tower lobby."},
                {"source": "Obadiah Stane", "target": "Stark Tower", "type": "LOCATED_IN", "description": "Obadiah sneaks into Stark Tower under the cover of night."},
                {"source": "Tony Stark", "target": "Obadiah Stane", "type": "INTERACTED_WITH", "description": "Tony confronts Obadiah Stane and forces him to leave."}
            ]
        })

    # 5. Extraction Mock for Chapter 1 (contains Tony's balcony scene only)
    elif "stands on the high balcony" in prompt or "Malibu" in prompt:
        return json.dumps({
            "characters": [
                {"name": "Tony Stark", "description": "A genius inventor and the hero Iron Man.", "aliases": ["Iron Man", "Tony"]},
                {"name": "Pepper Potts", "description": "The CEO of Stark Industries.", "aliases": ["Pepper"]}
            ],
            "locations": [
                {"name": "Stark Tower", "description": "A modern clean energy skyscraper.", "aliases": ["Stark HQ"]}
            ],
            "relationships": [
                {"source": "Tony Stark", "target": "Pepper Potts", "type": "INTERACTED_WITH", "description": "Tony enters and speaks with Pepper Potts about a trip."},
                {"source": "Tony Stark", "target": "Stark Tower", "type": "LOCATED_IN", "description": "Tony stands on the balcony of Stark Tower."},
                {"source": "Pepper Potts", "target": "Stark Tower", "type": "LOCATED_IN", "description": "Pepper Potts reviews logs inside Stark Tower."}
            ]
        })
        
    # 6. Extraction Mock for Chapter 2 (contains Obadiah's corporate intrusion only)
    elif "Obadiah Stane" in prompt or "hostile corporate takeover" in prompt:
        return json.dumps({
            "characters": [
                {"name": "Obadiah Stane", "description": "A corporate rival planning a hostile takeover.", "aliases": ["Stane"]},
                {"name": "Pepper Potts", "description": "The CEO of Stark Industries.", "aliases": ["Pepper"]},
                {"name": "Tony Stark", "description": "A genius inventor.", "aliases": ["Tony"]}
            ],
            "locations": [
                {"name": "Stark Tower", "description": "Stark Industries skyscraper.", "aliases": ["Stark HQ"]}
            ],
            "relationships": [
                {"source": "Obadiah Stane", "target": "Pepper Potts", "type": "INTERACTED_WITH", "description": "Obadiah confronts and threatens Pepper Potts in the Stark Tower lobby."},
                {"source": "Obadiah Stane", "target": "Stark Tower", "type": "LOCATED_IN", "description": "Obadiah sneaks into Stark Tower under the cover of night."},
                {"source": "Tony Stark", "target": "Obadiah Stane", "type": "INTERACTED_WITH", "description": "Tony confronts Obadiah Stane and forces him to leave."}
            ]
        })

    # 7. Adaptive Generic Mock Extraction Fallback for custom PDF uploads
    elif "Extract characters, locations, and relationships" in prompt:
        # Separate the raw text chunk from the prompt
        parts = prompt.split("---")
        text_chunk = parts[1].strip() if len(parts) >= 3 else prompt
        
        # Extract unique proper nouns
        names = extract_nouns(text_chunk)
        
        # Build adaptive characters and settings
        chars = names[:3] if len(names) >= 2 else ["Protagonist", "SupportingCharacter"]
        locs = names[3:5] if len(names) >= 4 else ["MainSetting"]
        
        mock_chars = []
        for c in chars:
            mock_chars.append({
                "name": c,
                "description": f"Character {c} extracted from the upload.",
                "aliases": [f"{c}_alias"]
            })
        mock_locs = []
        for l in locs:
            mock_locs.append({
                "name": l,
                "description": f"Location {l} extracted from the upload.",
                "aliases": [f"{l}_alias"]
            })
        mock_rels = []
        if len(chars) >= 2:
            mock_rels.append({
                "source": chars[0],
                "target": chars[1],
                "type": "INTERACTED_WITH",
                "description": f"{chars[0]} interacts with {chars[1]} in the chapter."
            })
        if len(chars) >= 1 and len(locs) >= 1:
            mock_rels.append({
                "source": chars[0],
                "target": locs[0],
                "type": "LOCATED_IN",
                "description": f"{chars[0]} is situated at {locs[0]}."
            })
            
        return json.dumps({
            "characters": mock_chars,
            "locations": mock_locs,
            "relationships": mock_rels
        })

    # 8. Default Grounded Query Response
    return (
        "### Introduction to the Conflict\n\n"
        "This is a detailed, structured narrative answer describing character connections, setting layouts, and thematic conflicts. "
        "The story takes place in New York City, a major metropolitan center that serves as the strategic backdrop for Stark Industries' technological operations. "
        "At the geographical and structural center of this narrative is Stark Tower, a massive state-of-the-art skyscraper built by Tony Stark. "
        "This tower is not just a corporate headquarters, but also represents a modern monument to clean energy, powered by an advanced arc reactor. "
        "Tony Stark stands on the high balcony of the tower, reflecting on his legacy and his responsibilities as a genius inventor. "
        "Inside, the main laboratory is managed by Pepper Potts, the highly capable CEO of Stark Industries who shares a deep personal and professional connection with Tony. "
        "Pepper Potts is reviewing flight logs for the Stark Industries jet when Tony enters to talk to her about their upcoming research trip to Malibu. "
        "Their conversation reveals the operational dynamics of the tower: Pepper reminds Tony that the core arc reactor inside Stark Tower requires urgent maintenance, "
        "warning him of potential system failures if it is neglected.\n\n"
        "### The Intrusion of Obadiah Stane\n\n"
        "The peace of Stark Tower is shattered by the actions of Obadiah Stane, a ruthless business executive and corporate rival. "
        "Obadiah Stane is deeply jealous of Tony Stark's genius and covets the technology housed inside Stark Tower. "
        "Seeking to force a hostile corporate takeover, Obadiah Stane decides to act under the cover of night, sneaks into Stark Tower, and bypasses the advanced security systems. "
        "In the lobby of Stark Tower, Obadiah Stane confronts Pepper Potts, who is carrying an encrypted USB drive containing vital corporate secrets. "
        "Obadiah Stane threatens Pepper Potts, demanding she hand over the encrypted files. "
        "This moment represents a high-stakes clash between the loyal defenders of Stark Industries and its external corporate threats.\n\n"
        "### The Intervention and Character Network\n\n"
        "The confrontation escalates quickly, but Tony Stark arrives in time to intervene and confront Obadiah Stane directly. "
        "Tony Stark forces Obadiah Stane to leave the premises, temporarily securing Stark Tower and protecting Pepper Potts from harm. "
        "This network of interactions forms a multi-hop character path: Obadiah Stane is connected to Pepper Potts through a hostile confrontation in the lobby. "
        "Pepper Potts is connected to Tony Stark through a close alliance and professional collaboration. "
        "Tony Stark is connected to Stark Tower as its builder, power source, and protector, and Pepper Potts is located in Stark Tower as its operational manager. "
        "Thus, Stark Tower acts as the central hub connecting all characters, and the corporate secrets serve as the primary item of value that drives the plot. "
        "By tracing the graph paths: Obadiah Stane interacts with Pepper Potts at Stark Tower, Pepper Potts is allied with Tony Stark, and Tony Stark confronts Obadiah Stane to resolve the intrusion. "
        "This forms a complete, closed-loop network showing the flow of conflict from Obadiah's jealousy to Tony's protective intervention, illustrating how a local search traversal of the graph "
        "recovers the complete narrative context."
    )

def get_simulated_embedding(text: str) -> list:
    """Generates deterministic mock embeddings for consistent testing."""
    val = sum(ord(c) for c in text) % 100 / 100.0
    return [val] * 384

# --- End of Mock LLM implementation ---

def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
        
    setup_logging()
    logger = logging.getLogger("run")

    parser = argparse.ArgumentParser(description="The 'Big Picture' Book Summarizer Pipeline")
    # Standard query arguments
    parser.add_argument("--query", type=str, default="How is Tony Stark connected to Pepper Potts?", help="The book question to ask")
    parser.add_argument("--mode", type=str, choices=["global", "local"], default="local", help="Search mode: global or local")
    parser.add_argument("--max-hops", type=int, choices=[1, 2, 3], default=2, help="Max graph search hops for local search (1-3)")
    parser.add_argument("--output-format", type=str, choices=["markdown", "json"], default="json", help="Format of the output print")
    
    # Infrastructure and pipeline arguments
    parser.add_argument("--chapters-dir", type=str, default="data/chapters", help="Directory containing book chapter txt files")
    parser.add_argument("--pdf", type=str, default=None, help="Path to an uploaded PDF story file to ingest")
    parser.add_argument("--database", type=str, default=None, help="Specify database instance to use in Neo4j (will create if missing)")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip the extraction/clustering stage")
    parser.add_argument("--no-mock-llm", dest="mock_llm", action="store_false", help="Disable mock LLM (Mock LLM is disabled by default)")
    parser.set_defaults(mock_llm=False)
    
    parser.add_argument("--chat", action="store_true", help="Start an interactive chatbot session")
    
    # Input JSON Schema parsing
    parser.add_argument("--input-json", type=str, default=None, help="Complete input parameters matching the JSON Input Schema")
    
    args = parser.parse_args()

    # We only read stdin if it is not a tty AND the user did not pass query/pdf flags
    # (to prevent blocking in environments that redirect stdin, like VS Code or terminal agents)
    input_json_data = None
    has_cli_args = any(arg in sys.argv for arg in ["--query", "--pdf", "--chapters-dir", "--input-json", "--chat"])
    
    if not sys.stdin.isatty() and not has_cli_args:
        logger.info("Piped stdin stream detected. Reading JSON Input Schema...")
        input_json_data = sys.stdin.read().strip()
    # Check CLI argument
    elif args.input_json:
        input_json_data = args.input_json

    if input_json_data:
        try:
            params = json.loads(input_json_data)
            args.query = params.get("query", args.query)
            args.mode = params.get("mode", args.mode)
            args.max_hops = int(params.get("max_hops", args.max_hops))
            args.output_format = params.get("output_format", args.output_format)
            logger.info(f"Strict Input Schema parsed successfully: {params}")
        except Exception as e:
            logger.error(f"Error parsing JSON Input Schema: {e}")
            sys.exit(1)

    # Apply mock LLM overrides if selected
    if args.mock_llm:
        logger.error("="*60)
        logger.error("ERROR: Mock/Simulated LLM mode is disabled.")
        logger.error("The engine is configured to run ONLY if Groq is available.")
        logger.error("="*60)
        sys.exit(1)

    # Verify Groq connectivity
    logger.info("Verifying Groq API Connectivity...")
    try:
        client = llm.get_groq_client()
        client.chat.completions.create(
            messages=[{"role": "user", "content": "ping"}],
            model=llm.config.GROQ_MODEL,
            max_tokens=1
        )
        logger.info("Groq API connection verified successfully.")
    except Exception as e:
        logger.error("="*60)
        logger.error(f"FATAL ERROR: Groq API is not available!")
        logger.error(f"Error Details: {e}")
        logger.error("Please verify that your GROQ_API_KEY is configured correctly in .env and that you have internet access.")
        logger.error("="*60)
        sys.exit(1)

    # Step 2: Initialize Database Client (will check/create target database)
    db_client = GraphDatabaseClient(database_name=args.database)

    try:
        # Step 3: Automatically create target database instance in Neo4j if configured
        if args.database:
            db_client.create_new_database(args.database)

        # Initialize constraints in database
        db_client.create_constraints()

        # Initialize the Orchestrator
        orchestrator = GraphRAGOrchestrator(db_client)

        # Step 4: Ingestion Stage (Map Maker + Cluster Grouper)
        if not args.skip_ingest:
            # If PDF path is provided, read text from PDF; otherwise, read from chapters folder
            if args.pdf:
                book_content = extract_text_from_pdf(args.pdf)
            else:
                book_content = read_chapters(args.chapters_dir)
                
            if not book_content.strip():
                logger.error("No book text content found to ingest. "
                             "Please specify a valid PDF (--pdf) or add chapter text files to 'data/chapters/'.")
                sys.exit(1)
                
            logger.info("Initializing Graph Ingestion Stage...")
            db_client.clear_database()
            orchestrator.ingest_chapters(book_content)
        else:
            logger.info("Skipping ingestion. Running query directly against active database.")

        # Step 5: Route to Chatbot Loop or Single Query
        if args.chat:
            logger.info("Entering interactive chatbot mode...")
            chat_history = []
            print("\n" + "="*60)
            print("GraphRAG Chatbot Mode. Type 'exit' or 'quit' to end the session.")
            print("="*60)
            while True:
                try:
                    query = input("\nChat> ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\nExiting chat. Goodbye!")
                    break
                if not query:
                    continue
                if query.lower() in ["exit", "quit"]:
                    print("\nExiting chat. Goodbye!")
                    break
                
                # Execute query pipeline passing current chat history
                result = orchestrator.query_pipeline(query, args.mode, args.max_hops, chat_history=chat_history)
                
                # Show the answer
                print(f"\n{result['answer']}")
                
                # Update the conversation chat history list
                chat_history = result.get("chat_history", [])
        else:
            logger.info(f"Executing query with mode '{args.mode}', max_hops {args.max_hops}...")
            result = orchestrator.query_pipeline(args.query, args.mode, args.max_hops)

            # Step 6: Save JSON response to output/query_result.json
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "query_result.json")
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved complete response JSON to {output_path}")

            # Step 7: Print output in requested format
            print("\n" + "="*80)
            print(f"QUERY: {result['query']}")
            print("="*80)
            
            if args.output_format == "json":
                print(json.dumps(result, indent=2))
            else:
                print(f"## Answer\n\n{result['answer']}\n")
                print(f"### Metadata\n")
                print(f"- **Query ID**: {result['query_id']}")
                print(f"- **Faithfulness Score**: {result['evaluation']['faithfulness_score']}")
                print(f"- **Factual Gaps**: {', '.join(result['evaluation']['factual_gaps']) or 'None'}")
                print(f"- **Total Nodes Visited**: {len(result['graph_context']['nodes_visited'])}")
                print(f"- **Total Edges Traversed**: {len(result['graph_context']['edges_traversed'])}")
                print(f"- **Communities Traversed**: {', '.join(result['graph_context']['communities_traversed']) or 'None'}")
                print(f"- **Execution Time**: {result['metadata']['execution_seconds']:.2f} seconds")

            # Step 8: Print timing execution breakdown
            orchestrator.print_query_breakdown(result["metadata"]["execution_seconds"])

    finally:
        db_client.close()

if __name__ == "__main__":
    main()

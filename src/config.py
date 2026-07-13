import os
from dotenv import load_dotenv

# load_dotenv() searches for a .env file in the current directory (and parent directories)
# and loads all defined variables into the operating system environment (os.environ).
# This keeps secrets (like API keys) secure and out of the source code.
load_dotenv()

# We retrieve variables from os.environ using os.getenv.
# The second parameter is a fallback default value in case the variable is not set in .env.

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# LLM and embedding configurations
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

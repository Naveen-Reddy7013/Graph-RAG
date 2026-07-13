import os
from typing import List
from groq import Groq
from sentence_transformers import SentenceTransformer
from src import config

# --- Sentence Transformers (Local Embeddings) ---

# We define a global variable to store our loaded embedding model.
# By setting it to None initially, we can load it lazily (only when first needed)
# and avoid loading it multiple times, saving RAM.
_embedding_model = None

def get_embedding_model() -> SentenceTransformer:
    """
    Loads and returns the SentenceTransformer model.
    Lazily initializes the model so it only loads once.
    """
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading local embedding model: {config.EMBEDDING_MODEL}...")
        # SentenceTransformer loads the model. If it's not cached locally,
        # it will download it from Hugging Face once.
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        print("Embedding model loaded successfully.")
    return _embedding_model

def get_embedding(text: str) -> List[float]:
    """
    Generates a dense vector representation of the input text.
    The model 'all-MiniLM-L6-v2' output is a 384-dimensional vector.
    """
    model = get_embedding_model()
    # model.encode runs the neural network forward pass on the text.
    # It returns a numpy.ndarray.
    vector = model.encode(text)
    # We convert the numpy array to a plain Python list of floats
    # because the Neo4j Python driver requires basic Python types for queries.
    return vector.tolist()

# --- Groq (Fast Cloud LLM) ---

# We do the same lazy loading pattern for the Groq client.
_groq_client = None

def get_groq_client() -> Groq:
    """
    Initializes and returns the Groq client.
    Raises an error if the GROQ_API_KEY is not set in the environment or .env.
    """
    global _groq_client
    if _groq_client is None:
        # Check if the API key is provided
        if not config.GROQ_API_KEY:
            raise ValueError(
                "Error: GROQ_API_KEY is missing! \n"
                "Please obtain a free Groq API key from https://console.groq.com/ \n"
                "and paste it into the '.env' file in the root of your project: \n"
                "GROQ_API_KEY=gsk_..."
            )
        # Initialize the official Groq client with our key
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
    return _groq_client

def generate_completion(
    prompt: str, 
    system_prompt: str = "You are a helpful assistant.", 
    temperature: float = 0.1, 
    max_tokens: int = 2000
) -> str:
    """
    Sends a request to Groq API to generate a chat completion.
    
    Parameters:
    - prompt: The text prompt from the user or agent.
    - system_prompt: Guidance instructions for the model's behavior.
    - temperature: Controls randomness. Low value (0.1) is best for structured extraction.
    - max_tokens: Limits the generated output length to control costs and prevent errors.
    """
    client = get_groq_client()
    
    # We make a standard OpenAI-style chat completion call via Groq
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        model=config.GROQ_MODEL,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Extract and return the generated text
    return response.choices[0].message.content

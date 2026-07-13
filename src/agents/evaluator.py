import json
import logging
from typing import List, Dict, Any
from src import llm
from src.agents.map_maker import MapMakerAgent # For clean JSON parsing utilities

# Set up logging for validation/evaluation steps
logger = logging.getLogger("graph_rag.evaluator")

class EvaluatorAgent:
    """
    Agent responsible for evaluating the generated RAG answers against
    the retrieved source context, calculating a faithfulness score,
    and flagging any factual gaps (hallucinations).
    """
    def __init__(self):
        pass

    def evaluate_answer(self, query: str, answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates the answer against the retrieved sources.
        
        Returns a dictionary:
        {
          "faithfulness_score": float (0.0 - 1.0),
          "factual_gaps": List[str]
        }
        """
        logger.info("Evaluating generated answer for faithfulness...")

        if not sources:
            logger.warning("No sources provided for evaluation. Defaulting to score 0.0 due to lack of grounding context.")
            return {
                "faithfulness_score": 0.0,
                "factual_gaps": ["No grounding sources were retrieved from the database to verify this answer."]
            }

        # Format sources text chunks for the LLM
        sources_text = ""
        for src in sources:
            sources_text += f"- [{src['source_id']}]: {src['text_chunk']}\n"

        system_prompt = (
            "You are an expert RAG Evaluator. Your job is to check if a generated answer "
            "is fully grounded in and supported by a list of source context snippets. "
            "You must identify any claims in the answer that are not supported by the sources, "
            "and calculate a faithfulness score (1.0 = fully faithful, 0.0 = completely ungrounded)."
        )

        prompt = f"""
        User Query: {query}
        
        Generated Answer:
        ---
        {answer}
        ---

        Grounding Sources:
        ---
        {sources_text}
        ---

        Evaluate the Generated Answer against the Grounding Sources:
        1. Identify any statements or claims in the Answer that are NOT supported by, or contradict, the Grounding Sources. These are 'factual gaps'.
        2. Compute a 'faithfulness_score' between 0.0 and 1.0 (e.g., 0.9 if 90% of the key claims are grounded).
        
        Your output must follow this JSON schema:
        {{
          "faithfulness_score": 1.0,
          "factual_gaps": [
             "List specific unsupported claims, or leave empty if the answer is 100% faithful"
          ]
        }}
        
        Return ONLY valid JSON.
        """

        try:
            response_text = llm.generate_completion(prompt, system_prompt, temperature=0.1)
            
            # Clean and parse JSON using the helper method from MapMakerAgent
            # We can recreate a simple cleaning regex here to keep it independent
            cleaned_response = response_text.strip()
            if "```" in response_text:
                import re
                match = re.search(r"```(?:json)?\s*(.*?)\s*```", response_text, re.DOTALL)
                if match:
                    cleaned_response = match.group(1).strip()
            
            data = json.loads(cleaned_response)
            
            # Clamp the score between 0.0 and 1.0 to guarantee schema compliance
            score = float(data.get("faithfulness_score", 1.0))
            score = max(0.0, min(1.0, score))
            
            factual_gaps = data.get("factual_gaps", [])
            
            logger.info(f"Evaluation complete. Faithfulness score: {score}. Factual gaps found: {len(factual_gaps)}")
            return {
                "faithfulness_score": score,
                "factual_gaps": factual_gaps
            }

        except Exception as e:
            logger.error(f"Error during faithfulness evaluation LLM call: {e}")
            # Fallback in case of JSON parse failure or API issue
            return {
                "faithfulness_score": 0.5,
                "factual_gaps": [f"Evaluation failed to execute properly due to error: {e}"]
            }

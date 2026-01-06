"""
LLM Client for batch query processing using Ollama
"""
import requests
import json
import time
from config import OLLAMA_API, MODEL, CATEGORIES, MAX_RETRIES, REQUEST_TIMEOUT


def query_llm_batch(queries, max_retries=MAX_RETRIES):
    """
    Process multiple queries with STRICT formatting requirements
    Returns: JSON response string or None
    """
    queries_text = "\n".join([f"{i+1}. \"{q}\"" for i, q in enumerate(queries)])
    
    prompt = f"""Analyze these search queries. Return EXACTLY {len(queries)} JSON objects in an array.
CRITICAL: EVERY object MUST have all three fields: entities, categories, attributes.
Never skip any field. If uncertain, assign lower confidence values.

Queries:
{queries_text}

Return ONLY this exact format (no explanation):
[
  {{"entities": ["e1","e2"], "categories": {{"Category": 0.8}}, "attributes": {{"key": "val"}}}},
  {{"entities": ["e1"], "categories": {{"Category": 0.7}}, "attributes": {{}}}},
  ...
]

Available categories: {', '.join(CATEGORIES)}

YOU MUST RETURN EXACTLY {len(queries)} COMPLETE OBJECTS."""
    
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 200
        }
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(OLLAMA_API, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '')
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return None
    return None


def parse_batch_response(response, num_queries):
    """
    Parse batch JSON response from LLM
    Returns: List of parsed objects or None
    """
    try:
        data = json.loads(response)
        if isinstance(data, list) and len(data) == num_queries:
            return data
    except:
        pass
    
    # Try extracting JSON from markdown code blocks
    try:
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        elif "[" in response and "]" in response:
            start = response.find("[")
            end = response.rfind("]") + 1
            json_str = response[start:end]
        else:
            json_str = response.strip()
        
        data = json.loads(json_str)
        if isinstance(data, list):
            return data
    except:
        pass
    
    return None
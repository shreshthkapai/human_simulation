"""
LLM Client - Two-Phase System (LLM extracts + scores generically)
"""
import requests
import json
import time
from config import OLLAMA_API, MODEL, CATEGORIES, MAX_RETRIES, REQUEST_TIMEOUT


# Category definitions for better LLM understanding
CATEGORY_DEFINITIONS = {
    'Travel': 'trips, vacations, tourism, destinations, hotels, flights',
    'Work_Career': 'jobs, employment, career development, workplace, skills',
    'Daily_Life': 'routine activities, errands, household tasks, local services',
    'Life_Transitions': 'major life changes, moving, relationships, births, deaths',
    'Location': 'maps, directions, addresses, geographic information, places',
    'Entertainment': 'movies, music, games, hobbies, leisure activities',
    'Technology': 'gadgets, software, apps, electronics, tech products',
    'Fashion': 'clothing, style, shopping, fashion trends, apparel',
    'News_Politics': 'current events, politics, news articles, elections'
}


def query_llm_batch(queries, max_retries=MAX_RETRIES):
    """
    Process multiple queries with formula-guided scoring
    LLM outputs generic scores (same for all users)
    """
    queries_text = "\n".join([f"{i+1}. \"{q}\"" for i, q in enumerate(queries)])
    
    # Build category descriptions
    cat_descriptions = "\n".join([f"- {cat}: {desc}" for cat, desc in CATEGORY_DEFINITIONS.items()])
    
    prompt = f"""Analyze these search queries and score them against categories.

CATEGORIES:
{cat_descriptions}

SCORING FORMULA:
For each query, calculate category scores (0.0 to 1.0) as:
score = 0.5 × keyword_match + 0.3 × entity_evidence + 0.2 × semantic_fit

Where:
- keyword_match: Does query contain category-related terms? (1.0 if yes, 0.0 if no)
- entity_evidence: Do extracted entities belong to this category? (average fit)
- semantic_fit: Does overall query intent match this category?

EXAMPLES:
Query: "best hotels in paris"
  Entities: ["Paris", "hotels"]
  Travel: keyword=0.0, entity=0.9, semantic=1.0 → score=0.77
  Location: keyword=0.0, entity=0.7, semantic=0.6 → score=0.51

Query: "iphone 15 pro review"
  Entities: ["iPhone 15 Pro"]
  Technology: keyword=0.0, entity=1.0, semantic=1.0 → score=0.80

RULES:
- Extract specific entities (names, products, places)
- Only include categories with score > 0.3
- Scores should reflect generic understanding (not personalized)

Queries to analyze:
{queries_text}

Return EXACTLY {len(queries)} JSON objects in array format:
[
  {{"entities": ["e1","e2"], "categories": {{"Category": 0.8}}, "attributes": {{}}}},
  ...
]

Return only valid JSON array, no explanation."""
    
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 300
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
    """Parse batch JSON response from LLM"""
    try:
        data = json.loads(response)
        if isinstance(data, list) and len(data) == num_queries:
            return data
    except:
        pass
    
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
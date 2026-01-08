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
    
    prompt = f"""Analyze these {len(queries)} search queries. 
Return a JSON array of objects with "entities", "categories", and "attributes".

EXTRACTION RULES:
1. ENTITIES: Extract specific proper nouns, locations, brands, and products.
2. KEYWORDS: Also extract important descriptive keywords or topics that are NOT in the predefined category list.
   - Example: "gloucester to london" -> ["Gloucester", "London", "Train", "Route"]
   - Example: "kings remote access" -> ["Kings", "Remote Access", "VPN", "IT Support"]
   - Example: "calories in buttermilk burger" -> ["Buttermilk Burger", "Calories", "Nutrition", "Fast Food"]
3. CATEGORIES: Use ONLY the predefined list below. Assign scores (0.3-1.0).

PREDEFINED CATEGORIES:
{cat_descriptions}

QUERIES:
{queries_text}

OUTPUT FORMAT:
[
  {{"entities": ["Specific Entity", "Derived Keyword"], "categories": {{"CategoryName": 0.8}}, "attributes": {{}}}},
  ...
]
Return ONLY the JSON array. No explanation."""

    
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json", # Forces Ollama JSON mode
        "options": {
            "temperature": 0.1,
            "num_predict": 500
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
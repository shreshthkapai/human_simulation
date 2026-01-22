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

EXTRACTION RULES (SMART GRANULARITY):
1. EXTRACT meaningful concepts. An entity can be multiple words if it represents ONE specific thing (e.g., "Canary Wharf", "Women in Business").
2. SPLIT distinct ideas. Do not lump the "who", "what", and "where" into one string.
   - BAD: ["Blackstone's women networking event"]
   - GOOD: ["Blackstone", "Women in Business", "Networking Event"]
3. KEYWORDS: Extract descriptive topics or intent not in the category list.
   - Example: "gloucester to london train" -> ["Gloucester", "London", "Train Journey"]
   - Example: "kings remote access vpn" -> ["Kings", "VPN", "Remote Access"]

CATEGORIES:
Use ONLY the predefined list below.
{cat_descriptions}

QUERIES:
{queries_text}

OUTPUT FORMAT:
[
  {{"entities": ["Concept One", "Concept Two"], "categories": {{"CategoryName": 0.8}}, "attributes": {{}}}},
  ...
]
Return ONLY the JSON array. No explanation."""


    
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json", 
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


def validate_and_clean_item(item):
    """
    Validate and clean a single parsed item. Returns cleaned item or None if invalid.
    This allows per-item validation so good items aren't lost due to one bad item.
    """
    if not isinstance(item, dict):
        return None
    
    # Extract and validate entities (must be list of strings)
    entities = item.get('entities', [])
    if not isinstance(entities, list):
        entities = []
    # Filter to only valid strings
    entities = [e for e in entities if isinstance(e, str) and e.strip()]
    
    # Extract and validate categories (must be dict with float values)
    categories = item.get('categories', {})
    if not isinstance(categories, dict):
        categories = {}
    # Clean categories: keep only valid key-value pairs, convert values to float
    cleaned_categories = {}
    for k, v in categories.items():
        if isinstance(k, str) and v is not None:
            try:
                cleaned_categories[k] = float(v)
            except (ValueError, TypeError):
                continue  # Skip invalid values
    
    # Must have at least one valid category
    if not cleaned_categories:
        print(f"REJECTED: No valid categories found in {item.get('categories')}")
        return None
    
    # Extract attributes (optional, default to empty dict)
    attributes = item.get('attributes', {})
    if not isinstance(attributes, dict):
        attributes = {}
    
    return {
        'entities': entities,
        'categories': cleaned_categories,
        'attributes': attributes
    }


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
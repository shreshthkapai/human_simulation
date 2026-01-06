"""
Utility functions for data loading and query extraction
"""
import json
import pandas as pd
from config import SEARCH_HISTORY_FILE


def load_search_data(filepath=SEARCH_HISTORY_FILE):
    """Load search history from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_query(title):
    """Extract the actual search query from the title field"""
    if pd.isna(title):
        return None
    
    title = str(title)
    
    # Pattern 1: "Searched for X"
    if title.startswith("Searched for "):
        return title.replace("Searched for ", "").strip()
    
    # Pattern 2: "Visited [URL or page title]"
    elif title.startswith("Visited "):
        visited = title.replace("Visited ", "").strip()
        if visited.startswith("http"):
            return None
        return visited
    
    # Pattern 3: Direct query
    else:
        return title.strip()


def extract_location(row):
    """Extract location from locationInfos if available"""
    if 'locationInfos' in row and row['locationInfos']:
        try:
            loc_info = row['locationInfos'][0]
            if 'name' in loc_info:
                return loc_info['name']
        except:
            pass
    return None


def prepare_dataframe(search_data):
    """Convert raw search data to cleaned DataFrame"""
    df = pd.DataFrame(search_data)
    
    # Handle mixed timestamp formats
    df['timestamp'] = pd.to_datetime(df['time'], format='mixed', utc=True)
    df = df.sort_values('timestamp')
    df = df.reset_index(drop=True)
    
    # Extract queries
    df['query'] = df['title'].apply(extract_query)
    
    # Remove null queries
    df_clean = df[df['query'].notna()].copy()
    
    # Extract location info
    df_clean['location_info'] = df_clean.apply(extract_location, axis=1)
    
    # Temporal features
    df_clean['year'] = df_clean['timestamp'].dt.year
    df_clean['month'] = df_clean['timestamp'].dt.month
    df_clean['hour'] = df_clean['timestamp'].dt.hour
    df_clean['day_of_week'] = df_clean['timestamp'].dt.dayofweek
    df_clean['query_length'] = df_clean['query'].str.len()
    
    return df_clean


def print_data_summary(df, df_clean):
    """Print summary statistics about the data"""
    print(f"Original entries: {len(df):,}")
    print(f"After cleaning: {len(df_clean):,}")
    print(f"Removed: {len(df) - len(df_clean):,}")
    print(f"\nOldest search: {df_clean.iloc[0]['timestamp']}")
    print(f"Newest search: {df_clean.iloc[-1]['timestamp']}")
    print(f"Time span: {(df_clean.iloc[-1]['timestamp'] - df_clean.iloc[0]['timestamp']).days} days")
    
    # Location data
    with_location = df_clean['location_info'].notna().sum()
    print(f"\nSearches with location data: {with_location:,} ({with_location/len(df_clean)*100:.1f}%)")
    
    # Temporal distribution
    print(f"\nSearches by year:")
    print(df_clean['year'].value_counts().sort_index())
"""
Configuration file for Graph-Based Persona Generation
Contains all constants and model parameters

ADAPTIVE THRESHOLDS: Magic numbers replaced with statistical derivation
"""

# File paths
SEARCH_HISTORY_FILE = 'search_history.json'
CHECKPOINT_DIR = './'
FINAL_CHECKPOINT = 'checkpoint_FINAL.pkl'

# Ollama API Configuration
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"

# Category definitions
CATEGORIES = [
    "Fashion", 
    "Travel", 
    "Daily_Life", 
    "Life_Transitions", 
    "Location", 
    "Work_Career", 
    "Entertainment", 
    "Technology", 
    "News_Politics"
]

# Model parameters
BATCH_SIZE = 3
MAX_RETRIES = 2
REQUEST_TIMEOUT = 60

# Graph predictor parameters
NUM_WALKS = 30
WALK_LENGTH = 4
LEARNING_RATE = 0.1

# Transition detection parameters - NOW ADAPTIVE
TRANSITION_WINDOW_SIZE = 25

# REMOVED: SURPRISE_THRESHOLD = 4.0  (was magic number)
# Now calculated adaptively as: mean(surprise) + SURPRISE_THRESHOLD_SIGMA * std(surprise)
SURPRISE_THRESHOLD_SIGMA = 1.5  # Only remaining tunable parameter (interpretable: "1.5 std deviations above mean")

# Checkpoint parameters
CHECKPOINT_EVERY = 500

# Visualization parameters
SMOOTHING_WINDOW = 500
TOP_N_ENTITIES = 10
TOP_N_EVENTS = 5
FIGURE_DPI = 300

# Transition clustering - NOW ADAPTIVE
# REMOVED: MAX_GAP_DAYS = 14  (was magic number)
# REMOVED: MIN_CLUSTER_SIZE = 5  (was magic number)

# Adaptive clustering parameters (interpretable multipliers)
MAX_GAP_DAYS_MULTIPLIER = 1.0  # mean_gap + multiplier * std_gap
MIN_CLUSTER_SIZE_SIGMA = 0.5   # mean_size + sigma * std_size
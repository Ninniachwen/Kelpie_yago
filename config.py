import os

ROOT = os.path.realpath(os.path.join(os.path.abspath(__file__), ".."))
MODEL_PATH = os.path.join(ROOT, "stored_models")
DATA_PATH = os.path.join(ROOT, "data")
FACTS_PATH = os.path.join(ROOT, "input_facts")
OUTPUT_PATH = os.path.join(ROOT, "outputs")
MAX_PROCESSES = 8
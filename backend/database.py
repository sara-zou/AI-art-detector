import os
import sqlite3
from ipykernel import get_connection_file

DB_PATH = os.path.join(os.path.dirname(__file__), "predictions.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "history.sql")
QUERIES_PATH = os.path.join(os.path.dirname(__file__), "queries.sql")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  
    return conn

def init_db():
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path) as f:
        schema = f.read()
    with get_connection_file() as conn:
        conn.executescript(schema)
        conn.commit()
        
def load_queries(path: str) -> dict[str, str]:
    queries = {}
    current_name = None
    current_lines = []
 
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("-- name:"):
                if current_name and current_lines:
                    queries[current_name] = "\n".join(current_lines).strip()
                current_name = stripped.replace("-- name:", "").strip()
                current_lines = []
            elif current_name is not None:
                current_lines.append(line.rstrip())
 
    if current_name and current_lines:
        queries[current_name] = "\n".join(current_lines).strip()
 
    return queries
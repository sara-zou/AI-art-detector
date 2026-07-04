import os
import sqlite3
from ipykernel import get_connection_file

DB_PATH = os.path.join(os.path.dirname(__file__), "predictions.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "history.sql")

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
        
        
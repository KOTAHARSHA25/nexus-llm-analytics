import sqlite3
import os
from pathlib import Path

DB_PATH = Path("data/samples/test_db.sqlite")

def setup_db():
    if DB_PATH.exists():
        os.remove(DB_PATH)
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY,
            product TEXT,
            amount REAL,
            region TEXT
        )
    ''')
    
    # Insert data
    data = [
        (1, 'Widget A', 100.50, 'North'),
        (2, 'Widget B', 200.00, 'South'),
        (3, 'Widget A', 150.00, 'East'),
        (4, 'Widget C', 300.00, 'West'),
        (5, 'Widget B', 250.00, 'North')
    ]
    cursor.executemany('INSERT INTO sales VALUES (?, ?, ?, ?)', data)
    
    conn.commit()
    conn.close()
    print(f"Database created at {DB_PATH}")

if __name__ == "__main__":
    setup_db()

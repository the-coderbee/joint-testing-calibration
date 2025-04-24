"""Database utilities for calibration application."""
import sqlite3
from datetime import datetime


def get_db_connection():
    """Connect to SQLite database."""
    conn = sqlite3.connect('calibration.db')
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database with required tables."""
    conn = get_db_connection()
    conn.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        input_data TEXT NOT NULL,
        prediction REAL NOT NULL,
        actual_value REAL,
        sensor_id TEXT,
        equipment_id TEXT
    )
    ''')
    conn.commit()
    conn.close()


def save_prediction(input_data, prediction, actual_value=None, sensor_id=None, equipment_id=None):
    """Save prediction to database."""
    conn = get_db_connection()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    conn.execute(
        'INSERT INTO predictions (timestamp, input_data, prediction, actual_value, sensor_id, equipment_id) VALUES (?, ?, ?, ?, ?, ?)',
        (timestamp, str(input_data), float(prediction), actual_value, sensor_id, equipment_id)
    )
    conn.commit()
    conn.close()


def get_past_predictions(limit=100):
    """Get past predictions from database."""
    conn = get_db_connection()
    predictions = conn.execute(
        'SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?', 
        (limit,)
    ).fetchall()
    conn.close()
    return predictions 
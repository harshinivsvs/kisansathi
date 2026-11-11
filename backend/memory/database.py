"""
Database module for farmer profiles and conversational memory.
Uses SQLite for zero-dependency local development and production portability.
"""

from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

DB_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "kisansathi_memory.db"


class FarmerProfile(BaseModel):
    """Profile representing a registered farmer with location and agronomic context."""
    phone_number: str
    name: Optional[str] = None
    village: Optional[str] = None
    taluka: Optional[str] = None
    district: Optional[str] = "Nashik"
    state: str = "Maharashtra"
    latitude: Optional[float] = 19.9975  # Default Nashik
    longitude: Optional[float] = 73.7898
    location_consent_given: bool = False
    primary_crop: Optional[str] = None
    sowing_date: Optional[str] = None
    land_size_acres: Optional[float] = None
    preferred_language: str = "te"  # Configured regional default: Telugu
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class PriceWatchlist(BaseModel):
    """Mandi price target alert request."""
    id: Optional[int] = None
    phone_number: str
    crop: str
    target_price: float
    condition: str = "above"  # "above" or "below"
    active: bool = True
    created_at: float = Field(default_factory=time.time)


def init_db(db_path: Path = DB_PATH) -> None:
    """Initializes tables for farmer profiles, dialog turns, and price watchlists."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS farmers (
        phone_number TEXT PRIMARY KEY,
        name TEXT,
        village TEXT,
        taluka TEXT,
        district TEXT DEFAULT 'Nashik',
        state TEXT DEFAULT 'Maharashtra',
        latitude REAL DEFAULT 19.9975,
        longitude REAL DEFAULT 73.7898,
        location_consent_given INTEGER DEFAULT 0,
        primary_crop TEXT,
        sowing_date TEXT,
        land_size_acres REAL,
        preferred_language TEXT DEFAULT 'te',
        created_at REAL,
        updated_at REAL
    )
    """)

    # Safe migration: add any missing columns if table pre-existed
    cursor.execute("PRAGMA table_info(farmers)")
    existing_cols = {col[1] for col in cursor.fetchall()}
    columns_to_add = [
        ("taluka", "TEXT"),
        ("district", "TEXT DEFAULT 'Nashik'"),
        ("state", "TEXT DEFAULT 'Maharashtra'"),
        ("latitude", "REAL DEFAULT 19.9975"),
        ("longitude", "REAL DEFAULT 73.7898"),
        ("location_consent_given", "INTEGER DEFAULT 0"),
        ("land_size_acres", "REAL"),
        ("preferred_language", "TEXT DEFAULT 'te'")
    ]
    for col_name, col_type in columns_to_add:
        if col_name not in existing_cols:
            try:
                cursor.execute(f"ALTER TABLE farmers ADD COLUMN {col_name} {col_type}")
            except Exception:
                pass

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        phone_number TEXT,
        session_id TEXT,
        user_query TEXT,
        assistant_answer TEXT,
        language TEXT,
        crop_context TEXT,
        timestamp REAL,
        FOREIGN KEY (phone_number) REFERENCES farmers (phone_number)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS query_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        phone_number TEXT,
        query TEXT,
        answer TEXT,
        retrieved_chunks_json TEXT,
        citations_json TEXT,
        confidence REAL,
        latency_ms REAL,
        token_cost REAL,
        audio_url TEXT,
        language TEXT,
        timestamp REAL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS price_watchlists (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        phone_number TEXT,
        crop TEXT,
        target_price REAL,
        condition TEXT DEFAULT 'above',
        active INTEGER DEFAULT 1,
        created_at REAL,
        FOREIGN KEY (phone_number) REFERENCES farmers (phone_number)
    )
    """)

    conn.commit()
    conn.close()


init_db()

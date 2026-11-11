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
    """Profile representing a registered farmer."""
    phone_number: str
    name: Optional[str] = None
    village: Optional[str] = None
    district: Optional[str] = None
    state: str = "Maharashtra"
    primary_crop: Optional[str] = None
    sowing_date: Optional[str] = None
    preferred_language: str = "te"  # Configured regional default: Telugu
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


def init_db(db_path: Path = DB_PATH) -> None:
    """Initializes tables for farmer profiles, dialog turns, and query telemetry logs."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS farmers (
        phone_number TEXT PRIMARY KEY,
        name TEXT,
        village TEXT,
        district TEXT,
        state TEXT DEFAULT 'Maharashtra',
        primary_crop TEXT,
        sowing_date TEXT,
        preferred_language TEXT DEFAULT 'te',
        created_at REAL,
        updated_at REAL
    )
    """)

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

    conn.commit()
    conn.close()


# Auto-initialize database schema on import
init_db()

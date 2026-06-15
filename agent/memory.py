"""
memory.py — Session Persistence untuk Matcha
Menyimpan dan memuat state percakapan user menggunakan SQLite.
"""

import json
import os
import sqlite3
from typing import Optional, Dict, Any

# Deteksi otomatis database PostgreSQL atau SQLite
DB_URL = os.environ.get("DATABASE_URL") or os.environ.get("MATCHA_DB_PATH", "matcha_sessions.db")

def get_connection():
    if DB_URL.startswith("postgres://") or DB_URL.startswith("postgresql://"):
        import psycopg2
        return psycopg2.connect(DB_URL), True
    return sqlite3.connect(DB_URL), False


# Init Database

def init_db():
    """Buat tabel sessions jika belum ada."""
    conn, is_pg = get_connection()
    cursor = conn.cursor()
    if is_pg:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id VARCHAR(255) PRIMARY KEY,
                state_json TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    else:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                state_json  TEXT NOT NULL,
                updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
    conn.commit()
    conn.close()


# Save Session

def save_session(session_id: str, state: Dict[str, Any]):
    """
    Simpan state agent ke database.
    Hanya menyimpan field yang penting (bukan pesan chat — itu di session_state Streamlit/frontend).
    """
    fields_to_save = [
        "user_profile",
        "skill_gaps",
        "detected_intent",
        "previous_intent_history",
        "drift_detected",
        "cv_text",
        "linkedin_text",
        "job_description",
        "ats_analysis",
        "learning_roadmap",
        "cv_uploaded",
        "linkedin_uploaded",
        "cv_filename",
        "linkedin_filename",
    ]
    payload = {k: state.get(k) for k in fields_to_save}

    conn, is_pg = get_connection()
    cursor = conn.cursor()
    
    if is_pg:
        cursor.execute("""
            INSERT INTO sessions (session_id, state_json, updated_at)
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (session_id) DO UPDATE SET
                state_json = EXCLUDED.state_json,
                updated_at = CURRENT_TIMESTAMP
        """, (session_id, json.dumps(payload, ensure_ascii=False)))
    else:
        cursor.execute("""
            INSERT INTO sessions (session_id, state_json, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(session_id) DO UPDATE SET
                state_json = excluded.state_json,
                updated_at = CURRENT_TIMESTAMP
        """, (session_id, json.dumps(payload, ensure_ascii=False)))
    conn.commit()
    conn.close()


# Load Session

def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Muat state agent dari database berdasarkan session_id.
    Kembalikan dict kosong jika session tidak ditemukan.
    """
    conn, is_pg = get_connection()
    cursor = conn.cursor()
    if is_pg:
        cursor.execute(
            "SELECT state_json FROM sessions WHERE session_id = %s",
            (session_id,)
        )
    else:
        cursor.execute(
            "SELECT state_json FROM sessions WHERE session_id = ?",
            (session_id,)
        )
    row = cursor.fetchone()
    conn.close()

    if row:
        return json.loads(row[0])
    return {}
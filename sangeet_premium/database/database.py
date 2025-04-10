# ==============================================================================
# database.py
# ==============================================================================
# Description:
#   This module handles the initialization and schema definition for various
#   SQLite databases used by the Sangeet application. It defines the paths
#   to the database files and provides functions to create the necessary
#   tables if they don't exist.
#
# Databases Managed:
#   - Main Application Database (users, downloads, history, stats, sessions, OTPs)
#   - Playlists Database (playlists, playlist songs)
#   - History Backup Database (alternative history storage)
#   - Issues Database (user-reported issues and comments)
#   - Lyrics Cache Database (cached song lyrics)
# ==============================================================================

import sqlite3
import os
import logging

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ==============================================================================
# Database File Paths
# These paths are used throughout the application to access different SQLite
# database files. They are defined globally for easy access from other modules.
# ==============================================================================
DB_PATH = os.path.join(os.getcwd(), "database_files", "sangeet_database_main.db")
"""Path to the main application SQLite database file."""

PLAYLIST_DB_PATH = os.path.join(os.getcwd(), "database_files", "playlists.db")
"""Path to the playlists SQLite database file."""

HISTORY_BACKUP_DATABASE_STORAGE = os.path.join(os.getcwd(), "database_files", "history_backup.db")
"""Path to the backup history SQLite database file."""

issues_db = os.path.join(os.getcwd(), "database_files", "issues.db")
"""Path to the user issues SQLite database file."""

LYRICS_CACHE_DB_PATH = os.path.join(os.getcwd(), "database_files", "lyrics_cache.db")
"""Path to the lyrics cache SQLite database file."""


# ==============================================================================
# Database Initialization Functions
# ==============================================================================

# --- Initialize Issues Database ---
def init_issues_db():
    """
    Initializes the database for tracking user-reported issues and comments.

    Creates the 'user_issues' and 'issue_comments' tables if they do not
    already exist.
    """
    conn = None
    try:
        conn = sqlite3.connect(issues_db) # Use the global path variable
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_issues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            topic TEXT NOT NULL,
            details TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Open', -- e.g., Open, In Progress, Resolved, Closed
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS issue_comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            issue_id INTEGER NOT NULL,
            user_id INTEGER, -- Nullable for admin/system comments
            is_admin BOOLEAN DEFAULT 0,
            comment TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (issue_id) REFERENCES user_issues(id) ON DELETE CASCADE -- Ensure comments are deleted if issue is deleted
        )
        ''')
        conn.commit()
        logger.info("Issues database initialized successfully")
    except sqlite3.Error as e:
        logger.error(f"Error initializing issues database: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

# --- Initialize Lyrics Cache Database ---
def init_lyrics_db():
    """
    Initializes the SQLite database used for caching song lyrics.

    Creates the 'lyrics_cache' table if it does not already exist.
    This table stores song IDs, their lyrics, and a timestamp.
    """
    conn = None
    try:
        conn = sqlite3.connect(LYRICS_CACHE_DB_PATH) # Use the global path variable
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS lyrics_cache (
            song_id TEXT PRIMARY KEY,
            lyrics TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        conn.commit()
        logger.info("Lyrics cache database initialized successfully")
    except sqlite3.Error as e:
        logger.error(f"Error initializing lyrics cache database: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

# --- Initialize Main Application Database ---
def init_db():
    """
    Initializes the main application database (sangeet_database_main.db).

    Creates tables for:
    - users: Stores user account information.
    - user_downloads: Tracks songs downloaded by users.
    - listening_history: Records user listening activity.
    - user_stats: Stores aggregated user statistics (optional/legacy).
    - active_sessions: Manages active user login sessions.
    - pending_otps: Stores temporary OTPs for verification purposes.

    Also creates indexes on key columns for performance optimization.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # User Authentication Tables
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                totp_secret TEXT,
                twofa_method TEXT DEFAULT 'none', -- e.g., 'none', 'totp', 'email'
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # User-specific Downloads
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_downloads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                video_id TEXT NOT NULL,
                title TEXT,
                artist TEXT,
                album TEXT,
                path TEXT,
                downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # Listening History
        c.execute("""
            CREATE TABLE IF NOT EXISTS listening_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER, -- Can be NULL for anonymous history if needed
                song_id TEXT NOT NULL,
                title TEXT,
                artist TEXT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                duration INTEGER, -- Total duration of the song in seconds
                listened_duration INTEGER, -- Actual time listened in seconds
                completion_rate FLOAT, -- listened_duration / duration
                session_id TEXT, -- To group listens within a session
                listen_type TEXT CHECK(listen_type IN ('full', 'partial', 'skip')) DEFAULT 'partial'
            )
        """)

        # User Statistics (Optional - can be derived from history)
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_stats (
                user_id INTEGER PRIMARY KEY,
                total_plays INTEGER DEFAULT 0,
                total_listened_time INTEGER DEFAULT 0, -- in seconds
                favorite_song_id TEXT,
                favorite_artist TEXT,
                last_played TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # Session Management
        c.execute("""
            CREATE TABLE IF NOT EXISTS active_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_token TEXT UNIQUE NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # OTP Management
        c.execute("""
            CREATE TABLE IF NOT EXISTS pending_otps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                otp TEXT NOT NULL,
                purpose TEXT NOT NULL, -- e.g., 'registration', 'password_reset', '2fa'
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL
            )
        """)

        # Create indexes for performance
        c.execute("CREATE INDEX IF NOT EXISTS idx_user_downloads_user ON user_downloads(user_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_listening_history_user ON listening_history(user_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_listening_dates ON listening_history(started_at)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_listening_song ON listening_history(song_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_listening_completion ON listening_history(completion_rate)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_active_sessions_token ON active_sessions(session_token)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_pending_otps_email_purpose ON pending_otps(email, purpose)")


        conn.commit()
        logger.info("Main database initialized successfully")

    except sqlite3.Error as e:
        logger.error(f"Main database initialization error: {e}")
        if conn:
            conn.rollback()
        raise # Re-raise the exception after logging and rollback
    finally:
        if conn:
            conn.close()

# --- Initialize Playlist Database ---
def init_playlist_db():
    """
    Initializes the database for storing user playlists (playlists.db).

    Creates tables for:
    - playlists: Stores information about each playlist (name, owner, public status).
    - playlist_songs: Maps songs to playlists.
    """
    conn = None
    try:
        conn = sqlite3.connect(PLAYLIST_DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS playlists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER, -- Associated user, can be NULL for global/default playlists if needed
            name TEXT NOT NULL,
            is_public INTEGER DEFAULT 0, -- Boolean (0 = False, 1 = True)
            share_id TEXT UNIQUE, -- Optional unique ID for sharing public playlists
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS playlist_songs (
            playlist_id INTEGER,
            song_id TEXT, -- Identifier for the song (e.g., video_id or internal ID)
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (playlist_id, song_id), -- Prevent duplicate songs in a playlist
            FOREIGN KEY (playlist_id) REFERENCES playlists(id) ON DELETE CASCADE -- If playlist is deleted, remove its songs
        )''')
        # Index for faster lookup of songs in a playlist
        c.execute("CREATE INDEX IF NOT EXISTS idx_playlist_songs_playlist ON playlist_songs(playlist_id)")
        conn.commit()
        logger.info("Playlist database initialized successfully")
    except sqlite3.Error as e:
        logger.error(f"Playlist database initialization error: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

# --- Initialize Backup History Database ---
def init_backup_his_db():
    """
    Initializes the backup database for listening history (history_backup.db).

    Creates the 'history' table if it does not already exist. This serves as
    a potential alternative or redundant storage for listening history records.
    """
    conn = None
    try:
        conn = sqlite3.connect(HISTORY_BACKUP_DATABASE_STORAGE)
        c = conn.cursor()
        # Note: Consider aligning this schema more closely with 'listening_history'
        # in the main DB if it's meant as a direct backup/replacement.
        c.execute('''CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            song_id TEXT,
            played_at TEXT, -- Consider using TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            session_id TEXT,
            sequence_number INTEGER, -- Useful for ordering within a session
            thumbnail TEXT -- URL or path to thumbnail
        );''')
        # Add indexes if needed for querying this backup table
        c.execute("CREATE INDEX IF NOT EXISTS idx_history_backup_user ON history(user_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_history_backup_song ON history(song_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_history_backup_played_at ON history(played_at)")

        conn.commit()
        logger.info("Backup history database initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Backup history database initialization error: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

# Example of how you might call these (e.g., in your main application startup)
# if __name__ == "__main__":
#     print("Initializing all databases...")
#     os.makedirs("database_files", exist_ok=True) # Ensure directory exists
#     init_db()
#     init_playlist_db()
#     init_lyrics_db()
#     init_issues_db()
#     init_backup_his_db()
#     print("Database initialization complete.")
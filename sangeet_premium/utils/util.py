# ==============================================================================
# util.py - Sangeet Premium Utility Functions
# ==============================================================================
# Description:
#   This module provides a collection of utility functions used throughout the
#   Sangeet Premium application. It includes helpers for:
#   - Database interactions (main DB, local songs DB, OTPs, sessions).
#   - Redis operations (caching, history, stats).
#   - File operations (downloading, path handling, metadata).
#   - External service interactions (YTMusic API, yt-dlp, SMTP).
#   - String manipulation and validation.
#   - Authentication and session management helpers.
#   - Data processing for insights and statistics.
#   - Setup for dependencies like yt-dlp.
# ==============================================================================

# --- Standard Library Imports ---
import base64
import json
import logging
import os
import platform
import random
import re
import secrets
import smtplib
import sqlite3
import stat
import subprocess
import time
from datetime import datetime, timedelta, timezone # Added timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional # Added Optional and Tuple

# --- External Library Imports ---
import requests
import redis
from dotenv import load_dotenv
from flask import jsonify, redirect, session # Keep Flask imports minimal if possible
from mutagen import File
from mutagen.flac import FLAC
from yt_dlp import YoutubeDL
from ytmusicapi import YTMusic
import pytz # Added pytz for timezone awareness

# --- Internal Application Imports ---
try:
    # Attempt to import time_helper from the parent directory's helpers module
    from ..helpers import time_helper
    time_sync = time_helper.TimeSync()
    logger.info("Successfully imported and initialized time_helper.TimeSync.")
except (ImportError, ModuleNotFoundError):
    logger.warning("time_helper module not found or failed to import. Using basic datetime functions.")
    # Define a fallback class if time_helper is unavailable
    class FallbackTimeSync:
        def get_current_time(self):
            # Return timezone-aware UTC time as a fallback standard
            return datetime.now(pytz.utc)
        def parse_datetime(self, time_str: str) -> Optional[datetime]:
            # Attempt to parse ISO format, assuming UTC if naive
            try:
                 dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                 if dt.tzinfo is None:
                      dt = pytz.utc.localize(dt) # Assume UTC if naive
                 return dt
            except (ValueError, TypeError):
                 # Try common formats as fallback
                 for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S.%fZ'):
                      try:
                           dt = datetime.strptime(time_str, fmt)
                           return pytz.utc.localize(dt) # Assume UTC
                      except (ValueError, TypeError):
                           continue
                 logger.error(f"Could not parse datetime string: {time_str}")
                 return None
        def format_time(self, dt: Optional[datetime], relative: bool = False) -> str:
            if not isinstance(dt, datetime):
                return "Invalid Date"
            # Ensure dt is timezone-aware (assume UTC if naive)
            if dt.tzinfo is None:
                 dt = pytz.utc.localize(dt)

            if relative:
                now = datetime.now(pytz.utc)
                diff = now - dt
                seconds = diff.total_seconds()
                if seconds < 0: return "in the future"
                if seconds < 60: return "just now"
                if seconds < 3600: return f"{int(seconds // 60)}m ago"
                if seconds < 86400: return f"{int(seconds // 3600)}h ago"
                return f"{int(seconds // 86400)}d ago"
            # Return in ISO format with UTC offset
            return dt.strftime('%Y-%m-%d %H:%M:%S %Z') # Example: 2023-10-27 10:30:00 UTC
    time_sync = FallbackTimeSync()


# ==============================================================================
# Configuration & Global Variables
# ==============================================================================

# --- Load Environment Variables ---
dotenv_path = os.path.join(os.getcwd(), "config", ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    logger.info(f"Loaded environment variables from: {dotenv_path}")
else:
    logger.warning(f".env file not found at: {dotenv_path}. Relying on system environment variables.")

# --- Database Paths ---
DB_PATH = os.path.join(os.getcwd(), "database_files", "sangeet_database_main.db")
DB_PATH_LOCAL_SONGS = os.path.join(os.getcwd(), "database_files", "local_songs.db") # Define path for local songs DB

# --- Directory Paths ---
# Music Directory (Case-insensitive check)
_music_path_env =  os.getenv("MUSIC_PATH")
if not _music_path_env:
    logger.warning("Environment variable 'music_path' or 'MUSIC_PATH' not set. Defaulting to './music'.")
    MUSIC_DIR = os.path.join(os.getcwd(), "music")
    os.makedirs(MUSIC_DIR, exist_ok=True) # Ensure default exists
else:
    MUSIC_DIR = _music_path_env
logger.info(f"Music directory set to: {MUSIC_DIR}")

# FFmpeg Directory (Used primarily on Windows)
FFMPEG_BIN_DIR = os.path.join(os.getcwd(), "ffmpeg", "bin")

# Local Songs Scan Paths (Case-insensitive check)
LOCAL_SONGS_PATHS = os.getenv("LOCAL_SONGS_PATHS", "")
if not LOCAL_SONGS_PATHS:
     logger.info("LOCAL_SONGS_PATHS environment variable not set. Local song scanning will be disabled.")
else:
     logger.info(f"Local song paths configured: {LOCAL_SONGS_PATHS}")


# --- SMTP Configuration (Case-insensitive check) ---
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
_smtp_port_str = os.getenv("SMTP_PORT", "587") 
try:
    SMTP_PORT = int(_smtp_port_str)
except ValueError:
    logger.warning(f"Invalid SMTP_PORT value '{_smtp_port_str}'. Using default 587.")
    SMTP_PORT = 587
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
if not SMTP_USER or not SMTP_PASSWORD:
     logger.warning("SMTP credentials (SMTP_USER, SMTP_PASSWORD) not fully configured. Email functionality may be disabled.")

# --- Caching ---
CACHE_DURATION = 3600 # Cache duration in seconds (1 hour)

# --- Service Clients & Globals ---
# Redis Client
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    logger.info("Redis client initialized and connected.")
except redis.exceptions.ConnectionError as e:
    logger.error(f"Failed to connect to Redis: {e}. Redis-dependent features will be unavailable.")
    redis_client = None

# YTMusic Client
try:
    ytmusic = YTMusic()
    logger.info("YTMusic API client initialized.")
except Exception as e:
     logger.error(f"Failed to initialize YTMusic API client: {e}")
     ytmusic = None # Allow app to run but YTMusic features will fail

# Logging Setup
logger = logging.getLogger(__name__) # Get logger for this module
# BasicConfig is often set in the main app entry point, avoid configuring multiple times if possible.
# logging.basicConfig(level=logging.INFO) # Keep if this is the primary config point

# In-memory Caches (Consider if these are still needed with Redis/SQLite caching)
song_cache = {} # Cache for ytmusic.get_song results
search_cache = {} # Cache for ytmusic.search results
lyrics_cache = {} # Note: SQLite is used for lyrics, is this dict needed? Seems unused.
local_songs = {} # Populated by load_local_songs()

# ==============================================================================
# yt-dlp Setup
# ==============================================================================

# --- Function: setup_ytdlp ---
def setup_ytdlp() -> Tuple[Optional[str], Optional[str]]:
    """
    Checks for, downloads (if necessary), and sets permissions for the yt-dlp executable.
    Determines the correct binary based on OS and architecture.

    Returns:
        tuple: (path_to_executable: str | None, path_to_version_file: str | None)
               Returns (None, None) on failure.
    """
    try:
        system = platform.system().lower()
        machine = platform.machine().lower()
        logger.info(f"Setting up yt-dlp for System: {system}, Architecture: {machine}")

        # Define download patterns based on platform and architecture
        # Ref: https://github.com/yt-dlp/yt-dlp#release-files
        download_patterns = {
            'windows': 'yt-dlp.exe',
            'darwin': 'yt-dlp_macos', # Assuming x86_64/arm64 macos binary works for both
            'linux': {
                'aarch64': 'yt-dlp_linux_aarch64',
                'armv7l': 'yt-dlp_linux_armv7l',
                'armv6l': 'yt-dlp_linux_armv7l', # Fallback for armv6
                'x86_64': 'yt-dlp_linux',       # Standard 64-bit Linux
                'amd64': 'yt-dlp_linux',        # Alias for x86_64
                'i386': 'yt-dlp_linux_x86',     # 32-bit Linux
                'i686': 'yt-dlp_linux_x86'      # Alias for 32-bit
            }
        }

        if system == 'linux':
            download_pattern = download_patterns.get(system, {}).get(machine)
            if not download_pattern:
                 # Fallback for generic Linux if specific arch not found? Risky.
                 logger.warning(f"Unsupported Linux architecture '{machine}'. Trying generic 'yt-dlp_linux'.")
                 download_pattern = 'yt-dlp_linux' # Attempt generic Linux binary
        elif system in download_patterns:
            download_pattern = download_patterns[system]
        else:
            raise Exception(f"Unsupported operating system: {system}")

        logger.info(f"Determined yt-dlp download pattern: {download_pattern}")

        # Define directory and file paths within 'res' folder
        res_dir = Path('res') / system / machine
        res_dir.mkdir(parents=True, exist_ok=True)

        executable_name = "yt-dlp.exe" if system == "windows" else "yt-dlp"
        executable_path = res_dir / executable_name
        version_path = res_dir / "version.txt"

        # --- Check latest version from GitHub API ---
        latest_version = None
        try:
            api_url = "https://api.github.com/repos/yt-dlp/yt-dlp/releases/latest"
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            release_data = response.json()
            latest_version = release_data.get('tag_name')
            if not latest_version:
                 raise ValueError("Could not parse 'tag_name' from GitHub API response.")
            logger.info(f"Latest yt-dlp version available: {latest_version}")
        except requests.exceptions.RequestException as e:
             logger.error(f"Failed to fetch latest yt-dlp version from GitHub: {e}")
             # Proceed without version check if API fails? Or abort? Abort safer.
             raise Exception(f"GitHub API request failed: {e}")
        except (KeyError, ValueError, json.JSONDecodeError) as e:
             logger.error(f"Failed to parse GitHub API response for yt-dlp version: {e}")
             raise Exception(f"GitHub API response parsing failed: {e}")
        # --- End Version Check ---


        # --- Compare with local version and decide whether to download ---
        should_download = True
        if executable_path.exists(): # Check if executable exists first
             if version_path.exists():
                  try:
                       current_version = version_path.read_text(encoding='utf-8').strip()
                       if current_version == latest_version:
                            logger.info(f"Current yt-dlp version ({current_version}) matches latest. Skipping download.")
                            should_download = False
                       else:
                            logger.info(f"yt-dlp update found: Current {current_version}, Latest {latest_version}. Downloading.")
                  except Exception as e:
                       logger.warning(f"Could not read local yt-dlp version file ({version_path}): {e}. Will re-download.")
             else:
                  logger.warning(f"yt-dlp executable exists but version file is missing. Re-downloading.")
        else:
             logger.info(f"yt-dlp executable not found at {executable_path}. Downloading.")
        # --- End Comparison ---

        # --- Download if necessary ---
        if should_download:
            # Find the correct download URL from release assets
            download_url = None
            for asset in release_data.get('assets', []):
                if asset.get('name') == download_pattern:
                    download_url = asset.get('browser_download_url')
                    logger.info(f"Found download URL: {download_url}")
                    break

            if not download_url:
                available_assets = [asset.get('name') for asset in release_data.get('assets', [])]
                logger.error(f"Could not find asset matching pattern '{download_pattern}' in release.")
                logger.error(f"Available assets: {available_assets}")
                raise Exception(f"Required yt-dlp binary '{download_pattern}' not found in latest release.")

            # Download the binary
            try:
                logger.info(f"Downloading yt-dlp from {download_url}...")
                download_response = requests.get(download_url, stream=True, timeout=60) # Use stream=True for potentially large files
                download_response.raise_for_status()
                with open(executable_path, 'wb') as f:
                    for chunk in download_response.iter_content(chunk_size=8192):
                         f.write(chunk)
                logger.info(f"Download complete. Saved to {executable_path}")
            except requests.exceptions.RequestException as e:
                 logger.error(f"Failed to download yt-dlp binary: {e}")
                 # Clean up potentially partial download?
                 if executable_path.exists(): executable_path.unlink(missing_ok=True)
                 raise Exception(f"yt-dlp download failed: {e}")

            # Save the new version information
            try:
                version_path.write_text(latest_version, encoding='utf-8')
                logger.info(f"Updated version file ({version_path}) to {latest_version}")
            except Exception as e:
                 logger.error(f"Failed to write yt-dlp version file: {e}")
                 # Proceed, but version check might fail next time

            # Set executable permissions on non-Windows systems
            if system != "windows":
                try:
                    current_permissions = executable_path.stat().st_mode
                    executable_path.chmod(current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH) # Add execute permissions for all
                    logger.info(f"Set execute permissions for {executable_path}")
                except Exception as e:
                    logger.error(f"Failed to set execute permissions on {executable_path}: {e}")
                    # This might cause issues later when trying to run it

        # Return paths if setup was successful (or skipped because up-to-date)
        if executable_path.exists():
             return str(executable_path), str(version_path)
        else:
             # Should not happen if logic is correct, but as a safeguard
             raise Exception("yt-dlp setup finished but executable path does not exist.")

    except Exception as e:
        logger.exception(f"Fatal error during yt-dlp setup: {e}") # Use logger.exception to include traceback
        return None, None

# --- Initialize yt-dlp Path ---
# Run setup and store the path globally
YTDLP_PATH, YTDLP_VERSION_PATH = setup_ytdlp()
if not YTDLP_PATH:
     logger.critical("yt-dlp executable setup failed. Download functionality will be severely limited or unavailable.")
     # Application might need to exit or disable features here depending on requirements


# ==============================================================================
# Authentication & Session Utilities
# ==============================================================================

# --- Function: generate_session_id ---
def generate_session_id() -> str:
    """Generates a unique ID for grouping related listening events."""
    # Combines timestamp with random hex for better uniqueness
    return f"session_{int(time.time())}_{secrets.token_hex(4)}"

# --- Decorator: login_required ---
# Note: This decorator relies on Flask's `session` object and `redirect`.
# It might be better placed in the routes file (`playback.py`) if only used there.
# Kept here as per original structure, but be mindful of dependency.
def login_required(f):
    """Decorator to ensure a user is logged in via Flask session."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or 'session_token' not in session:
            logger.debug(f"Access denied for {request.path}: No user session found.")
            # For API requests, return JSON error; otherwise, redirect to login
            if request.path.startswith('/api/'):
                 return jsonify({"error": "Authentication required"}), 401
            return redirect(url_for('playback.login', next=request.url)) # Added next URL

        # --- Verify session validity in database ---
        # Optimization: Consider checking less frequently if performance is an issue.
        conn = None
        is_valid = False
        try:
            conn = sqlite3.connect(DB_PATH) # Use global DB_PATH
            c = conn.cursor()
            c.execute("""
                SELECT 1 FROM active_sessions
                WHERE user_id = ? AND session_token = ?
                AND expires_at > CURRENT_TIMESTAMP
            """, (session['user_id'], session['session_token']))
            is_valid = c.fetchone() is not None
        except sqlite3.Error as e:
            logger.error(f"Database error verifying session for user {session.get('user_id')}: {e}")
            # Treat DB error as invalid session? Or allow access but log? Safer to deny.
            is_valid = False
        finally:
            if conn: conn.close()

        if not is_valid:
            logger.warning(f"Invalid or expired session detected for user {session.get('user_id')}. Clearing session.")
            session.clear() # Clear the invalid Flask session
            if request.path.startswith('/api/'):
                 return jsonify({"error": "Session expired or invalid"}), 401
            return redirect(url_for('playback.login'))
        # --- End session verification ---

        # Session is valid, proceed with the original function
        return f(*args, **kwargs)
    return decorated_function

# --- Function: cleanup_expired_sessions ---
def cleanup_expired_sessions():
    """Removes expired session records from the active_sessions table."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # Use CURRENT_TIMESTAMP which is timezone-aware in SQLite if stored correctly
        c.execute("DELETE FROM active_sessions WHERE expires_at <= CURRENT_TIMESTAMP")
        deleted_count = c.rowcount
        conn.commit()
        if deleted_count > 0:
             logger.info(f"Cleaned up {deleted_count} expired sessions from database.")
    except sqlite3.Error as e:
        logger.error(f"Database error during expired session cleanup: {e}")
        if conn: conn.rollback()
    finally:
        if conn: conn.close()


# ==============================================================================
# History & Statistics Recording Utilities (Redis)
# ==============================================================================

# --- Function: record_song ---
@login_required # Requires user to be logged in
def record_song(song_id: str, user_id: int, client_timestamp: Optional[str] = None):
    """
    Records a song play event in Redis history for the specified user.
    Handles session continuity and updates basic play stats.
    Includes thumbnail URL in the history entry.
    """
    if not redis_client:
        logger.error(f"Cannot record song play for user {user_id}: Redis client unavailable.")
        return # Or raise an exception if this is critical

    try:
        # Determine timestamp: Use client's if valid, else server's current time
        played_at_dt = None
        if client_timestamp:
            parsed_client_dt = time_sync.parse_datetime(client_timestamp)
            if parsed_client_dt:
                 played_at_dt = parsed_client_dt
            else:
                 logger.warning(f"Could not parse client timestamp '{client_timestamp}'. Using server time.")

        if played_at_dt is None:
             played_at_dt = time_sync.get_current_time() # Get timezone-aware time

        # Format for storage (consistent format)
        played_at_str = played_at_dt.strftime('%Y-%m-%d %H:%M:%S') # Store as naive string for compatibility? Or ISO UTC? Using simple format now.

        history_key = f"user_history:{user_id}"
        session_id = None
        sequence_number = 1

        # --- Session Continuity Logic ---
        # Check the latest entry in the history list
        last_entry_raw = redis_client.lindex(history_key, 0)
        if last_entry_raw:
            try:
                last_data = json.loads(last_entry_raw)
                last_played_at_str = last_data.get('played_at')
                if last_played_at_str:
                     last_played_dt = time_sync.parse_datetime(last_played_at_str)
                     # Check if within continuity threshold (e.g., 1 hour)
                     if last_played_dt and (played_at_dt - last_played_dt).total_seconds() < 3600:
                          session_id = last_data.get('session_id')
                          sequence_number = last_data.get('sequence_number', 0) + 1
                     # else: time gap too large, start new session
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                 logger.warning(f"Error parsing last history entry for session continuity (user {user_id}): {e}")
                 # Start new session if parsing fails

        # If no continuing session found, generate a new one
        if session_id is None:
            session_id = generate_session_id()
            sequence_number = 1
        # --- End Session Logic ---

        # --- Fetch Thumbnail (from Redis cache or fallback) ---
        thumbnail_url = ""
        try:
             # Try getting thumbnail from song metadata hash in Redis
             # Using song_id directly assumes it's the key for metadata hash
             thumb_from_redis = redis_client.hget(song_id, "thumbnail")
             if thumb_from_redis:
                  thumbnail_url = thumb_from_redis
             else:
                  # Fallback to standard YouTube thumbnail URL format if not local
                  if not song_id.startswith("local-") and is_potential_video_id(song_id):
                       thumbnail_url = f"https://i.ytimg.com/vi/{song_id}/hqdefault.jpg"
                  # else: Leave empty if local and no thumb found, or ID invalid
        except redis.exceptions.RedisError as e:
             logger.error(f"Redis error fetching thumbnail for history record {song_id}: {e}")
             # Use fallback if Redis fails
             if not song_id.startswith("local-") and is_potential_video_id(song_id):
                  thumbnail_url = f"https://i.ytimg.com/vi/{song_id}/hqdefault.jpg"
        # --- End Thumbnail Fetch ---


        # Create the history entry dictionary
        history_entry = {
            "song_id": song_id,
            "played_at": played_at_str,
            "session_id": session_id,
            "sequence_number": sequence_number,
            "thumbnail": thumbnail_url # Include the thumbnail
        }

        # --- Store in Redis ---
        # Add to the beginning of the list (LPUSH)
        redis_client.lpush(history_key, json.dumps(history_entry))
        # Trim the list to maintain a maximum size (e.g., 1000 entries)
        redis_client.ltrim(history_key, 0, 999)
        # --- End Redis Store ---

        # --- Update Basic Stats (Optional - can be derived later) ---
        # stats_key = f"user_stats:{user_id}"
        # redis_client.hincrby(stats_key, "total_plays", 1)
        # redis_client.hset(stats_key, "last_played", played_at_str)
        # --- End Stats Update ---

        logger.debug(f"Recorded play: User {user_id}, Song {song_id}, Session {session_id}, Seq {sequence_number}")

    except redis.exceptions.RedisError as e:
        logger.error(f"Redis error recording song play for user {user_id}: {e}")
        # Decide if this should raise an exception to the caller
    except Exception as e:
        logger.exception(f"Unexpected error recording song play for user {user_id}, song {song_id}: {e}")
        # Decide if this should raise


# --- Function: record_download (Redis based) ---
@login_required
def record_download(video_id: str, title: str, artist: str, album: str, path: str, user_id: int):
    """
    Records metadata about a downloaded file in Redis.
    Uses a hash per download and adds ID to a user-specific set.
    """
    if not redis_client:
        logger.error(f"Cannot record download for user {user_id}: Redis client unavailable.")
        return

    download_key = f"download:{user_id}:{video_id}"
    user_set_key = f"user_downloads:{user_id}"
    download_data = {
        "video_id": video_id, # Include for clarity
        "title": title or "Unknown Title",
        "artist": artist or "Unknown Artist",
        "album": album or "Unknown Album",
        "path": path,
        "downloaded_at": datetime.now(pytz.utc).isoformat() # Use ISO format with UTC
    }

    try:
        # Use a pipeline for atomicity (though less critical here than multi-key ops)
        pipe = redis_client.pipeline()
        pipe.hset(download_key, mapping=download_data) # Store download details
        pipe.sadd(user_set_key, video_id) # Add video ID to user's download set
        pipe.execute()
        logger.info(f"Recorded download in Redis: User {user_id}, Video {video_id}")
    except redis.exceptions.RedisError as e:
        logger.error(f"Redis error recording download for user {user_id}, video {video_id}: {e}")


# ==============================================================================
# Data Retrieval Utilities
# ==============================================================================

# --- Function: get_play_history (Redis based) ---
@login_required
def get_play_history(user_id: int, limit: int = 5, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Retrieves paginated play history for a user from Redis.
    Fetches song metadata efficiently (from Redis cache or API).
    """
    if not redis_client:
        logger.error(f"Cannot get play history for user {user_id}: Redis client unavailable.")
        return []

    history_key = f"user_history:{user_id}"
    history_list = []
    song_ids_to_fetch_meta = []
    raw_history_entries = []

    try:
        # Fetch paginated JSON strings from Redis list
        start = offset
        end = offset + limit - 1
        raw_history_entries = redis_client.lrange(history_key, start, end)
    except redis.exceptions.RedisError as e:
        logger.error(f"Redis error fetching raw history for user {user_id}: {e}")
        return [] # Return empty on Redis error

    # --- Prepare initial history list and identify songs needing metadata ---
    temp_history_map = {} # Use dict for easier update: {song_id: [list of history entries for this song]}
    song_id_order = [] # Maintain original order

    for entry_raw in raw_history_entries:
        try:
            data = json.loads(entry_raw)
            song_id = data.get('song_id')
            if not song_id: continue

            # Store base data
            history_item = {
                "id": song_id,
                "played_at_str": data.get('played_at'), # Keep original string for now
                "session_id": data.get('session_id'),
                "sequence_number": data.get('sequence_number'),
                "thumbnail_from_history": data.get('thumbnail') # Get thumb stored in history
            }

            if song_id not in temp_history_map:
                 temp_history_map[song_id] = []
                 song_id_order.append(song_id) # Add to order list
                 song_ids_to_fetch_meta.append(song_id) # Mark for metadata fetch

            temp_history_map[song_id].append(history_item)

        except json.JSONDecodeError:
            logger.warning(f"Skipping invalid JSON history entry for user {user_id}")

    # --- Batch Fetch Metadata ---
    metadata_cache = {} # {song_id: metadata_dict}
    if song_ids_to_fetch_meta:
         # Try fetching metadata from Redis hashes first (assuming metadata stored per song_id)
         try:
              pipe = redis_client.pipeline()
              for sid in song_ids_to_fetch_meta:
                   # Skip local songs for Redis metadata check? Or assume they might be there too?
                   # Assuming metadata hash key is just the song_id
                   pipe.hgetall(sid)
              redis_results = pipe.execute()

              missing_meta_ids = []
              for sid, meta_dict in zip(song_ids_to_fetch_meta, redis_results):
                   if meta_dict and all(k in meta_dict for k in ['title', 'artist', 'duration']): # Basic check
                        # Convert duration back to int
                        try: meta_dict['duration'] = int(meta_dict['duration'])
                        except: meta_dict['duration'] = 0
                        metadata_cache[sid] = meta_dict
                   else:
                        # If not in Redis or incomplete, mark for API/local fetch
                        missing_meta_ids.append(sid)
         except redis.exceptions.RedisError as e:
              logger.error(f"Redis error during batch metadata fetch for history (user {user_id}): {e}")
              missing_meta_ids = song_ids_to_fetch_meta # Fetch all if Redis fails

         # Fetch remaining metadata using get_media_info (local or API)
         for sid in missing_meta_ids:
              try:
                   # get_media_info uses its own cache (song_cache dict)
                   media_info = get_media_info(sid)
                   if media_info and media_info.get('title') != 'Unknown Title':
                        metadata_cache[sid] = media_info
                        # Optionally update Redis cache here?
                        # if redis_client and not sid.startswith("local-"):
                        #    try: redis_client.hset(sid, mapping=media_info); redis_client.expire(sid, CACHE_DURATION*24)
                        #    except: pass
                   else:
                        logger.warning(f"Could not retrieve valid metadata for history song: {sid}")
                        # Store placeholder to avoid refetching constantly?
                        metadata_cache[sid] = {"title": "Info Unavailable", "artist": "", "duration": 0, "thumbnail": util.get_default_thumbnail()}
              except Exception as e:
                   logger.error(f"Error fetching metadata for history song {sid}: {e}")
                   metadata_cache[sid] = {"title": "Error Loading Info", "artist": "", "duration": 0, "thumbnail": util.get_default_thumbnail()}

    # --- Combine history entries with metadata in original order ---
    final_history_list = []
    processed_indices = set() # To handle pagination correctly if multiple entries have same song_id

    for i, entry_raw in enumerate(raw_history_entries):
         if i in processed_indices: continue # Skip if already processed via temp_history_map logic

         try:
              data = json.loads(entry_raw)
              song_id = data.get('song_id')
              if not song_id: continue

              meta = metadata_cache.get(song_id, {"title": "Metadata Error", "artist": "", "duration": 0, "thumbnail": util.get_default_thumbnail()})

              # Get timestamp and format it
              played_at_dt = time_sync.parse_datetime(data.get('played_at', ''))
              played_at_formatted = time_sync.format_time(played_at_dt) if played_at_dt else "Invalid Time"
              played_at_relative = time_sync.format_time(played_at_dt, relative=True) if played_at_dt else ""

              # Prefer thumbnail from history record if available, else from metadata
              thumbnail = data.get('thumbnail') or meta.get('thumbnail', util.get_default_thumbnail())


              final_history_list.append({
                   "id": song_id,
                   "title": meta.get('title'),
                   "artist": meta.get('artist'),
                   "album": meta.get('album', ''),
                   "duration": meta.get('duration'),
                   "thumbnail": thumbnail,
                   "played_at": played_at_formatted,
                   "played_at_relative": played_at_relative,
                   "session_id": data.get('session_id'),
                   "sequence_number": data.get('sequence_number')
              })
              processed_indices.add(i) # Mark as processed

         except json.JSONDecodeError:
              # Already warned earlier, just skip here
              pass
         except Exception as e:
              logger.exception(f"Error combining metadata for history entry (user {user_id}): {e}")


    return final_history_list


# --- Function: get_download_info ---
# @login_required # This depends if only logged-in users can trigger downloads checked elsewhere
def get_download_info(video_id: str) -> Optional[str]:
    """
    Checks if a video has been downloaded (by any user, based on file existence).
    Returns the file path if it exists, otherwise None.
    Note: This does NOT check user-specific download records from `record_download`.
    """
    # Construct the expected path using the configured MUSIC_DIR
    flac_path = os.path.join(MUSIC_DIR, f"{video_id}.flac")

    if os.path.exists(flac_path):
        logger.debug(f"Download check: File exists for {video_id} at {flac_path}")
        return flac_path
    else:
        logger.debug(f"Download check: File does not exist for {video_id} at {flac_path}")
        return None

# --- Function: get_download_info_for_user (More specific) ---
@login_required
def get_download_info_for_user(video_id: str, user_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieves download information for a specific user and video ID from Redis.
    Returns the download metadata dictionary if found, otherwise None.
    """
    if not redis_client:
        logger.error("Redis client unavailable for get_download_info_for_user.")
        return None

    download_key = f"download:{user_id}:{video_id}"
    try:
        download_data = redis_client.hgetall(download_key)
        if download_data and download_data.get("path"):
            # Optionally check os.path.exists(download_data["path"]) here?
            return download_data
        return None
    except redis.exceptions.RedisError as e:
        logger.error(f"Redis error getting download info for user {user_id}, video {video_id}: {e}")
        return None


# ==============================================================================
# Local Music File Handling
# ==============================================================================

# --- Function: init_db_local (SQLite based - seems unused if Redis is primary) ---
# Commenting out as Redis seems to be the primary mechanism now for local songs.
# If SQLite for local songs is still needed, uncomment and adjust.
# def init_db_local():
#     """Initialize SQLite database for local song metadata."""
#     # ... (original implementation using DB_PATH_LOCAL_SONGS) ...
#     pass

# --- Function: get_new_local_id (SQLite based - seems unused) ---
# Commenting out as Redis ID generation is handled differently.
# def get_new_local_id(cursor):
#     """Generate a new 'local-X' ID based on existing SQLite entries."""
#     # ... (original implementation) ...
#     pass

# --- Function: load_local_songs (Redis based) ---
def load_local_songs() -> Dict[str, Dict]:
    """
    Scans configured local directories for music files, extracts metadata,
    updates Redis records, and populates the global `local_songs` dictionary.
    Cleans up Redis entries for files that no longer exist.
    """
    global local_songs # Declare intention to modify global variable
    local_songs = {} # Reset global cache at start of scan

    if not LOCAL_SONGS_PATHS:
        logger.info("Local song scanning skipped: LOCAL_SONGS_PATHS not configured.")
        return local_songs
    if not redis_client:
        logger.error("Local song scanning failed: Redis client unavailable.")
        return local_songs

    # --- Prepare for Scan ---
    try:
        dirs_to_scan = [os.path.abspath(d.strip()) for d in LOCAL_SONGS_PATHS.split(";") if d.strip()]
        logger.info(f"Starting local song scan in directories: {dirs_to_scan}")
        supported_extensions = {".mp3", ".flac", ".m4a", ".wav", ".ogg", ".opus", ".aac"} # Add/remove as needed
        current_file_paths_on_disk = set() # Keep track of files found in this scan
        processed_paths_in_redis = set(redis_client.hkeys("path_to_id")) # Paths currently known to Redis
    except redis.exceptions.RedisError as e:
        logger.error(f"Redis error preparing for local scan: {e}")
        return local_songs
    except Exception as e:
        logger.exception(f"Error preparing for local scan: {e}")
        return local_songs


    # --- Scan Directories ---
    for target_dir in dirs_to_scan:
        if not os.path.isdir(target_dir):
            logger.warning(f"Configured local path is not a valid directory, skipping: {target_dir}")
            continue

        for root, _, files in os.walk(target_dir):
            for filename in files:
                file_path = os.path.abspath(os.path.join(root, filename))
                _, ext = os.path.splitext(filename)

                if ext.lower() in supported_extensions:
                    current_file_paths_on_disk.add(file_path) # Add found file to set
                    # Check if Redis already knows this path
                    song_id_from_redis = redis_client.hget("path_to_id", file_path)

                    # --- Get or Generate Song ID ---
                    if song_id_from_redis:
                        song_id = song_id_from_redis
                        # Verify the main hash for this ID exists, maybe update?
                        if not redis_client.exists(song_id):
                             logger.warning(f"Path '{file_path}' mapped to non-existent song ID '{song_id}' in Redis. Re-processing.")
                             song_id = None # Force re-generation/re-processing
                        # else: Optionally check if metadata needs update based on file mod time?
                    else:
                        song_id = None # Needs processing

                    if song_id is None:
                         # Generate a likely unique ID based on filename (avoid simple 'local-N')
                         base_name, _ = os.path.splitext(filename)
                         # Sanitize basename slightly for ID use
                         temp_id = re.sub(r'[^a-zA-Z0-9_.-]', '_', base_name)[:50] # Limit length
                         unique_id_candidate = f"local_{temp_id}"
                         # Ensure uniqueness (append suffix if needed)
                         suffix = 0
                         while redis_client.exists(unique_id_candidate): # Check if ID hash exists
                              suffix += 1
                              unique_id_candidate = f"local_{temp_id}_{suffix}"
                         song_id = unique_id_candidate
                         logger.info(f"Assigning new ID '{song_id}' to local file: {file_path}")
                         # Add mapping to Redis
                         try:
                              redis_client.hset("path_to_id", file_path, song_id)
                              redis_client.sadd("local_songs_ids", song_id) # Add to master set of local IDs
                         except redis.exceptions.RedisError as e:
                              logger.error(f"Redis error adding new local song mapping ({song_id}, {file_path}): {e}")
                              continue # Skip processing this file on error


                    # --- Extract Metadata using Mutagen ---
                    metadata = {
                         "id": song_id,
                         "path": file_path,
                         "title": os.path.splitext(filename)[0], # Default to filename base
                         "artist": "Unknown Artist",
                         "album": "Unknown Album",
                         "duration": 0,
                         "thumbnail": "" # Base64 encoded thumbnail
                    }
                    try:
                        audio = File(file_path) # Try loading with mutagen base class
                        if audio:
                             if hasattr(audio, "info") and audio.info and hasattr(audio.info, "length"):
                                  metadata["duration"] = int(audio.info.length)

                             # Use easy tags if available
                             if hasattr(audio, 'tags') and hasattr(audio.tags, 'get'):
                                  metadata["title"] = audio.tags.get('title', [metadata["title"]])[0]
                                  metadata["artist"] = audio.tags.get('artist', [metadata["artist"]])[0]
                                  metadata["album"] = audio.tags.get('album', [metadata["album"]])[0]
                                  # Add more tags if needed: genre, tracknumber, date etc.
                                  # metadata["genre"] = audio.tags.get('genre', [''])[0]

                             # --- Extract Thumbnail ---
                             # Try standard picture extraction
                             pictures = getattr(audio, "pictures", [])
                             if not pictures and hasattr(audio, 'tags'): # Fallback for some formats (e.g., ID3 APIC)
                                  if "APIC:" in audio.tags: pictures = [audio.tags["APIC:"]]
                                  elif "covr" in audio.tags: pictures = audio.tags["covr"] # MP4 cover art

                             if pictures:
                                  # Select the first picture (often front cover)
                                  pic = pictures[0]
                                  if hasattr(pic, 'data') and hasattr(pic, 'mime'):
                                       try:
                                            base64_thumb = base64.b64encode(pic.data).decode('ascii')
                                            metadata["thumbnail"] = f"data:{pic.mime};base64,{base64_thumb}"
                                       except Exception as thumb_err:
                                            logger.warning(f"Error encoding thumbnail for {file_path}: {thumb_err}")
                             # --- End Thumbnail ---
                        else:
                             logger.warning(f"Mutagen could not load audio info for: {file_path}")

                    except Exception as meta_err:
                        logger.error(f"Error reading metadata from {file_path} using Mutagen: {meta_err}")
                        # Use defaults set above

                    # --- Store/Update Metadata in Redis Hash ---
                    try:
                        # Convert duration back to string for Redis storage
                        metadata_to_store = metadata.copy()
                        metadata_to_store["duration"] = str(metadata["duration"])
                        redis_client.hset(song_id, mapping=metadata_to_store)
                        # No expiry needed for local song metadata unless files change often

                        # Add to the in-memory cache for current run
                        local_songs[song_id] = metadata # Use dict with int duration

                    except redis.exceptions.RedisError as e:
                         logger.error(f"Redis error storing metadata for {song_id}: {e}")
                         # Remove from in-memory cache if Redis fails?
                         if song_id in local_songs: del local_songs[song_id]

    # --- Cleanup Stale Redis Entries ---
    stale_paths = processed_paths_in_redis - current_file_paths_on_disk
    if stale_paths:
         logger.info(f"Found {len(stale_paths)} stale local file paths in Redis. Cleaning up...")
         stale_song_ids = []
         try:
              # Find song IDs associated with stale paths
              if stale_paths: # Ensure not empty list for hdel/hmget
                   stale_song_ids_raw = redis_client.hmget("path_to_id", list(stale_paths))
                   stale_song_ids = [sid for sid in stale_song_ids_raw if sid] # Filter out None results

              # Remove mappings and metadata
              pipe = redis_client.pipeline()
              if stale_paths:
                  pipe.hdel("path_to_id", *stale_paths)
              if stale_song_ids:
                  pipe.delete(*stale_song_ids) # Delete the song metadata hashes
                  pipe.srem("local_songs_ids", *stale_song_ids) # Remove from the master set
              pipe.execute()
              logger.info(f"Cleanup complete for {len(stale_paths)} paths and {len(stale_song_ids)} song IDs.")

              # Remove from in-memory cache as well
              for sid in stale_song_ids:
                    if sid in local_songs:
                         del local_songs[sid]

         except redis.exceptions.RedisError as e:
              logger.error(f"Redis error during stale local song cleanup: {e}")


    logger.info(f"Local song scan complete. In-memory cache size: {len(local_songs)}")
    return local_songs


# --- Function: filter_local_songs ---
def filter_local_songs(query: str) -> List[Dict]:
    """Filters the globally loaded `local_songs` based on a query string."""
    # Ensure local_songs is loaded - consider calling load_local_songs() if needed,
    # but usually called elsewhere before search.
    if not local_songs:
         logger.debug("filter_local_songs called but local_songs cache is empty.")
         return []

    q_lower = query.lower()
    results = []
    # Iterate through the values (metadata dicts) of the global cache
    for song_meta in local_songs.values():
        # Check if query matches title, artist, or album (case-insensitive)
        try:
             if (q_lower in song_meta.get('title', '').lower() or
                 q_lower in song_meta.get('artist', '').lower() or
                 q_lower in song_meta.get('album', '').lower()):
                 results.append(song_meta)
        except AttributeError as e:
             # Handle cases where metadata values might not be strings
             logger.warning(f"AttributeError during local song filter for song ID {song_meta.get('id')}: {e}")
             continue
        except Exception as e:
             logger.exception(f"Unexpected error filtering local song ID {song_meta.get('id')}: {e}")
             continue

    # De-duplication based on title/artist is handled in the main search route usually.
    # This function just returns all matches found in the local cache.
    return results

# ==============================================================================
# YTMusic API Interaction Utilities
# ==============================================================================

# --- Function: get_song_info ---
def get_song_info(song_id: str):
    """Wrapper for ytmusic.get_song with basic error handling and JSON response."""
    if not ytmusic:
         return jsonify({"error": "Music service unavailable"}), 503
    if not is_potential_video_id(song_id): # Validate ID format
         return jsonify({"error": "Invalid song ID format"}), 400

    # Check cache first (using the module-level song_cache dict)
    if song_id in song_cache:
         # Add expiry check to dict cache? Optional.
         logger.debug(f"Returning cached song info for {song_id}")
         return jsonify(song_cache[song_id])

    try:
        info = ytmusic.get_song(song_id)
        if not info:
            # Song ID might be invalid or song removed
            logger.warning(f"ytmusic.get_song returned no info for {song_id}")
            return jsonify({"error": "Song not found or unavailable"}), 404

        vd = info.get("videoDetails", {})
        title = vd.get("title", "Unknown Title")
        artist_name = "Unknown Artist"
        if info.get("artists") and isinstance(info["artists"], list):
             artist_name = info["artists"][0].get("name", artist_name)
        elif vd.get("author"):
             artist_name = vd["author"]

        thumbnail_url = get_best_thumbnail(vd.get("thumbnail", {}).get("thumbnails", [])) \
                        or f"https://i.ytimg.com/vi/{song_id}/hqdefault.jpg"

        result_data = {
            "id": song_id,
            "title": title,
            "artist": artist_name,
            "album": info.get("album", {}).get("name", ""),
            "thumbnail": thumbnail_url,
            "duration": safe_int(vd.get("lengthSeconds"))
        }

        # Update cache
        song_cache[song_id] = result_data

        return jsonify(result_data)

    except Exception as e:
        logger.error(f"Error calling ytmusic.get_song for {song_id}: {e}", exc_info=True)
        # Return generic error, avoid exposing detailed API errors
        return jsonify({"error": "Failed to retrieve song information from music service."}), 500


# --- Function: search_songs ---
@lru_cache(maxsize=100) # Keep LRU cache for repeated searches
def search_songs(query: str, limit: int = 20) -> List[Dict]:
    """
    Performs a song search using YTMusic API. Results are cached.
    Handles potential errors and standardizes the output format.
    """
    if not ytmusic:
         logger.error("YTMusic client unavailable for search.")
         return []
    if not query:
         return []

    # LRU cache handles time-based expiry implicitly (by replacing oldest entry)
    # No need for manual timestamp check here.

    try:
        logger.debug(f"Performing YTMusic search for: '{query}' (limit: {limit})")
        # Use 'songs' filter for more relevant results
        raw_results = ytmusic.search(query, filter="songs", limit=limit)
        if not raw_results:
             return []

        processed_results = []
        seen_combinations = set() # To avoid exact title/artist duplicates

        for item in raw_results:
            if item.get('resultType') != 'song' or not item.get('videoId'):
                continue # Skip non-song results or those missing ID

            video_id = item['videoId']
            title = item.get('title', 'Unknown Title')
            artist = "Unknown Artist"
            if item.get('artists') and isinstance(item['artists'], list):
                artist = item['artists'][0].get('name', artist)

            # Basic deduplication based on Title + Artist (case-insensitive)
            combination_key = (title.lower(), artist.lower())
            if combination_key in seen_combinations:
                 continue
            seen_combinations.add(combination_key)

            # Extract other details
            album = item.get('album', {}).get('name', '') if item.get('album') else ''
            duration_str = item.get('duration') # e.g., "3:15"
            duration_sec = parse_duration(duration_str) # Use helper to convert
            thumbnail = get_best_thumbnail(item.get('thumbnails', [])) \
                        or f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg" # Fallback

            processed_results.append({
                "id": video_id,
                "title": title,
                "artist": artist,
                "album": album,
                "duration": duration_sec,
                "thumbnail": thumbnail
            })

        return processed_results

    except Exception as e:
        logger.error(f"Error during YTMusic search for '{query}': {e}", exc_info=True)
        return [] # Return empty list on error


# ==============================================================================
# Recommendation Utilities
# ==============================================================================

# --- Function: get_recommendations_for_song ---
def get_recommendations_for_song(song_id: str, limit: int = 5) -> List[Dict]:
    """
    Gets recommendations based on a song_id. Tries watch playlist first,
    then artist songs, then search. Handles local songs via search.
    """
    if not ytmusic:
         logger.error("YTMusic client unavailable for recommendations.")
         return []
    if not song_id:
         return []

    recommendations = []
    seen_ids = {song_id} # Start with the original song ID to avoid recommending it

    # --- Handle Local Songs: Use Search ---
    if song_id.startswith("local-"):
        logger.info(f"Getting recommendations for local song {song_id} via search.")
        load_local_songs_from_file() # Ensure local meta is loaded
        local_meta = local_songs.get(song_id)
        if local_meta and local_meta.get('title') and local_meta.get('artist'):
             search_query = f"{local_meta['title']} {local_meta['artist']}"
             try:
                  search_results = ytmusic.search(search_query, filter="songs", limit=limit + 5)
                  for track in search_results:
                       if add_recommendation(track, recommendations, seen_ids, current_song_id=song_id):
                            if len(recommendations) >= limit: break
             except Exception as e:
                  logger.error(f"Search-based recommendation failed for local song {song_id}: {e}")
        # Proceed to fallback if search fails or yields too few results


    # --- Handle YouTube Songs: Multiple Strategies ---
    else:
         if not is_potential_video_id(song_id):
              logger.warning(f"Invalid song ID format for recommendations: {song_id}")
              return []

         song_info = None # Store fetched song info for reuse

         # Strategy 1: Watch Playlist (Radio) - Most relevant
         logger.debug(f"Trying watch playlist recommendations for {song_id}")
         try:
              watch_playlist = ytmusic.get_watch_playlist(videoId=song_id, limit=limit + 10) # Fetch more initially
              if watch_playlist and "tracks" in watch_playlist:
                   for track in watch_playlist["tracks"]:
                        if add_recommendation(track, recommendations, seen_ids, current_song_id=song_id):
                             if len(recommendations) >= limit: break
              if len(recommendations) >= limit:
                   logger.info(f"Got {len(recommendations)} recommendations from watch playlist.")
                   random.shuffle(recommendations)
                   return recommendations[:limit]
              else:
                   # Fetch song info now if needed for next steps and not already fetched
                   if not song_info: song_info = ytmusic.get_song(song_id)
         except Exception as e:
              logger.warning(f"Error getting watch playlist for {song_id}: {e}")
              if not song_info: # Ensure song_info is fetched if watch playlist failed
                   try: song_info = ytmusic.get_song(song_id)
                   except: logger.error(f"Failed to get song info for {song_id} after watch playlist error.")


         # Strategy 2: Artist's Other Songs (if watch playlist insufficient)
         if len(recommendations) < limit and song_info and song_info.get("artists"):
              logger.debug(f"Trying artist recommendations for {song_id}")
              try:
                   artist_id = song_info["artists"][0].get("id")
                   if artist_id:
                        artist_data = ytmusic.get_artist(artist_id)
                        if artist_data and "songs" in artist_data and artist_data["songs"]:
                             # Shuffle artist songs for variety
                             artist_songs = list(artist_data["songs"])
                             random.shuffle(artist_songs)
                             for track in artist_songs: # Iterate through shuffled list
                                  if add_recommendation(track, recommendations, seen_ids, current_song_id=song_id):
                                       if len(recommendations) >= limit: break
              except Exception as e:
                   logger.warning(f"Error getting artist recommendations for {song_id}: {e}")

         # Strategy 3: Search Based on Title/Artist (if still insufficient)
         if len(recommendations) < limit and song_info:
              logger.debug(f"Trying search-based recommendations for {song_id}")
              try:
                   title = song_info.get("videoDetails", {}).get("title", "")
                   artist = ""
                   if song_info.get("artists"): artist = song_info["artists"][0].get("name", "")
                   elif song_info.get("videoDetails"): artist = song_info["videoDetails"].get("author", "")

                   if title and artist:
                        search_query = f"{title} {artist}"
                        search_results = ytmusic.search(search_query, filter="songs", limit=limit + 5)
                        for track in search_results:
                             if add_recommendation(track, recommendations, seen_ids, current_song_id=song_id):
                                  if len(recommendations) >= limit: break
              except Exception as e:
                   logger.warning(f"Error getting search-based recommendations for {song_id}: {e}")


    # --- Fallback Strategy (if needed) ---
    if len(recommendations) < limit:
         logger.info(f"Insufficient recommendations ({len(recommendations)}/{limit}) for {song_id}. Using fallback.")
         try:
              fallback_recs = get_fallback_tracks(seen_ids) # Use helper function
              for track in fallback_recs:
                    # Need to ensure fallback tracks are in the right format for add_recommendation
                    # Or adjust add_recommendation to handle different structures if needed.
                    # Assuming get_fallback_tracks returns items compatible with add_recommendation
                    if add_recommendation(track, recommendations, seen_ids, current_song_id=song_id):
                         if len(recommendations) >= limit: break
         except Exception as e:
              logger.error(f"Error during fallback recommendation fetching: {e}")


    # Final shuffle and limit
    random.shuffle(recommendations)
    return recommendations[:limit]


# --- Function: get_fallback_recommendations ---
def get_fallback_recommendations() -> List[Dict]:
    """Provides a list of fallback recommendations using general search terms."""
    if not ytmusic:
         logger.error("YTMusic client unavailable for fallback recommendations.")
         return []

    logger.info("Fetching fallback recommendations using popular/trending searches.")
    fallback_queries = ["Top Hits Global", "Trending Music Now", "Popular Songs This Week"]
    recommendations = []
    seen_ids = set()

    try:
        # Try one random query first
        query = random.choice(fallback_queries)
        results = ytmusic.search(query, filter="songs", limit=10)
        if results:
             for track in results:
                  if add_recommendation(track, recommendations, seen_ids):
                       if len(recommendations) >= 5: break # Aim for 5 fallbacks

        # If still not enough, try another query if available
        if len(recommendations) < 5 and len(fallback_queries) > 1:
             fallback_queries.remove(query) # Don't repeat
             query = random.choice(fallback_queries)
             results = ytmusic.search(query, filter="songs", limit=10)
             if results:
                  for track in results:
                       if add_recommendation(track, recommendations, seen_ids):
                            if len(recommendations) >= 5: break

        random.shuffle(recommendations)
        logger.info(f"Returning {len(recommendations[:5])} fallback recommendations.")
        return recommendations[:5] # Return up to 5

    except Exception as e:
        logger.error(f"Error fetching fallback recommendations: {e}", exc_info=True)
        return [] # Return empty list on error

# --- Function: get_local_song_recommendations (Simplified) ---
def get_local_song_recommendations(local_song_id: str) -> List[Dict]:
    """Gets recommendations for local songs primarily via search."""
    # This can likely be merged into get_recommendations_for_song which handles local prefix
    logger.debug(f"Getting recommendations for local song {local_song_id} (using main recommender).")
    return get_recommendations_for_song(local_song_id, limit=5)


# --- Function: add_recommendation ---
def add_recommendation(track_data: Dict, recommendations_list: List, seen_ids_set: set, current_song_id: Optional[str] = None) -> bool:
    """
    Validates and processes a track dictionary, adding it to the list if valid and unseen.
    Returns True if added, False otherwise.
    """
    try:
        video_id = track_data.get("videoId")
        # Basic checks for essential data
        if not video_id or video_id == current_song_id or video_id in seen_ids_set:
            return False
        if track_data.get("isAvailable") is False or track_data.get("isPrivate") is True:
             return False

        title = track_data.get("title", "").strip()
        if not title or title == "Unknown Title":
            return False

        artist = "Unknown Artist"
        # Handle different possible artist structures
        if track_data.get('artists') and isinstance(track_data['artists'], list):
            artist = track_data['artists'][0].get('name', artist).strip()
        elif track_data.get('artist') and isinstance(track_data['artist'], str): # Simple string artist?
             artist = track_data['artist'].strip()
        elif track_data.get('author') and isinstance(track_data['author'], str): # Fallback to author
             artist = track_data['author'].strip()

        # Skip if artist is still unknown or seems invalid
        if not artist or artist == "Unknown Artist":
             # Allow if title seems descriptive enough? Maybe not for recommendations.
             return False

        album = ""
        if track_data.get('album') and isinstance(track_data['album'], dict):
            album = track_data['album'].get('name', "").strip()

        thumbnail = get_best_thumbnail(track_data.get('thumbnails', [])) \
                    or f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"

        # Get duration - check 'duration_seconds' first, then 'duration' string
        duration_sec = 0
        if 'duration_seconds' in track_data:
             duration_sec = safe_int(track_data['duration_seconds'])
        elif 'duration' in track_data:
             duration_sec = parse_duration(track_data['duration'])

        # Filter out extremely short/long tracks (e.g., < 30s or > 30min)
        if not (30 <= duration_sec <= 1800):
            # logger.debug(f"Skipping recommendation {video_id} due to duration: {duration_sec}s")
            return False

        # Add the processed track to the list
        recommendations_list.append({
            "id": video_id,
            "title": title,
            "artist": artist,
            "album": album,
            "thumbnail": thumbnail,
            "duration": duration_sec
        })
        seen_ids_set.add(video_id)
        return True

    except Exception as e:
        # Log error but don't crash the recommendation process
        logger.warning(f"Error processing recommendation track data: {e} - Data: {track_data.get('videoId', 'N/A')}")
        return False


# --- Function: get_fallback_tracks (Likely Redundant) ---
# This seems very similar to get_fallback_recommendations. Consolidate if possible.
# Commenting out for now.
# def get_fallback_tracks(seen_songs):
#    """Get fallback tracks from trending/popular songs or charts."""
#    # ... (implementation similar to get_fallback_recommendations) ...
#    pass

# ==============================================================================
# String & Data Processing Utilities
# ==============================================================================

# --- Function: is_potential_video_id ---
def is_potential_video_id(text: Optional[str]) -> bool:
    """Checks if a string matches the typical format of a YouTube video ID."""
    if not text:
        return False
    # YouTube video IDs are typically 11 characters long
    # and consist of letters (upper/lower), numbers, underscore, and hyphen.
    return bool(re.fullmatch(r'[a-zA-Z0-9_-]{11}', text))


# --- Function: sanitize_filename ---
def sanitize_filename(filename: Optional[str]) -> str:
    """Removes or replaces characters invalid in filenames across common OS."""
    if not filename:
        return 'unknown_filename'

    # Remove characters invalid in Windows/Linux/MacOS filenames
    # Chars: < > : " / \ | ? * and control characters (0-31)
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '', filename)

    # Replace multiple spaces/dots with single underscore
    sanitized = re.sub(r'[\s.]+', '_', sanitized)

    # Limit length (e.g., 150 chars) to avoid issues with path limits
    max_len = 150
    if len(sanitized) > max_len:
        # Try to keep extension if present
        base, ext = os.path.splitext(sanitized)
        base = base[:max_len - len(ext)]
        sanitized = base + ext
    sanitized = sanitized.strip('_') # Remove leading/trailing underscores

    # Ensure filename is not empty after sanitization
    return sanitized if sanitized else 'sanitized_filename'


# --- Function: process_description ---
def process_description(description: Optional[Any]) -> str:
    """Cleans and formats an artist description string."""
    if isinstance(description, list):
        # Join list elements if description is accidentally a list
        description = ' '.join(map(str, description))
    elif not isinstance(description, str):
        description = str(description or '') # Convert to string, default to empty

    # Basic cleanup: remove extra whitespace
    cleaned_description = ' '.join(description.split())

    return cleaned_description if cleaned_description else 'No description available.'

# --- Function: get_best_thumbnail (already defined earlier) ---
# Re-defined earlier, ensure only one definition exists.

# --- Function: process_genres ---
def process_genres(artist_data: Dict) -> List[str]:
    """Extracts and cleans a list of genres from artist data."""
    try:
        genres_raw = artist_data.get('genres')
        if isinstance(genres_raw, list):
            # Filter out empty strings and strip whitespace
            return [genre.strip() for genre in genres_raw if isinstance(genre, str) and genre.strip()]
        elif isinstance(genres_raw, str) and genres_raw.strip():
            # Handle case where it might be a single string
            return [genres_raw.strip()]
        else:
            return [] # Return empty list if no valid genres found
    except Exception as e:
        logger.warning(f"Error processing genres: {e}")
        return [] # Return empty list on error


# --- Function: get_artist_stats ---
# @login_required # This shouldn't require login, it processes data
def get_artist_stats(artist_data: Dict) -> Dict[str, str]:
    """Extracts and formats key statistics for an artist."""
    stats = {
        'subscribers': 'N/A',
        'views': 'N/A',
        'monthlyListeners': 'N/A'
    }
    try:
        # Subscribers (prefer 'subscribers' field, fallback to header/button text parsing if needed)
        sub_count = artist_data.get('subscribers')
        # Add more complex parsing from subscriptionButton/header if 'subscribers' field is unreliable
        stats['subscribers'] = safe_format_count(sub_count) if sub_count else 'N/A'

        # Views (usually available)
        view_count = artist_data.get('views')
        stats['views'] = safe_format_count(view_count) if view_count else 'N/A'

        # Monthly Listeners (requires helper due to inconsistency)
        listeners_count = get_monthly_listeners(artist_data) # Use helper function
        stats['monthlyListeners'] = safe_format_count(listeners_count) if listeners_count else 'N/A'

        # Optionally add other stats if YTMusic API provides them consistently
        # e.g., total plays if available in artist_data['stats']

    except Exception as e:
        logger.warning(f"Error processing artist stats: {e}")
        # Return defaults with 'N/A' on error
    return stats

# --- Function: get_monthly_listeners ---
def get_monthly_listeners(artist_data: Dict) -> Optional[str]:
    """Helper to extract monthly listeners count from potentially inconsistent fields."""
    try:
        # Prioritize specific fields if they exist
        for key in ['listeners', 'monthlyListeners']: # Check common keys
             val = artist_data.get(key)
             if val: return str(val) # Return the first non-empty value found

        # Fallback: Check 'description' text (less reliable)
        desc = artist_data.get('description', '')
        if isinstance(desc, str):
             # Look for patterns like "X listeners", "X monthly listeners"
             match = re.search(r'([\d,.]+[KMB]?)\s+(?:monthly\s+)?listeners', desc, re.IGNORECASE)
             if match: return match.group(1)

        # Fallback: Check subscriber count text if available and seems numeric
        sub_text = artist_data.get('subscribers') # Use the direct field if available
        if not sub_text and 'subscriptionButton' in artist_data: # Check button text
             sub_button = artist_data['subscriptionButton']
             if isinstance(sub_button, dict): sub_text = sub_button.get('title', {}).get('label') # Example path

        if isinstance(sub_text, str) and re.match(r'^[\d,.]+[KMB]?$', sub_text.replace(' subscribers','').strip(), re.IGNORECASE):
             # If subscriber text looks like a number (e.g., "1.2M subscribers"), use it as proxy
             return sub_text.replace(' subscribers','').strip()

        return None # Return None if no reliable count found
    except Exception as e:
        logger.warning(f"Error extracting monthly listeners: {e}")
        return None


# --- Function: process_top_songs ---
def process_top_songs(artist_data: Dict, limit: int = 10) -> List[Dict]:
    """Extracts and formats a list of top songs for an artist."""
    top_songs_list = []
    try:
        # YTMusic API structure for artist songs can vary, adjust path as needed
        songs_section = artist_data.get('songs') # Might be directly under artist, or nested
        if isinstance(songs_section, dict):
             # Common structure: songs_section['browseEndpoint'] might exist, or items directly
             song_items = songs_section.get('results', []) # Check for 'results' list
        elif isinstance(songs_section, list):
             song_items = songs_section # Assume it's directly a list of songs
        else:
             song_items = []

        if not song_items:
             logger.debug(f"No song items found in artist data for processing top songs.")
             return []

        for song_data in song_items[:limit]: # Process up to the limit
            if not isinstance(song_data, dict): continue

            video_id = song_data.get('videoId')
            if not video_id: continue # Skip songs without an ID

            title = song_data.get('title', 'Unknown Title')
            album_name = song_data.get('album', {}).get('name', '') if song_data.get('album') else ''
            year = song_data.get('year') # Might be string or int
            duration_str = song_data.get('duration') # e.g., "3:45"
            duration_sec = parse_duration(duration_str)
            plays = song_data.get('views') or song_data.get('playCount') # Check multiple possible keys for plays/views
            thumbnail = get_best_thumbnail(song_data.get('thumbnails', [])) \
                        or f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"

            top_songs_list.append({
                'title': title,
                'videoId': video_id,
                'plays': safe_format_count(plays) if plays else 'N/A', # Format plays
                'duration': duration_sec, # Store as seconds
                'duration_str': duration_str or '', # Keep original string if needed
                'thumbnail': thumbnail,
                'album': album_name,
                'year': str(year) if year else '' # Ensure year is string
            })

    except Exception as e:
        logger.warning(f"Error processing top songs: {e}", exc_info=True)
        # Return potentially partial list on error

    return top_songs_list


# --- Function: process_artist_links ---
def process_artist_links(artist_data: Dict, artist_id: Optional[str]) -> Dict[str, Optional[str]]:
    """Extracts relevant external links for an artist."""
    links = {
        'youtube_music': f"https://music.youtube.com/channel/{artist_id}" if artist_id else None,
        'youtube_channel': None, # Placeholder if YT channel link can be found
        'official_website': None,
        'wikipedia': None, # Placeholder
        # Add social media if needed: 'instagram', 'twitter', 'facebook', 'spotify', etc.
    }
    try:
        # Official Website (less common in YTMusic API)
        # Check if there's a specific field or parse from description?
        pass # Add logic if source for official website is found

        # Other links might be in a 'links' array or similar structure
        # Example (adapt based on actual API response structure):
        # for link_info in artist_data.get('externalLinks', []):
        #     url = link_info.get('url')
        #     link_type = link_info.get('type', '').lower() # e.g., 'website', 'wikipedia', 'instagram'
        #     if url and link_type in links:
        #          links[link_type] = url

        # Placeholder for YouTube Channel (might need separate API call or parsing)
        # links['youtube_channel'] = ...

    except Exception as e:
        logger.warning(f"Error processing artist links: {e}")

    # Return only links that have a value
    return {k: v for k, v in links.items() if v}


# --- Function: extract_video_id (Redundant with is_potential_video_id + string slicing) ---
# Can likely be replaced by using `is_potential_video_id` and basic URL parsing.
# Commenting out, replace calls with standard URL parsing where needed.
# def extract_video_id(url):
#    """Extract video ID from various YouTube/YouTube Music URL formats"""
#    # ... (original implementation) ...
#    pass

# --- Function: extract_year ---
def extract_year(artist_data: Dict) -> Optional[str]:
    """Extracts the artist's formation or first active year."""
    try:
        # Try specific fields first
        year = artist_data.get('year') or artist_data.get('startYear')
        if year and isinstance(year, (int, str)) and str(year).isdigit() and 1900 < int(year) <= datetime.now().year:
            return str(year)

        # Fallback: Parse from description (less reliable)
        desc = artist_data.get('description', '')
        if isinstance(desc, str):
            # Look for a 4-digit year likely representing start year
            match = re.search(r'\b(19[5-9]\d|20\d{2})\b', desc) # Look for years 1950 onwards
            if match:
                return match.group(0)

        return None # Return None if no year found
    except Exception as e:
        logger.warning(f"Error extracting artist year: {e}")
        return None


# --- Function: safe_format_count ---
def safe_format_count(count: Optional[Any]) -> str:
    """
    Safely formats a number (or string representing a number) into a
    human-readable format with K, M, B suffixes. Handles various input types.
    Returns 'N/A' if formatting fails or input is invalid.
    """
    if count is None or count == '':
        return 'N/A'

    count_str = str(count).strip()
    if not count_str or count_str.lower() in ['none', 'null']:
        return 'N/A'

    # Remove commas, spaces for processing
    cleaned_count_str = re.sub(r'[,\s]', '', count_str)

    # Check if already formatted (e.g., "1.5M")
    if re.fullmatch(r'[\d.]+[KMB]$', cleaned_count_str, re.IGNORECASE):
        # Ensure consistent case
        if cleaned_count_str[-1].islower():
             return cleaned_count_str[:-1] + cleaned_count_str[-1].upper()
        return cleaned_count_str

    try:
        num = float(cleaned_count_str)
        if num < 1000:
            # Show exact number for smaller counts
            return str(int(num)) if num.is_integer() else f"{num:.0f}" # Avoid ".0"
        elif num < 1_000_000:
            # Format as K (e.g., 1.2K, 123K)
            val = num / 1000.0
            return f"{val:.1f}K".replace(".0K", "K") # Avoid ".0"
        elif num < 1_000_000_000:
            # Format as M
            val = num / 1_000_000.0
            return f"{val:.1f}M".replace(".0M", "M")
        else:
            # Format as B
            val = num / 1_000_000_000.0
            return f"{val:.1f}B".replace(".0B", "B")
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not format count '{count}': {e}. Returning original.")
        # Return the original string representation if conversion fails
        return count_str


# --- Function: parse_duration ---
def parse_duration(duration_str: Optional[str]) -> int:
    """Converts a duration string (e.g., "3:15", "1:02:30") into seconds."""
    if not duration_str or not isinstance(duration_str, str):
        return 0

    parts = duration_str.split(':')
    seconds = 0
    try:
        if len(parts) == 1: # Only seconds
            seconds = int(parts[0])
        elif len(parts) == 2: # Minutes and seconds
            seconds = int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3: # Hours, minutes, and seconds
            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        else:
             return 0 # Invalid format
        return max(0, seconds) # Ensure non-negative
    except (ValueError, TypeError):
        return 0 # Return 0 if parsing fails

# --- Function: safe_int (Redundant with int() try-except) ---
# Using `int()` with try-except is standard. Commenting out.
# def safe_int(value, default=0): ...

# ==============================================================================
# Image Fetching Utility
# ==============================================================================

# --- Function: fetch_image ---
@lru_cache(maxsize=500) # Cache image fetches
def fetch_image(url: str) -> Tuple[Optional[bytes], Optional[str]]:
    """Fetches image content and content type from a URL."""
    if not url:
        return None, None
    try:
        headers = {'User-Agent': 'Sangeet/1.0 (Image Proxy)'} # Identify proxy
        response = requests.get(url, headers=headers, timeout=8, stream=True) # Use stream=True
        response.raise_for_status() # Check for HTTP errors

        content_type = response.headers.get('Content-Type')
        # Basic validation of content type
        if not content_type or not content_type.startswith('image/'):
             logger.warning(f"Invalid content type '{content_type}' for image URL: {url}")
             return None, None

        # Read content (safer with stream=True)
        content = response.content # Read the whole content

        logger.debug(f"Successfully fetched image from: {url}")
        return content, content_type

    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching image: {url}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching image {url}: {e}")
    except Exception as e:
         logger.exception(f"Unexpected error fetching image {url}: {e}")

    return None, None


# ==============================================================================
# Email & OTP Utilities
# ==============================================================================

# --- Function: send_email ---
def send_email(to_email: str, subject: str, html_body: str) -> bool:
    """Sends an HTML email using configured SMTP settings."""
    if not all([SMTP_USER, SMTP_PASSWORD, SMTP_HOST]):
        logger.error("SMTP settings are not fully configured. Cannot send email.")
        return False
    if not to_email or not subject or not html_body:
         logger.error("Missing 'to_email', 'subject', or 'html_body' for send_email.")
         return False

    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = SMTP_USER
        msg['To'] = to_email
        msg['Subject'] = subject

        # Attach the HTML part
        msg.attach(MIMEText(html_body, 'html', 'utf-8')) # Specify UTF-8

        # Connect and send
        # Use SMTP_SSL if port is 465, otherwise use standard SMTP with STARTTLS
        if SMTP_PORT == 465:
             server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=10)
        else:
             server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10)
             server.ehlo()
             server.starttls()
             server.ehlo()

        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        logger.info(f"Email sent successfully to {to_email} with subject '{subject}'")
        return True

    except smtplib.SMTPAuthenticationError:
         logger.error("SMTP Authentication Error: Check SMTP_USER and SMTP_PASSWORD.")
         return False
    except Exception as e:
        logger.exception(f"Failed to send email to {to_email}: {e}")
        return False

# --- Function: generate_otp ---
def generate_otp(length: int = 6) -> str:
    """Generates a secure random OTP string of specified length."""
    if length < 4 or length > 8: length = 6 # Enforce reasonable length
    return ''.join([str(secrets.randbelow(10)) for _ in range(length)])

# --- Function: store_otp ---
def store_otp(email: str, otp: str, purpose: str):
    """Stores OTP in the database with an expiry time."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # Use timezone-aware UTC time for expiry
        expires_at = datetime.now(pytz.utc) + timedelta(minutes=10) # 10-minute validity

        # Ensure OTP table exists (optional, could be done at app init)
        c.execute("""
            CREATE TABLE IF NOT EXISTS pending_otps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                otp TEXT NOT NULL,
                purpose TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_pending_otps_email_purpose ON pending_otps(email, purpose)")

        # Remove old OTPs for the same email and purpose first
        c.execute("DELETE FROM pending_otps WHERE email = ? AND purpose = ?", (email, purpose))

        # Insert the new OTP
        c.execute("""
            INSERT INTO pending_otps (email, otp, purpose, expires_at, created_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (email, otp, purpose, expires_at.isoformat())) # Store as ISO string

        conn.commit()
        logger.info(f"Stored OTP for {email} (Purpose: {purpose})")

    except sqlite3.Error as e:
        logger.error(f"Database error storing OTP for {email} (Purpose: {purpose}): {e}")
        if conn: conn.rollback()
        # Raise error? Or just log? Logging for now.
    finally:
        if conn: conn.close()


# --- Function: verify_otp ---
def verify_otp(email: str, otp: str, purpose: str) -> bool:
    """Verifies an OTP against the database record and clears it if valid."""
    conn = None
    is_valid = False
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row # Use row factory for easier access
        c = conn.cursor()

        # Fetch the latest valid OTP for the email and purpose
        current_time_utc_iso = datetime.now(pytz.utc).isoformat()
        c.execute("""
            SELECT id, otp FROM pending_otps
            WHERE email = ? AND purpose = ?
            AND expires_at > ?
            ORDER BY created_at DESC LIMIT 1
        """, (email, purpose, current_time_utc_iso))
        row = c.fetchone()

        if row and row['otp'] == otp:
             is_valid = True
             otp_record_id = row['id']
             # --- OTP is valid, delete it ---
             c.execute("DELETE FROM pending_otps WHERE id = ?", (otp_record_id,))
             conn.commit()
             logger.info(f"OTP verified successfully for {email} (Purpose: {purpose}). Record deleted.")
        elif row:
             # OTP found but didn't match
             logger.warning(f"Invalid OTP entered for {email} (Purpose: {purpose}).")
             # Consider adding rate limiting or temporary lockout logic here
        else:
             # No valid OTP found (either expired or never existed)
             logger.warning(f"No valid OTP found for {email} (Purpose: {purpose}).")

    except sqlite3.Error as e:
        logger.error(f"Database error verifying OTP for {email} (Purpose: {purpose}): {e}")
        # Treat DB error as invalid OTP for security?
        is_valid = False
    finally:
        if conn: conn.close()

    return is_valid


# ==============================================================================
# Download Utilities (yt-dlp wrappers)
# ==============================================================================

# --- Function: download_flac ---
@login_required # Usually triggered by logged-in user action
def download_flac(video_id: str, user_id: int) -> Optional[Dict[str, str]]:
    """
    Downloads a YouTube video as FLAC using yt-dlp (executable preferred, module fallback).
    Records download details in Redis. Returns dictionary with path and metadata on success.
    """
    if not YTDLP_PATH: # Check if yt-dlp setup was successful
         logger.error(f"Cannot download {video_id}: yt-dlp path is not configured.")
         return None
    if not is_potential_video_id(video_id):
         logger.error(f"Invalid video ID format for download: {video_id}")
         return None

    flac_path = os.path.join(MUSIC_DIR, f"{video_id}.flac")
    yt_url = f"https://music.youtube.com/watch?v={video_id}" # Use music URL

    # --- Check if file already exists ---
    if os.path.exists(flac_path):
        logger.info(f"FLAC file already exists for {video_id} at {flac_path}. Skipping download.")
        # Ensure download is recorded even if file exists but wasn't recorded before
        try:
            # Check if recorded in Redis for this user
            if not redis_client or not redis_client.exists(f"download:{user_id}:{video_id}"):
                 logger.info(f"File exists but not recorded for user {user_id}. Fetching metadata and recording.")
                 info = get_media_info(video_id) # Fetch metadata
                 record_download(
                      video_id,
                      info.get('title', 'Unknown Title'),
                      info.get('artist', 'Unknown Artist'),
                      info.get('album', 'Unknown Album'),
                      flac_path,
                      user_id
                 )
            # Return existing path and potentially fetch metadata if needed by caller
            # For consistency, fetch metadata even if file exists
            info = get_media_info(video_id)
            return {
                 "path": flac_path,
                 "title": info.get('title'),
                 "artist": info.get('artist'),
                 "album": info.get('album')
                 }
        except Exception as e:
             logger.error(f"Error recording/fetching metadata for existing download {video_id}: {e}")
             # Fallback: just return the path
             return {"path": flac_path}
    # --- End Check ---

    logger.info(f"Attempting download for {video_id} to {flac_path}")
    download_success = False
    metadata = {}

    # --- Try downloading with executable ---
    try:
        logger.debug(f"Trying download with executable: {YTDLP_PATH}")
        metadata = download_with_executable(video_id, user_id, yt_url, flac_path, is_init=False)
        download_success = True
        logger.info(f"Download successful using executable for {video_id}.")
    except Exception as exe_error:
        logger.warning(f"yt-dlp executable download failed for {video_id}: {exe_error}. Falling back to module.")
        # Clean up partial file if executable failed? yt-dlp might handle this.
        if os.path.exists(flac_path): os.unlink(flac_path) # Remove partial if exists

        # --- Fallback: Try downloading with Python module ---
        try:
            logger.debug(f"Trying download with yt-dlp Python module for {video_id}")
            metadata = download_with_module(video_id, user_id, yt_url, flac_path, is_init=False)
            download_success = True
            logger.info(f"Download successful using module for {video_id}.")
        except Exception as mod_error:
            logger.error(f"yt-dlp module download also failed for {video_id}: {mod_error}", exc_info=True)
            # Ensure partial file is removed on final failure
            if os.path.exists(flac_path):
                 try: os.unlink(flac_path)
                 except OSError as unlink_err: logger.error(f"Failed to remove partial download file {flac_path}: {unlink_err}")

    # --- Return result ---
    if download_success and os.path.exists(flac_path):
        # Metadata should have been populated by the successful download function
        return {
             "path": flac_path,
             "title": metadata.get('title'),
             "artist": metadata.get('artist'),
             "album": metadata.get('album')
        }
    else:
        logger.error(f"Download failed for {video_id} after all attempts.")
        return None


# --- Function: download_flac_init ---
def download_flac_init(video_id: str) -> Optional[str]:
    """
    Special version of download_flac for use during application initialization.
    Does not record download in user-specific records. Returns only the path.
    """
    if not YTDLP_PATH:
        logger.error(f"Cannot download initial song {video_id}: yt-dlp path is not configured.")
        return None
    if not is_potential_video_id(video_id):
         logger.error(f"Invalid video ID format for initial download: {video_id}")
         return None

    flac_path = os.path.join(MUSIC_DIR, f"{video_id}.flac")
    yt_url = f"https://music.youtube.com/watch?v={video_id}"

    if os.path.exists(flac_path):
        logger.info(f"Initial song {video_id} already exists at {flac_path}.")
        return flac_path

    logger.info(f"Attempting initial download for {video_id}...")
    download_success = False

    # Try executable first
    try:
        download_with_executable(video_id, None, yt_url, flac_path, is_init=True)
        download_success = True
        logger.info(f"Initial download successful using executable for {video_id}.")
    except Exception as exe_error:
        logger.warning(f"Executable download failed during init for {video_id}: {exe_error}. Falling back to module.")
        if os.path.exists(flac_path): os.unlink(flac_path) # Clean partial

        # Fallback to module
        try:
            download_with_module(video_id, None, yt_url, flac_path, is_init=True)
            download_success = True
            logger.info(f"Initial download successful using module for {video_id}.")
        except Exception as mod_error:
            logger.error(f"Module download also failed during init for {video_id}: {mod_error}", exc_info=True)
            if os.path.exists(flac_path):
                 try: os.unlink(flac_path)
                 except OSError as unlink_err: logger.error(f"Failed to remove partial init download file {flac_path}: {unlink_err}")


    return flac_path if download_success and os.path.exists(flac_path) else None


# --- Function: download_with_executable ---
def download_with_executable(video_id: str, user_id: Optional[int], url: str, flac_path: str, is_init: bool) -> Dict[str, str]:
    """Helper to download using the yt-dlp executable and capture metadata."""
    if not YTDLP_PATH or not os.path.exists(YTDLP_PATH):
         raise FileNotFoundError("yt-dlp executable not found or path not configured.")

    logger.debug(f"Using yt-dlp executable at: {YTDLP_PATH}")
    metadata = {}

    # --- Command Construction ---
    command = [
        YTDLP_PATH,
        '--no-check-certificate', # Add if needed for some environments
        '--ignore-errors',        # Continue on non-fatal errors
        '--no-warnings',
        '--extract-audio',        # -x
        '--audio-format', 'flac',
        '--audio-quality', '0',    # Best quality for FLAC (lossless)
        '--embed-metadata',       # Embed metadata if possible
        '--embed-thumbnail',      # Embed thumbnail if possible
        '--write-thumbnail',      # Write thumbnail to disk first (needed for embedding)
        '--output', os.path.join(MUSIC_DIR, '%(id)s.%(ext)s'), # Output template
        '--print', 'after_move:filepath', # Print final filepath after processing
        # Metadata printing (use specific fields) - These might print before download finishes
        # '--print', 'title',
        # '--print', 'artist',
        # '--print', 'album',
        # '--print', 'duration',
    ]
    # Add ffmpeg location if needed (especially for Windows)
    if platform.system().lower() == "windows" and os.path.isdir(FFMPEG_BIN_DIR):
        command.extend(['--ffmpeg-location', FFMPEG_BIN_DIR])
        logger.debug(f"Using ffmpeg location: {FFMPEG_BIN_DIR}")
    elif not os.path.isdir(FFMPEG_BIN_DIR):
         logger.debug("FFmpeg directory not found, relying on system PATH for ffmpeg.")


    command.append(url) # Add the URL last
    logger.debug(f"Executing command: {' '.join(command)}")
    # --- End Command Construction ---

    # --- Execute Download ---
    try:
        # Use subprocess.run for simpler execution and output capture
        result = subprocess.run(
             command,
             capture_output=True,
             text=True, # Decode output as text
             encoding='utf-8', # Specify encoding
             check=False, # Don't raise exception immediately on non-zero exit
             timeout=600 # Add a timeout (e.g., 10 minutes)
        )

        logger.debug(f"yt-dlp stdout:\n{result.stdout}")
        logger.debug(f"yt-dlp stderr:\n{result.stderr}")

        # Check exit code
        if result.returncode != 0:
             # Look for common errors in stderr
             err_msg = f"yt-dlp executable failed with code {result.returncode}."
             if "Unsupported URL" in result.stderr: err_msg += " (Unsupported URL?)"
             if "proxy" in result.stderr.lower(): err_msg += " (Proxy error?)"
             if "HTTP Error 429" in result.stderr: err_msg += " (Rate limited?)"
             logger.error(err_msg + f"\nStderr: {result.stderr[:500]}...") # Log truncated stderr
             raise Exception(err_msg)

        # Check if the expected FLAC file was created
        # The printed filepath might be more reliable if format changes happen
        printed_filepath = result.stdout.strip().split('\n')[-1] # Get last line (should be filepath)
        if printed_filepath and os.path.exists(printed_filepath) and printed_filepath.endswith('.flac'):
             actual_flac_path = printed_filepath
             logger.info(f"Confirmed FLAC file exists at printed path: {actual_flac_path}")
        elif os.path.exists(flac_path):
             actual_flac_path = flac_path
             logger.info(f"Confirmed FLAC file exists at expected path: {actual_flac_path}")
        else:
             logger.error(f"yt-dlp finished successfully but expected FLAC file not found.")
             logger.error(f"Expected: {flac_path}, Printed: {printed_filepath}")
             raise FileNotFoundError("Output FLAC file missing after download.")

        # --- Fetch Metadata Separately (more reliable) ---
        # Using get_media_info ensures consistency and uses caching
        info = get_media_info(video_id)
        metadata = {
             "title": info.get('title', 'Unknown Title'),
             "artist": info.get('artist', 'Unknown Artist'),
             "album": info.get('album', 'Unknown Album')
        }
        # --- End Metadata Fetch ---

        # Record download if not initialization and user ID provided
        if not is_init and user_id is not None:
             record_download(
                  video_id,
                  metadata['title'],
                  metadata['artist'],
                  metadata['album'],
                  actual_flac_path, # Use the confirmed path
                  user_id
             )
             load_local_songs() # Refresh local song list after download

        # Return metadata along with path
        metadata['path'] = actual_flac_path
        return metadata

    except subprocess.TimeoutExpired:
        logger.error(f"yt-dlp executable timed out for {video_id}.")
        raise TimeoutError("Download process timed out.")
    except FileNotFoundError:
         logger.error(f"yt-dlp executable not found at path: {YTDLP_PATH}")
         raise
    except Exception as e:
        logger.error(f"Error running yt-dlp executable for {video_id}: {e}", exc_info=True)
        raise # Re-raise the caught exception


# --- Function: download_with_module ---
def download_with_module(video_id: str, user_id: Optional[int], url: str, flac_path: str, is_init: bool) -> Dict[str, str]:
    """Helper to download using the yt-dlp Python module."""
    logger.debug("Using yt-dlp Python module for download.")
    metadata = {}

    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio/best', # Prioritize M4A for faster processing? Or keep default? Default bestaudio/best is safer.
        'outtmpl': os.path.join(MUSIC_DIR, '%(id)s.%(ext)s'), # Template for output filename
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'flac',
            'preferredquality': '0', # Best quality for FLAC
        },{
             'key': 'EmbedThumbnail', # Embed thumbnail if available
             'already_have_thumbnail': False,
        },{
             'key': 'FFmpegMetadata', # Embed metadata
             'add_metadata': True,
        }],
        'writethumbnail': True, # Need thumbnail file for embedding
        'keepvideo': False, # Don't keep original download
        'ignoreerrors': True, # Continue processing even if some parts fail (e.g., metadata)
        'no_warnings': True,
        'quiet': True, # Suppress console output from ydl itself
        'noprogress': True,
        'postprocessor_args': { # Arguments specifically for postprocessors
             'FFmpegExtractAudio': ['-acodec', 'flac', '-compression_level', '5'], # Example: specify compression level
             'FFmpegMetadata': ['-map_metadata', '0', '-map_metadata:s:a', '0:s:a'], # Map all metadata
        },
        # Specify ffmpeg location if necessary (read from config or detect)
        # 'ffmpeg_location': FFMPEG_BIN_DIR if platform.system().lower() == "windows" else None
    }
    if platform.system().lower() == "windows" and os.path.isdir(FFMPEG_BIN_DIR):
         ydl_opts['ffmpeg_location'] = FFMPEG_BIN_DIR
         logger.debug(f"yt-dlp module using ffmpeg location: {FFMPEG_BIN_DIR}")
    elif not os.path.isdir(FFMPEG_BIN_DIR):
         logger.debug("yt-dlp module relying on system PATH for ffmpeg.")

    try:
        with YoutubeDL(ydl_opts) as ydl:
             # Extract info dict first to get metadata before download starts
             info_dict = ydl.extract_info(url, download=False)
             metadata = {
                  "title": info_dict.get('title', 'Unknown Title'),
                  "artist": info_dict.get('artist', info_dict.get('uploader', 'Unknown Artist')),
                  "album": info_dict.get('album', 'Unknown Album')
             }

             # Now perform the download and processing
             logger.debug(f"Starting download & processing for {video_id} via module...")
             ydl.download([url]) # Pass URL in a list

             # Verify output file exists
             if not os.path.exists(flac_path):
                  logger.error(f"yt-dlp module finished but output FLAC file missing: {flac_path}")
                  # Check if a different extension was produced?
                  # List files in MUSIC_DIR matching video_id?
                  possible_files = list(Path(MUSIC_DIR).glob(f"{video_id}.*"))
                  logger.error(f"Files matching ID in music dir: {possible_files}")
                  raise FileNotFoundError("Output FLAC file missing after processing.")

             logger.info(f"yt-dlp module successfully processed: {flac_path}")

             # Record download if needed
             if not is_init and user_id is not None:
                  record_download(
                       video_id,
                       metadata['title'],
                       metadata['artist'],
                       metadata['album'],
                       flac_path,
                       user_id
                  )
                  load_local_songs() # Refresh local song list

             # Return metadata along with path
             metadata['path'] = flac_path
             return metadata

    except Exception as e:
        logger.error(f"Error using yt-dlp module for {video_id}: {e}", exc_info=True)
        # Clean up partial file if it exists
        if os.path.exists(flac_path):
             try: os.unlink(flac_path)
             except OSError as unlink_err: logger.error(f"Failed to remove partial download file {flac_path}: {unlink_err}")
        raise # Re-raise the exception


# ==============================================================================
# Database Interaction Utilities (Insights - SQLite based)
# ==============================================================================
# These functions interact directly with the SQLite `listening_history` table.

# --- Function: get_overview_stats ---
def get_overview_stats(cursor: sqlite3.Cursor, user_id: int) -> Dict[str, Any]:
    """Calculates overall listening stats for a user from the history table."""
    stats = {
        "total_time_seconds": 0,
        "total_songs_played": 0,
        "unique_artists_count": 0,
        "average_daily_plays": 0.0,
        "first_listen_date": None
    }
    try:
        # Total listening time (sum of listened_duration)
        cursor.execute("SELECT SUM(listened_duration) FROM listening_history WHERE user_id = ? AND listened_duration IS NOT NULL", (user_id,))
        total_time = cursor.fetchone()[0]
        stats["total_time_seconds"] = int(total_time) if total_time else 0

        # Total plays (count of entries)
        cursor.execute("SELECT COUNT(*) FROM listening_history WHERE user_id = ?", (user_id,))
        stats["total_songs_played"] = cursor.fetchone()[0] or 0

        # Unique artists count
        cursor.execute("SELECT COUNT(DISTINCT artist) FROM listening_history WHERE user_id = ? AND artist IS NOT NULL AND artist != ''", (user_id,))
        stats["unique_artists_count"] = cursor.fetchone()[0] or 0

        # Average daily plays (requires first listen date)
        first_listen_iso = get_first_listen_date(cursor, user_id) # Use helper
        stats["first_listen_date"] = first_listen_iso
        if first_listen_iso and stats["total_songs_played"] > 0:
             try:
                  first_date = datetime.fromisoformat(first_listen_iso).date()
                  today = datetime.now(pytz.utc).date()
                  days_since_first = (today - first_date).days + 1 # Add 1 to include first day
                  if days_since_first > 0:
                       stats["average_daily_plays"] = round(stats["total_songs_played"] / days_since_first, 1)
             except ValueError:
                  logger.warning(f"Could not parse first listen date '{first_listen_iso}' for daily average calculation.")

    except sqlite3.Error as e:
         logger.error(f"Database error getting overview stats for user {user_id}: {e}")
         # Return defaults on error
    return stats


# --- Function: get_first_listen_date ---
def get_first_listen_date(cursor: sqlite3.Cursor, user_id: int) -> Optional[str]:
    """Fetches the ISO date string of the earliest listen event for a user."""
    try:
        cursor.execute("SELECT MIN(started_at) FROM listening_history WHERE user_id = ?", (user_id,))
        first_listen_ts = cursor.fetchone()[0] # Format: 'YYYY-MM-DD HH:MM:SS'
        if first_listen_ts:
             # Parse and return only the date part in ISO format
             return datetime.strptime(first_listen_ts, '%Y-%m-%d %H:%M:%S').date().isoformat()
        return None
    except (sqlite3.Error, ValueError, TypeError) as e:
         logger.error(f"Error getting first listen date for user {user_id}: {e}")
         return None


# --- Function: get_recent_activity ---
def get_recent_activity(cursor: sqlite3.Cursor, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves the most recent listening activities for a user."""
    activities = []
    try:
        cursor.execute("""
            SELECT song_id, title, artist, started_at, listened_duration, completion_rate
            FROM listening_history
            WHERE user_id = ?
            ORDER BY started_at DESC
            LIMIT ?
        """, (user_id, limit))

        for row in cursor.fetchall(): # Assuming row_factory is set or access by index
            try:
                 # started_at is stored as 'YYYY-MM-DD HH:MM:SS' (likely naive UTC)
                 started_at_dt = time_sync.parse_datetime(row['started_at']) # Use TimeSync helper
                 if started_at_dt:
                      activities.append({
                           "id": row['song_id'], # Include ID
                           "title": row['title'] or "Unknown Title",
                           "artist": row['artist'] or "Unknown Artist",
                           "started_at": time_sync.format_time(started_at_dt), # Formatted time (e.g., IST)
                           "started_at_relative": time_sync.format_time(started_at_dt, relative=True), # Relative time
                           "listened_duration": row['listened_duration'], # Seconds
                           "completion_rate": round(row['completion_rate'], 1) if row['completion_rate'] is not None else None # Rounded %
                      })
                 else:
                      logger.warning(f"Could not parse started_at timestamp: {row['started_at']}")
            except Exception as e:
                 logger.error(f"Error processing recent activity row: {e} - Row: {dict(row)}")
                 continue # Skip malformed rows
    except sqlite3.Error as e:
        logger.error(f"Database error getting recent activity for user {user_id}: {e}")
    return activities


# --- Function: get_top_artists ---
def get_top_artists(cursor: sqlite3.Cursor, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieves the top artists based on total listening time for a user."""
    artists = []
    try:
        cursor.execute("""
            SELECT artist, COUNT(*) as plays, SUM(listened_duration) as total_time_seconds
            FROM listening_history
            WHERE user_id = ? AND artist IS NOT NULL AND artist != '' AND listened_duration IS NOT NULL
            GROUP BY artist
            HAVING total_time_seconds > 0 -- Exclude artists with zero listen time
            ORDER BY total_time_seconds DESC
            LIMIT ?
        """, (user_id, limit))

        artists = [dict(row) for row in cursor.fetchall()] # Convert rows to dicts

    except sqlite3.Error as e:
        logger.error(f"Database error getting top artists for user {user_id}: {e}")
    return artists


# --- Function: get_listening_patterns ---
def get_listening_patterns(cursor: sqlite3.Cursor, user_id: int) -> Dict[str, Any]:
    """Analyzes listening frequency by hour of day and day of week."""
    patterns = {
        "hourly": {str(h).zfill(2): 0 for h in range(24)}, # 00-23
        "daily": {str(d): 0 for d in range(7)} # 0=Sun, 1=Mon,..., 6=Sat
    }
    try:
        # Hourly pattern (using UTC time from DB - consider converting to user's TZ?)
        cursor.execute("""
            SELECT strftime('%H', started_at) as hour_of_day, COUNT(*) as play_count
            FROM listening_history
            WHERE user_id = ? AND started_at IS NOT NULL
            GROUP BY hour_of_day
        """, (user_id,))
        for row in cursor.fetchall():
            hour_str = row['hour_of_day']
            if hour_str in patterns["hourly"]:
                patterns["hourly"][hour_str] = row['play_count']

        # Daily pattern (0=Sunday, ..., 6=Saturday based on strftime('%w'))
        cursor.execute("""
            SELECT strftime('%w', started_at) as day_of_week, COUNT(*) as play_count
            FROM listening_history
            WHERE user_id = ? AND started_at IS NOT NULL
            GROUP BY day_of_week
        """, (user_id,))
        for row in cursor.fetchall():
             day_str = row['day_of_week']
             if day_str in patterns["daily"]:
                  patterns["daily"][day_str] = row['play_count']

    except sqlite3.Error as e:
        logger.error(f"Database error getting listening patterns for user {user_id}: {e}")
        # Return empty patterns on error
    return patterns


# --- Function: get_completion_rates ---
def get_completion_rates(cursor: sqlite3.Cursor, user_id: int) -> Dict[str, Any]:
    """Analyzes song completion distribution (full, partial, skip) and average."""
    results = {
        "completion_distribution": {'full': 0, 'partial': 0, 'skip': 0},
        "average_completion_rate": 0.0
    }
    try:
        # Get counts for each listen_type category
        cursor.execute("""
            SELECT COALESCE(listen_type, 'partial') as type, COUNT(*) as count
            FROM listening_history
            WHERE user_id = ?
            GROUP BY type
        """, (user_id,))
        for row in cursor.fetchall():
             listen_type = row['type']
             if listen_type in results["completion_distribution"]:
                  results["completion_distribution"][listen_type] = row['count']

        # Calculate average completion rate (where rate is valid)
        cursor.execute("""
            SELECT AVG(completion_rate)
            FROM listening_history
            WHERE user_id = ? AND completion_rate IS NOT NULL AND completion_rate BETWEEN 0 AND 100
        """, (user_id,))
        avg_rate = cursor.fetchone()[0]
        results["average_completion_rate"] = round(float(avg_rate), 1) if avg_rate is not None else 0.0

    except sqlite3.Error as e:
        logger.error(f"Database error getting completion rates for user {user_id}: {e}")
        # Return defaults on error
    return results


# --- Function: get_average_completion (Potentially Redundant) ---
# This calculates global average, not user-specific. Keep if needed, else remove.
# def get_average_completion(cursor: sqlite3.Cursor) -> float: ...


# ==============================================================================
# Database Interaction Utilities (Listen Start/End - SQLite based)
# ==============================================================================
# These functions interact directly with the SQLite `listening_history` table
# for starting and ending listen records.

# --- Function: record_listen_start ---
def record_listen_start(user_id: int, song_id: str, title: str, artist: str, session_id: str) -> Optional[int]:
    """Records the start of a listen event in the SQLite history table."""
    conn = None
    listen_id = None
    try:
        conn = sqlite3.connect(DB_PATH) # Use global main DB path
        c = conn.cursor()

        # Sanitize inputs (basic)
        song_id = str(song_id or '').strip()
        title = str(title or 'Unknown Title').strip()
        artist = str(artist or 'Unknown Artist').strip()
        session_id = str(session_id or '').strip()
        if not song_id or not session_id:
             raise ValueError("Song ID and Session ID are required to record listen start.")

        # Get current UTC time for started_at
        started_at_utc = datetime.now(pytz.utc)
        started_at_str = started_at_utc.strftime('%Y-%m-%d %H:%M:%S') # Store as naive UTC string

        c.execute("""
            INSERT INTO listening_history
                (user_id, song_id, title, artist, session_id, started_at, listen_type)
            VALUES (?, ?, ?, ?, ?, ?, 'partial') -- Default to partial initially
        """, (user_id, song_id, title, artist, session_id, started_at_str))

        listen_id = c.lastrowid
        conn.commit()
        logger.info(f"Recorded listen start in SQLite for user {user_id}, song {song_id}. Listen ID: {listen_id}")

    except sqlite3.Error as e:
        logger.error(f"Database error recording listen start (SQLite) for user {user_id}, song {song_id}: {e}")
        if conn: conn.rollback()
    except ValueError as e:
         logger.error(f"Value error recording listen start (SQLite): {e}")
    finally:
        if conn: conn.close()
    return listen_id


# --- Function: record_listen_end ---
def record_listen_end(listen_id: int, duration: Optional[Any], listened_duration: Optional[Any]):
    """Updates an existing listen record in SQLite with end details."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # --- Data Cleaning & Calculation ---
        try:
             # Use safe_int helper for robust conversion
             total_duration_sec = safe_int(duration, 0)
             listened_duration_sec = safe_int(listened_duration, 0)
        except Exception as conv_err:
             logger.error(f"Error converting duration values for listen end ({listen_id}): {conv_err}")
             total_duration_sec = 0
             listened_duration_sec = 0

        # Ensure listened duration is valid
        listened_duration_sec = max(0, listened_duration_sec)
        if total_duration_sec > 0:
             listened_duration_sec = min(listened_duration_sec, total_duration_sec)
        elif listened_duration_sec > 0:
             # If total duration is unknown/zero, use listened as total? Or leave total as 0?
             # Let's assume total duration should ideally be known. Setting completion to 0 if total unknown.
             pass # Keep total_duration_sec as 0 or invalid

        # Calculate completion rate (%)
        completion_rate = 0.0
        if total_duration_sec > 0:
            completion_rate = round((listened_duration_sec / total_duration_sec) * 100.0, 2)

        # Determine listen type
        # Thresholds: >= 90% = full, <= 15% = skip (adjust as needed)
        listen_type = 'partial'
        if completion_rate >= 90.0:
            listen_type = 'full'
        elif completion_rate <= 15.0 and listened_duration_sec < 30 : # Consider short duration for skips too
             listen_type = 'skip'
        # --- End Calculation ---

        # Get current UTC time for ended_at
        ended_at_utc = datetime.now(pytz.utc)
        ended_at_str = ended_at_utc.strftime('%Y-%m-%d %H:%M:%S') # Store as naive UTC string

        # --- Update Database ---
        c.execute("""
            UPDATE listening_history SET
                ended_at = ?,
                duration = ?,
                listened_duration = ?,
                completion_rate = ?,
                listen_type = ?
            WHERE id = ?
        """, (ended_at_str, total_duration_sec, listened_duration_sec, completion_rate, listen_type, listen_id))

        # Check if row was actually updated
        if c.rowcount == 0:
             logger.warning(f"Attempted to record listen end, but no row found for listen ID: {listen_id}")
             # This might happen if listen_start failed or ID is wrong
        else:
             conn.commit()
             logger.info(f"Recorded listen end in SQLite for listen ID {listen_id}. Listened: {listened_duration_sec}s, Rate: {completion_rate}%, Type: {listen_type}")

    except sqlite3.Error as e:
        logger.error(f"Database error recording listen end (SQLite) for listen ID {listen_id}: {e}")
        if conn: conn.rollback()
    except Exception as e:
         logger.exception(f"Unexpected error recording listen end (SQLite) for listen ID {listen_id}: {e}")
         if conn: conn.rollback() # Rollback on unexpected errors too
    finally:
        if conn: conn.close()


# --- Functions: update_artist_stats / update_daily_stats (SQLite based) ---
# These seem less critical if primary stats/insights come from history analysis.
# Keep if they serve a specific purpose (e.g., pre-aggregation for very high traffic).
# Ensure they use the correct connection and handle errors. Adding basic structure.

# @login_required # Decorator might not be needed if called internally after verification
def update_artist_stats(cursor: sqlite3.Cursor, artist: str, duration: int, listened_duration: int):
    """(SQLite based) Updates aggregated stats for an artist."""
    # Deprecated or requires separate artist_stats table
    logger.debug("update_artist_stats (SQLite based) called - check if still needed.")
    # Implementation would require an `artist_stats` table similar to the function body
    pass

# @login_required
def update_daily_stats(cursor: sqlite3.Cursor, duration: int, listened_duration: int):
    """(SQLite based) Updates aggregated daily listening stats."""
    # Deprecated or requires separate daily_stats table
    logger.debug("update_daily_stats (SQLite based) called - check if still needed.")
    # Implementation would require a `daily_stats` table similar to the function body
    pass


# ==============================================================================
# Fallback/Default Data Utilities
# ==============================================================================

# --- Function: get_default_thumbnail ---
def get_default_thumbnail() -> str:
    """Returns the URL or path to a default thumbnail image."""
    # Use url_for if serving from Flask static, or provide absolute URL
    try:
        # Assuming a default image exists in static/images
        return url_for('static', filename='images/default-cover.png', _external=False)
    except RuntimeError:
         # url_for might fail if called outside request context
         return '/static/images/default-cover.png' # Hardcoded path fallback

# --- Function: get_fallback_artist_info ---
def get_fallback_artist_info(artist_name: str) -> Dict:
     """Returns a minimal structure for artist info when API calls fail."""
     return {
          'name': artist_name,
          'thumbnail': get_default_thumbnail(),
          'description': 'Artist information currently unavailable.',
          'genres': [],
          'year': None,
          'stats': {'subscribers': 'N/A', 'views': 'N/A', 'monthlyListeners': 'N/A'},
          'topSongs': [],
          'links': {}
     }


# ==============================================================================
# Initialization Calls (Run on module load)
# ==============================================================================

# --- Load Local Songs Cache on Startup ---
# load_local_songs() # Load local songs into memory and sync with Redis
# Consider calling this explicitly at app startup instead of module load
# if scan time is significant or Redis might not be ready yet.
logger.info("util.py loaded. Consider calling load_local_songs() at app startup.")

# --- Download Default Songs (Optional) ---
# download_default_songs() # Download essential fallback songs if they don't exist
# Also consider running this explicitly at startup.


# ==============================================================================
# End of File
# ==============================================================================

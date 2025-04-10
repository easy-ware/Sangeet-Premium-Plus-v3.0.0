# ==============================================================================
# playback.py - Sangeet Premium Playback & Core Routes
# ==============================================================================
# Description:
#   This module defines the main Flask Blueprint ('playback') for the Sangeet
#   Premium application. It includes routes for:
#   - Core application pages (home, login, register, etc.)
#   - API endpoints for searching, streaming, downloading songs.
#   - User authentication and session management.
#   - Playback control (play sequence, queue).
#   - User insights and statistics.
#   - Playlist management.
#   - Issue reporting and management (user and admin views).
#   - Embeddable player functionality.
#   - Extension interaction and downloads.
#   - Helper functions for data retrieval and processing.
# ==============================================================================

# --- Standard Library Imports ---
import os
import json
import logging
import random
import re
import secrets
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, concurrent
from datetime import datetime, timedelta
from functools import wraps, partial, lru_cache
from threading import Thread
from urllib.parse import urlparse, parse_qs

# --- External Library Imports ---
import bcrypt
import redis
import requests
import yt_dlp
from bs4 import BeautifulSoup
from flask import (
    Blueprint, session, jsonify, send_file, url_for, render_template,
    request, redirect, render_template_string, make_response, current_app, abort
)
from pytz import timezone as pytz_timezone, UnknownTimeZoneError # Import specific items needed
from ytmusicapi import YTMusic

# --- Internal Application Imports ---
import pytz # Keep pytz for potential direct use elsewhere if needed
from ..utils import util  # Assuming util is in a 'utils' directory one level up
from sangeet_premium import var_templates # Assuming var_templates is at the root

# ==============================================================================
# Global Variables, Constants & Configuration
# ==============================================================================

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Database Paths ---
DB_PATH = os.path.join(os.getcwd(), "database_files", "sangeet_database_main.db")
DB_PATH_MAIN = os.path.join(os.getcwd(), "database_files", "sangeet_database_main.db")
DB_PATH_ISSUES = os.path.join(os.getcwd(), "database_files", "issues.db")
PLAYLIST_DB_PATH = os.path.join(os.getcwd(), "database_files", "playlists.db")

# --- Local Data Path ---
LOCAL_JSON_PATH = os.path.join(os.getcwd(), "locals", "local.json")

# --- Environment Variable Handling (Case-Insensitive Fallbacks) ---
# Server Domain: Try lowercase 'sangeet_backend', then uppercase, then default
_backend_env = os.getenv('SANGEET_BACKEND') 
_port_env = os.getenv("PORT") 
SERVER_DOMAIN = _backend_env if _backend_env else f'http://127.0.0.1:{_port_env}'

# Music Path: Try lowercase 'music_path', then uppercase
MUSIC_PATH_ENV = os.getenv("MUSIC_PATH") 
if not MUSIC_PATH_ENV:
    logger.warning("Environment variable 'music_path' or 'MUSIC_PATH' not set. Defaulting to './music'.")
    MUSIC_PATH_ENV = os.path.join(os.getcwd(), "music")
    os.makedirs(MUSIC_PATH_ENV, exist_ok=True) # Ensure default exists

# Admin Password (Consider secure methods like Vault or proper env management)
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# --- Caches & In-Memory Stores ---
local_songs = {} # Populated by load_local_songs_from_file
search_cache = {} # Note: This seems unused, consider removing if confirmed
song_cache = {} # Simple dict cache for YTMusic API song results
lyrics_cache = {} # Note: This seems unused (SQLite cache is used), consider removing if confirmed
CACHE_DURATION = 3600 # In seconds (1 hour)

# --- Blueprint & External Service Initialization ---
bp = Blueprint('playback', __name__)
try:
    # Standard Redis connection
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping() # Check connection
    logger.info("Successfully connected to Redis.")
except redis.exceptions.ConnectionError as e:
    logger.error(f"Could not connect to Redis: {e}. Some features requiring Redis might be unavailable.")
    redis_client = None # Set to None to handle gracefully in functions

try:
    ytmusic = YTMusic()
    logger.info("YTMusic API client initialized.")
except Exception as e:
     logger.error(f"Failed to initialize YTMusic API client: {e}")
     ytmusic = None # Allow app to run but YTMusic features will fail

# --- Load Initial Data ---
try:
    with open(LOCAL_JSON_PATH, 'r', encoding='utf-8') as f:
        local_data = json.load(f)
except Exception as e:
    logger.error(f"Error reading the local JSON file ({LOCAL_JSON_PATH}): {e}")
    local_data = {} # Initialize as empty if loading fails

# ==============================================================================
# Decorators & Utility Functions
# ==============================================================================

# --- Decorator: login_required ---
def login_required(f):
    """Decorator to ensure a user is logged in and has a valid session."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or 'session_token' not in session:
            if request.path.startswith('/api/'):
                 return jsonify({"error": "Authentication required"}), 401
            return redirect(url_for('playback.login', next=request.url))

        # Verify session is still valid in database
        conn = None
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("""
                SELECT 1 FROM active_sessions
                WHERE user_id = ? AND session_token = ?
                AND expires_at > CURRENT_TIMESTAMP
            """, (session['user_id'], session['session_token']))
            valid_session = c.fetchone()
        except sqlite3.Error as e:
            logger.error(f"Database error checking session validity: {e}")
            valid_session = None # Assume invalid on DB error
        finally:
            if conn:
                conn.close()

        if not valid_session:
            logger.warning(f"Invalid session detected for user {session.get('user_id')}. Clearing session.")
            session.clear()
            if request.path.startswith('/api/'):
                 return jsonify({"error": "Session expired or invalid"}), 401
            return redirect(url_for('playback.login'))

        return f(*args, **kwargs)
    return decorated_function

# --- Helper Function: load_local_songs_from_file ---
def load_local_songs_from_file():
    """Load songs from Redis into the local_songs dictionary, ensuring keys and path existence."""
    global local_songs
    local_songs = {}
    if not redis_client:
        logger.error("Redis client not available. Cannot load local songs.")
        return local_songs

    try:
        song_ids = redis_client.smembers("local_songs_ids") or set()
        required_keys = ["id", "title", "artist", "album", "path", "thumbnail", "duration"]
        keys_to_delete_from_redis = set()
        paths_to_remove_from_mapping = set()

        pipeline = redis_client.pipeline()
        for song_id in song_ids:
            pipeline.hgetall(song_id)
        all_song_data_raw = pipeline.execute()

        for song_id, song_data in zip(song_ids, all_song_data_raw):
            if not song_data: # Skip if hash doesn't exist
                logger.warning(f"Song ID {song_id} found in set but hash is missing in Redis. Scheduling removal.")
                keys_to_delete_from_redis.add(song_id)
                redis_client.srem("local_songs_ids", song_id) # Remove from set immediately
                continue

            # Check for required keys
            if not all(key in song_data for key in required_keys):
                missing_keys = set(required_keys) - set(song_data.keys())
                logger.warning(f"Song {song_id} is missing keys: {missing_keys}. Scheduling removal.")
                keys_to_delete_from_redis.add(song_id)
                if "path" in song_data:
                     paths_to_remove_from_mapping.add(song_data["path"])
                continue

            # Check if path exists
            song_path = song_data.get("path")
            if not song_path or not os.path.exists(song_path):
                logger.warning(f"Path '{song_path}' for song {song_id} does not exist. Scheduling removal.")
                keys_to_delete_from_redis.add(song_id)
                if song_path:
                    paths_to_remove_from_mapping.add(song_path)
                continue

            # Validate duration is integer
            try:
                duration_int = int(song_data["duration"])
            except (ValueError, TypeError):
                logger.warning(f"Invalid duration '{song_data['duration']}' for song {song_id}. Scheduling removal.")
                keys_to_delete_from_redis.add(song_id)
                paths_to_remove_from_mapping.add(song_path)
                continue

            # If all checks pass, add to local_songs cache
            local_songs[song_id] = {
                "id": song_data["id"],
                "title": song_data["title"],
                "artist": song_data["artist"],
                "album": song_data["album"],
                "path": song_path,
                "thumbnail": song_data["thumbnail"],
                "duration": duration_int
            }

        # Perform cleanup in Redis
        if keys_to_delete_from_redis:
            logger.info(f"Cleaning up {len(keys_to_delete_from_redis)} invalid song entries from Redis...")
            redis_client.delete(*keys_to_delete_from_redis)
            redis_client.srem("local_songs_ids", *keys_to_delete_from_redis) # Ensure removal from set too
        if paths_to_remove_from_mapping:
            redis_client.hdel("path_to_id", *paths_to_remove_from_mapping)

    except redis.exceptions.RedisError as e:
        logger.error(f"Redis error loading local songs: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading local songs: {e}")

    logger.info(f"Loaded {len(local_songs)} valid songs from Redis into memory.")
    return local_songs

# --- Helper Function: get_default_songs (Cached) ---
@lru_cache(maxsize=1)
def get_default_songs():
    """Generate a cached list of default songs for empty queries (uses loaded local songs)."""
    # Note: This function depends on load_local_songs_from_file having been run.
    # It might be better to pass local_songs as an argument if needed elsewhere before loading.
    current_local_songs = local_songs # Use the globally loaded dictionary
    combined = []
    seen_ids = set()

    # Add songs from the current in-memory local_songs cache
    for song in current_local_songs.values():
        if song["id"] not in seen_ids:
            combined.append(song)
            seen_ids.add(song["id"])

    # Consider adding a few hardcoded popular/fallback songs if local_songs is empty
    # if not combined:
    #     # fallback_ids = ["dQw4w9WgXcQ", ...]
    #     # fetch info for fallback_ids and add
    #     pass

    return combined


# --- Helper Function: extract_playlist_info ---
def extract_playlist_info(url, max_workers=4):
    """Extract playlist item information using yt-dlp."""
    try:
        ydl_opts = {
            'quiet': True,
            'extract_flat': 'discard_in_playlist', # Extract full info per item
            'force_generic_extractor': False,
            'playlistend': 50 # Limit number of items extracted to avoid long waits
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            if info and info.get('_type') == 'playlist' and 'entries' in info:
                results = []
                # Use ThreadPoolExecutor for potentially faster metadata extraction if needed,
                # but yt-dlp might already be efficient. Simpler loop first:
                for entry in info['entries']:
                     if entry and entry.get('id'):
                          video_info = extract_video_info_from_entry(entry)
                          if video_info:
                               results.append(video_info)
                return results
            # Handle cases where it's not detected as a playlist but might be single video
            elif info and info.get('id'):
                 video_info = extract_video_info_from_entry(info)
                 return [video_info] if video_info else []
            else:
                 logger.warning(f"Could not extract playlist or video info from URL: {url}")
                 return []

    except yt_dlp.utils.DownloadError as e:
        logger.error(f"yt-dlp DownloadError extracting playlist info from {url}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error extracting playlist info from {url}: {e}")
        return []

# --- Helper Function: extract_video_info_from_entry ---
def extract_video_info_from_entry(entry):
    """Extracts structured video info from a yt-dlp entry dictionary."""
    if not entry or not entry.get('id'):
        return None
    return {
        "id": entry['id'],
        "title": entry.get('title', 'Unknown Title'),
        "artist": entry.get('artist') or entry.get('uploader', 'Unknown Artist'),
        "album": entry.get('album', ''),
        "duration": int(entry.get('duration', 0)),
        "thumbnail": get_best_thumbnail(entry.get('thumbnails', [])),
    }


# --- Helper Function: extract_video_info (Original, potentially redundant with above) ---
# Kept for compatibility if called directly elsewhere, but extract_playlist_info is preferred.
def extract_video_info(url, ydl_opts):
    """Extract single video information using yt-dlp."""
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return extract_video_info_from_entry(info) # Reuse the helper
    except yt_dlp.utils.DownloadError as e:
        logger.error(f"yt-dlp DownloadError extracting video info from {url}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error extracting video info from {url}: {e}")
    return None

# --- Helper Function: get_best_thumbnail ---
def get_best_thumbnail(thumbnails):
    """Get the best quality thumbnail URL from a list of thumbnail dictionaries."""
    if not thumbnails or not isinstance(thumbnails, list):
        return ""
    # Sort by preference: height * width (larger is better)
    # Handle entries without width/height gracefully
    valid_thumbs = [t for t in thumbnails if isinstance(t, dict) and t.get('url')]
    if not valid_thumbs:
        return ""

    sorted_thumbs = sorted(
        valid_thumbs,
        key=lambda x: x.get('height', 0) * x.get('width', 0),
        reverse=True
    )
    return sorted_thumbs[0].get('url', '')


# --- Helper Function: get_cached_lyrics ---
def get_cached_lyrics(song_id):
    """Retrieve lyrics from the SQLite cache."""
    conn = None
    try:
        # Ensure the lyrics cache DB path is defined and used
        lyrics_db_path = os.path.join(os.getcwd(), "database_files", "lyrics_cache.db")
        conn = sqlite3.connect(lyrics_db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT lyrics FROM lyrics_cache WHERE song_id = ?', (song_id,))
        result = cursor.fetchone()
        if result and result[0]:
            # Return as list of lines
            return result[0].split('\n')
        return None
    except sqlite3.Error as e:
        logger.error(f"Database error getting cached lyrics for {song_id}: {e}")
        return None
    finally:
        if conn:
            conn.close()

# --- Helper Function: cache_lyrics ---
def cache_lyrics(song_id, lyrics_lines):
    """Cache lyrics lines into the SQLite database."""
    if not isinstance(lyrics_lines, list):
        logger.error("cache_lyrics expects a list of lines.")
        return
    lyrics_text = '\n'.join(lyrics_lines)
    conn = None
    try:
        # Ensure the lyrics cache DB path is defined and used
        lyrics_db_path = os.path.join(os.getcwd(), "database_files", "lyrics_cache.db")
        conn = sqlite3.connect(lyrics_db_path)
        cursor = conn.cursor()
        # Ensure table exists (optional, could be done at init)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS lyrics_cache (
            song_id TEXT PRIMARY KEY,
            lyrics TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        # Insert or replace
        cursor.execute('''
        INSERT OR REPLACE INTO lyrics_cache (song_id, lyrics, timestamp)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (song_id, lyrics_text))
        conn.commit()
        logger.info(f"Cached lyrics for {song_id}")
    except sqlite3.Error as e:
        logger.error(f"Database error caching lyrics for {song_id}: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


# --- Helper Function: get_video_info ---
def get_video_info(video_id):
    """Fetch video metadata primarily using YTMusic API, with a fallback to web scraping."""
    # Check cache first
    if video_id in song_cache:
        return song_cache[video_id]

    # Try YTMusic API
    if ytmusic:
        try:
            song_info = ytmusic.get_song(video_id)
            details = song_info.get("videoDetails", {})
            title = details.get("title", "Unknown Title")
            # Prefer specific artist field if available, fallback to author/uploader
            artist = "Unknown Artist"
            if song_info.get("artists") and isinstance(song_info["artists"], list) and song_info["artists"]:
                 artist = song_info["artists"][0].get("name", artist)
            elif details.get("author"):
                 artist = details["author"]

            thumbnails = details.get("thumbnail", {}).get("thumbnails", [])
            thumbnail_url = get_best_thumbnail(thumbnails) or f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg" # Fallback thumb

            result = {
                "title": title,
                "artist": artist,
                "thumbnail": thumbnail_url,
                "video_id": video_id,
                "album": song_info.get("album", {}).get("name", ""), # Add album info
                "duration": int(details.get("lengthSeconds", 0)) # Add duration
                }
            song_cache[video_id] = result # Cache the result
            return result
        except Exception as e:
            logger.warning(f"YTMusic API failed for get_video_info({video_id}): {e}. Falling back.")
            # Fall through to web scraping

    # Fallback: minimal extraction from YouTube page metadata (less reliable)
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        soup = BeautifulSoup(response.text, "html.parser")
        og_title_tag = soup.find("meta", property="og:title")
        title = og_title_tag["content"] if og_title_tag and og_title_tag.get("content") else "Unknown Title"
        # Artist extraction from meta tags is unreliable, default to Unknown
        artist = "Unknown Artist"
        # Extract description for potential artist info (though often not structured)
        # og_desc_tag = soup.find("meta", property="og:description")
        # description = og_desc_tag["content"] if og_desc_tag else ""

        # Use standard high-quality thumbnail as fallback
        thumbnail_url = f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg"

        result = {
            "title": title,
            "artist": artist,
            "thumbnail": thumbnail_url,
            "video_id": video_id,
            "album": "", # Not easily available via basic scraping
            "duration": 0 # Not easily available via basic scraping
            }
        song_cache[video_id] = result # Cache the fallback result
        return result

    except requests.exceptions.RequestException as e:
         logger.error(f"Web scraping fallback failed for get_video_info({video_id}): {e}")
    except Exception as e:
         logger.error(f"Unexpected error during web scraping fallback for {video_id}: {e}")

    # Final fallback if all methods fail
    fallback_result = {
        "title": "Unknown Title",
        "artist": "Unknown Artist",
        "thumbnail": f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg",
        "video_id": video_id,
        "album": "",
        "duration": 0
        }
    song_cache[video_id] = fallback_result # Cache the final fallback
    return fallback_result


# --- Helper Function: get_media_info ---
def get_media_info(media_id):
    """
    Returns media details based on the given media_id.
    Handles 'local-' prefix for local songs, otherwise uses get_video_info.
    """
    if media_id.startswith("local-"):
        # Load local songs fresh each time or rely on the global cache?
        # Relying on global cache 'local_songs' which should be updated periodically or at start
        load_local_songs_from_file() # Ensure it's up-to-date for this call
        details = local_songs.get(media_id)
        if details:
            # Ensure the returned dictionary has the keys expected by templates/callers
            # The structure from load_local_songs_from_file should match get_video_info's structure
            return details
        else:
            # Return a fallback structure if local song ID is not found
            logger.warning(f"Local media ID '{media_id}' not found in local_songs cache.")
            return {
                "title": "Unknown Local Song",
                "artist": "Unknown Artist",
                "thumbnail": "", # Provide a default placeholder image URL if needed
                "video_id": media_id,
                "id": media_id, # Ensure 'id' key is present
                "album": "",
                "duration": 0,
                "path": "" # Indicate path is unknown
            }
    else:
        # For non-local IDs, use the standard video info function
        return get_video_info(media_id)


# --- Helper: Get DB Connection for Issues ---
def get_issues_db():
    """Gets a connection to the issues database, ensuring tables exist."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH_ISSUES)
        conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
        cursor = conn.cursor()
        # Create user_issues table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_issues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                topic TEXT NOT NULL,
                details TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Open', -- e.g., Open, In Progress, Resolved, Closed
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Create issue_comments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS issue_comments (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 issue_id INTEGER NOT NULL,
                 user_id INTEGER, -- Nullable for admin/system comments
                 is_admin BOOLEAN DEFAULT 0,
                 comment TEXT NOT NULL,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                 FOREIGN KEY (issue_id) REFERENCES user_issues(id) ON DELETE CASCADE -- Ensure comments are deleted if issue is deleted
            )
        """)
        conn.commit()
        return conn
    except sqlite3.Error as e:
        logger.error(f"Failed to get or initialize issues DB: {e}")
        if conn:
            conn.rollback() # Rollback any partial changes if table creation failed mid-way
            conn.close()
        raise # Re-raise the exception so the calling function knows DB isn't available


# --- Helper: Get DB Connection for Main ---
def get_main_db():
    """Gets a connection to the main application database."""
    conn = sqlite3.connect(DB_PATH_MAIN)
    conn.row_factory = sqlite3.Row
    return conn

# --- Decorator: admin_required ---
def admin_required(f):
    """Decorator to ensure the user is authenticated as an admin via session flag."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_authed'):
            # For API calls, return 401 Unauthorized
            if request.path.startswith('/api/admin'):
                 return jsonify({"error": "Admin authentication required"}), 401
            # For regular page loads, redirect to the password entry page
            # Pass a flag indicating the reason for redirect if needed
            return redirect(url_for('playback.view_admin_issues', error='auth_required'))
        return f(*args, **kwargs)
    return decorated_function

# ==============================================================================
# Flask Routes
# ==============================================================================

# --- Route: / (Home Page) ---
@bp.route('/')
@login_required
def home():
    """Renders the main application page (index.html)."""
    # You could pass user-specific data here if needed for the template
    user_id = session.get('user_id')
    # Example: Fetch username
    username = "User"
    conn = None
    try:
         conn = get_main_db()
         c = conn.cursor()
         c.execute("SELECT username FROM users WHERE id = ?", (user_id,))
         user_record = c.fetchone()
         if user_record:
              username = user_record['username']
    except sqlite3.Error as e:
         logger.error(f"Error fetching username for home page: {e}")
    finally:
         if conn:
              conn.close()

    return render_template("index.html", username=username)

# --- Route: /api/play-sequence/<song_id>/<action> ---
@bp.route("/api/play-sequence/<song_id>/<action>")
@login_required
def api_play_sequence(song_id, action):
    """
    Handles fetching the previous or next song in the user's listening sequence
    based on Redis history. Provides recommendations if 'next' is requested at the end.
    """
    if not redis_client:
        return jsonify({"error": "History service unavailable"}), 503

    load_local_songs_from_file() # Ensure local song cache is fresh
    user_id = session['user_id']
    history_key = f"user_history:{user_id}"

    try:
        # Get all history entries (consider limiting for very long histories if performance becomes an issue)
        history_entries_raw = redis_client.lrange(history_key, 0, -1)
        if not history_entries_raw:
            return jsonify({"error": "No listening history found"}), 404

        # Parse history entries
        history_entries = []
        for entry_raw in history_entries_raw:
             try:
                  history_entries.append(json.loads(entry_raw))
             except json.JSONDecodeError:
                  logger.warning(f"Skipping invalid JSON entry in history for user {user_id}")
                  continue

        # Find current song's latest occurrence and its sequence info
        current_entry_index = -1
        current_session = None
        current_seq = -1
        for i, entry in reversed(list(enumerate(history_entries))): # Search backwards for latest play
             if entry.get('song_id') == song_id:
                  current_entry_index = i
                  current_session = entry.get('session_id')
                  current_seq = entry.get('sequence_number', -1) # Use get with default
                  break

        if current_entry_index == -1 or current_session is None or current_seq == -1:
            logger.warning(f"Current song {song_id} not found in parsed history for user {user_id}.")
            return jsonify({"error": "Current song not found in recent history"}), 404

        target_song_id = None

        if action == "previous":
            # Find the immediate previous song in the *same session*
            for i in range(current_entry_index - 1, -1, -1):
                 prev_entry = history_entries[i]
                 if prev_entry.get('session_id') == current_session:
                     target_song_id = prev_entry.get('song_id')
                     break # Found the immediately preceding song in session
            if not target_song_id:
                return jsonify({"message": "Start of session history reached"}), 404 # Or handle differently

        elif action == "next":
            # Find the immediate next song in the *same session*
            for i in range(current_entry_index + 1, len(history_entries)):
                 next_entry = history_entries[i]
                 if next_entry.get('session_id') == current_session:
                      target_song_id = next_entry.get('song_id')
                      break # Found the immediately succeeding song in session

            # If no next song in the *current session*, get recommendations
            if not target_song_id:
                 logger.info(f"End of session history reached for {song_id}. Fetching recommendations.")
                 # Use the recommendation function directly
                 recommendations = util.get_recommendations_for_song(song_id, limit=5) # Assuming util has this func
                 if recommendations:
                     # Return the first recommendation as the "next" song
                     return jsonify(recommendations[0])
                 else:
                     # Fallback if recommendations fail
                     return jsonify({"error": "End of history and no recommendations available"}), 404

        else:
            return jsonify({"error": "Invalid action specified (use 'previous' or 'next')"}), 400

        # If a target song (previous/next) was found
        if target_song_id:
            # Get metadata for the target song
            media_info = get_media_info(target_song_id)
            if media_info and media_info.get('title') != 'Unknown Title': # Basic check for validity
                 return jsonify(media_info)
            else:
                 logger.error(f"Failed to get valid media info for target song {target_song_id}")
                 # Fallback: Maybe try the next/prev again or return error
                 return jsonify({"error": f"Could not retrieve info for {action} song"}), 404

    except redis.exceptions.RedisError as e:
         logger.error(f"Redis error accessing history for user {user_id}: {e}")
         return jsonify({"error": "History service error"}), 500
    except Exception as e:
        logger.exception(f"Unexpected error in api_play_sequence for user {user_id}, song {song_id}, action {action}: {e}")
        # Use logger.exception to include traceback
        return jsonify({"error": "An internal error occurred"}), 500

# --- Route: /api/download/<song_id> (Consolidated Download Logic) ---
@bp.route("/api/download/<song_id>")
@login_required
def api_download2(song_id):
    """
    Handles downloading songs. Attempts YouTube download first if ID looks like a video ID.
    Falls back to local file handling if prefixed with 'local-' or if YouTube fails.
    Provides the file as an attachment with a sanitized filename.
    """
    user_id = session['user_id']
    load_local_songs_from_file() # Ensure local song cache is fresh

    # --- Inner function to handle sending the file ---
    def send_the_file(file_path, title, artist, video_id, is_local=False):
        if not os.path.exists(file_path):
             logger.error(f"File path does not exist for download: {file_path}")
             return jsonify({"error": "File not found on server"}), 404

        # Determine download filename
        _, ext = os.path.splitext(file_path) # Get the actual extension
        ext = ext if ext else ".flac" # Default to .flac if no extension found

        # Sanitize title and artist for filename
        safe_title = util.sanitize_filename(title) if title else None
        safe_artist = util.sanitize_filename(artist) if artist else None

        if safe_title and safe_artist:
            download_name = f"{safe_artist} - {safe_title}{ext}"
        elif safe_title:
            download_name = f"{safe_title}{ext}"
        else:
             # Fallback using the video ID
             download_name = f"{video_id}{ext}"

        logger.info(f"Sending file '{file_path}' as download '{download_name}'")
        try:
            return send_file(
                file_path,
                as_attachment=True,
                download_name=download_name,
                mimetype='audio/flac' if ext.lower() == '.flac' else None # Set mimetype for FLAC
            )
        except Exception as send_err:
            logger.error(f"Error sending file {file_path}: {send_err}")
            return jsonify({"error": "Failed to send file"}), 500

    # --- Main Download Logic ---
    is_local_prefix = song_id.startswith("local-")
    potential_vid = song_id[6:] if is_local_prefix else song_id

    # 1. Handle Local Song Request ('local-' prefix)
    if is_local_prefix:
        meta = local_songs.get(song_id)
        if meta and meta.get("path"):
            logger.info(f"Processing download for local song: {song_id}")
            return send_the_file(meta["path"], meta.get("title"), meta.get("artist"), song_id, is_local=True)
        else:
            logger.warning(f"Local song ID {song_id} not found or path missing.")
            # If local fails, should we try treating the rest as a video ID? Maybe not if prefixed.
            return jsonify({"error": "Local song not found"}), 404

    # 2. Handle Non-Local (Assumed YouTube ID)
    else:
        logger.info(f"Processing download for potential YouTube ID: {song_id}")
        # Check if already downloaded (using utility function)
        try:
             existing_download_info = util.get_download_info_for_user(potential_vid, user_id) # Check user-specific download record
             if existing_download_info and os.path.exists(existing_download_info["path"]):
                  logger.info(f"Found existing download for {potential_vid} at {existing_download_info['path']}")
                  # Use metadata from the database record for the filename
                  return send_the_file(
                       existing_download_info["path"],
                       existing_download_info.get("title"),
                       existing_download_info.get("artist"),
                       potential_vid
                  )
        except Exception as db_err:
             logger.error(f"Error checking existing download for {potential_vid}: {db_err}")
             # Continue to download attempt

        # If not found or check failed, attempt download
        try:
            logger.info(f"Attempting to download FLAC for {potential_vid}")
            # download_flac should return the path and potentially metadata
            download_result = util.download_flac(potential_vid, user_id)

            if not download_result or not download_result.get("path"):
                raise Exception("download_flac utility failed to return a valid path.")

            flac_path = download_result["path"]
            # Use metadata returned by download_flac if available, otherwise fetch it
            title = download_result.get("title")
            artist = download_result.get("artist")

            if not title or not artist:
                 logger.warning(f"Metadata not returned by download_flac for {potential_vid}. Fetching separately.")
                 info = get_video_info(potential_vid)
                 title = info.get("title")
                 artist = info.get("artist")

            return send_the_file(flac_path, title, artist, potential_vid)

        except Exception as download_err:
            logger.error(f"Failed to download or process FLAC for {potential_vid}: {download_err}", exc_info=True)
            return jsonify({"error": f"Download failed: {download_err}"}), 500

# --- Route: /health ---
@bp.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint."""
    # Add checks for dependencies like DB, Redis if needed
    db_ok = False
    redis_ok = False
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.cursor().execute("SELECT 1")
        conn.close()
        db_ok = True
    except Exception as e:
        logger.warning(f"Health check DB connection failed: {e}")

    if redis_client:
        try:
            redis_ok = redis_client.ping()
        except Exception as e:
            logger.warning(f"Health check Redis ping failed: {e}")

    status = "healthy" if db_ok and redis_ok else "unhealthy"
    message = f"Server is running. DB: {'OK' if db_ok else 'Error'}, Redis: {'OK' if redis_ok else 'Error'}"

    return jsonify({"status": status, "message": message}), 200 if status == "healthy" else 503

# --- Route: /api/artist-info/<artist_name> ---
@bp.route('/api/artist-info/<artist_name>')
@login_required
def get_artist_info(artist_name):
    """Fetches detailed artist information from YTMusic API."""
    if not ytmusic:
         return jsonify({"error": "Music service integration is unavailable."}), 503

    try:
        # Split multiple artists (e.g., "Artist A, Artist B") and use the primary one
        primary_artist_name = artist_name.split(',')[0].strip()
        if not primary_artist_name:
             return jsonify({"error": "Invalid artist name provided."}), 400

        logger.info(f"Searching for artist info: '{primary_artist_name}'")

        # Search YTMusic specifically for artists
        search_results = ytmusic.search(primary_artist_name, filter='artists', limit=1)

        if not search_results:
            logger.warning(f"No artist found directly for: '{primary_artist_name}'. Trying broader search.")
            # Try a general search and filter results (less reliable)
            general_results = ytmusic.search(primary_artist_name, limit=5)
            artist_results = [r for r in general_results if r.get('resultType') == 'artist'] # Check 'resultType'
            if not artist_results:
                 logger.warning(f"No artist found even with broader search for: '{primary_artist_name}'")
                 # Return minimal info to prevent UI errors, indicate data is missing
                 return jsonify(util.get_fallback_artist_info(primary_artist_name))
            artist_summary = artist_results[0] # Take the first likely match
        else:
            artist_summary = search_results[0]

        artist_id = artist_summary.get('browseId')
        if not artist_id:
            logger.warning(f"No browseId found for artist: '{primary_artist_name}'")
            return jsonify(util.get_fallback_artist_info(primary_artist_name))

        logger.info(f"Found artist ID: {artist_id}. Fetching details.")
        # Get detailed artist data using the browseId
        artist_data = ytmusic.get_artist(artist_id)
        if not artist_data:
            logger.error(f"Failed to fetch detailed artist data for ID: {artist_id}")
            return jsonify(util.get_fallback_artist_info(primary_artist_name))

        # Process the detailed data using utility functions
        description = util.process_description(artist_data.get('description', ''))
        thumbnail_url = util.get_best_thumbnail(artist_data.get('thumbnails', []))
        genres = util.process_genres(artist_data) # Assumes util handles this
        stats = util.get_artist_stats(artist_data) # Assumes util handles this
        top_songs = util.process_top_songs(artist_data) # Assumes util handles this
        links = util.process_artist_links(artist_data, artist_id) # Assumes util handles this

        response_data = {
            'name': artist_data.get('name', primary_artist_name), # Use fetched name if available
            'thumbnail': thumbnail_url,
            'description': description,
            'genres': genres,
            'year': util.extract_year(artist_data), # Assumes util handles this
            'stats': stats,
            'topSongs': top_songs,
            'links': links
        }

        return jsonify(response_data)

    except Exception as e:
        logger.exception(f"Error in get_artist_info for '{artist_name}': {e}")
        # Return minimal fallback info on any unexpected error
        return jsonify(util.get_fallback_artist_info(artist_name.split(',')[0].strip()))

# --- Route: /api/search ---
@bp.route("/api/search")
@login_required
def api_search():
    """
    Handles search requests. Supports text queries, YouTube URLs (songs/playlists),
    and video IDs. Returns combined results from local (Redis) and online sources (yt-dlp/YTMusic).
    Provides default/local songs for empty queries. Uses pagination.
    """
    load_local_songs_from_file() # Ensure local song cache is fresh
    if not redis_client:
         logger.warning("Redis not available for search operations.")
         # Decide how to handle this - maybe only allow online search?
         # For now, proceed but local results will be empty.

    query = request.args.get("q", "").strip()
    try:
        page = int(request.args.get("page", 0))
        limit = int(request.args.get("limit", 20))
        if page < 0 or limit <= 0:
            raise ValueError("Invalid pagination parameters")
    except ValueError:
        return jsonify({"error": "Invalid page or limit parameter"}), 400

    start_index = page * limit
    end_index = start_index + limit

    # --- Helper to get thumbnail from Redis or fallback ---
    def get_redis_thumbnail(song_id_key):
        if redis_client:
            try:
                 # Use HGET for single field retrieval efficiency
                 thumb = redis_client.hget(song_id_key, "thumbnail")
                 if thumb:
                      return thumb
            except redis.exceptions.RedisError as e:
                 logger.error(f"Redis error getting thumbnail for {song_id_key}: {e}")
        # Fallback if Redis fails or no thumbnail stored
        # Determine if it's local or YT based on prefix? Unreliable. Use standard fallback.
        potential_vid_id = song_id_key[6:] if song_id_key.startswith("local-") else song_id_key
        if util.is_potential_video_id(potential_vid_id):
             return f"https://i.ytimg.com/vi/{potential_vid_id}/hqdefault.jpg"
        return None # No fallback if not like a video ID

    # === 1. Handle YouTube URL Input ===
    if "youtube.com" in query or "youtu.be" in query:
        logger.info(f"Processing search query as URL: {query}")
        try:
            # Use extract_playlist_info which handles both playlists and single videos
            playlist_or_video_results = extract_playlist_info(query)
            if playlist_or_video_results:
                 # Apply pagination to the results from the URL
                 paginated_results = playlist_or_video_results[start_index:end_index]
                 # Ensure thumbnails are fetched (they should be included by extract_...)
                 for item in paginated_results:
                      if 'thumbnail' not in item or not item['thumbnail']:
                           item['thumbnail'] = get_redis_thumbnail(item['id']) or util.get_default_thumbnail()
                 return jsonify(paginated_results)
            else:
                 logger.warning(f"Could not extract info from URL: {query}. Falling back to text search.")
                 # Fall through to text search
        except Exception as e:
            logger.error(f"Error processing URL '{query}': {e}. Falling back to text search.")
            # Fall through to text search

    # === 2. Handle Direct YouTube Video ID Input ===
    # Use a robust regex for YouTube video IDs (11 chars, specific character set)
    if re.fullmatch(r'[a-zA-Z0-9_-]{11}', query):
        logger.info(f"Processing search query as Video ID: {query}")
        try:
            # Fetch info using get_video_info helper
            video_info = get_video_info(query)
            if video_info and video_info.get('title') != 'Unknown Title':
                 # Return as a list containing the single result
                 return jsonify([video_info])
            else:
                 logger.warning(f"Could not get info for video ID: {query}")
                 return jsonify([]) # Return empty if ID is invalid or info fetch fails
        except Exception as e:
            logger.error(f"Error processing video ID '{query}': {e}")
            return jsonify([])

    # === 3. Handle Empty Query ===
    if not query:
        logger.info("Processing empty search query. Returning default/local songs.")
        combined_default_results = []
        seen_ids = set()

        # Add local songs from the in-memory cache
        for song in local_songs.values():
            if song["id"] not in seen_ids:
                combined_default_results.append(song)
                seen_ids.add(song["id"])

        # Optionally add hardcoded defaults or popular songs if local is empty
        # default_songs = get_default_songs() # This also uses local_songs currently
        # for song in default_songs: ...

        # Apply pagination
        return jsonify(combined_default_results[start_index:end_index])

    # === 4. Handle Regular Text Search ===
    logger.info(f"Processing text search query: '{query}'")
    combined_text_results = []
    seen_ids = set()

    # Add matching local songs (filter the in-memory cache)
    if local_songs:
         try:
              query_lower = query.lower()
              local_matches = [
                   song for song in local_songs.values()
                   if query_lower in song.get('title', '').lower() or \
                      query_lower in song.get('artist', '').lower() or \
                      query_lower in song.get('album', '').lower()
              ]
              for song in local_matches:
                   if song["id"] not in seen_ids:
                        combined_text_results.append(song)
                        seen_ids.add(song["id"])
         except Exception as e:
              logger.error(f"Error filtering local songs for query '{query}': {e}")

    # --- Perform Online Searches Concurrently ---
    online_results = []
    ytmusic_results = []
    ytdlp_results = []

    def search_ytmusic_task(q):
        if ytmusic:
            try:
                # Search for songs primarily
                return ytmusic.search(q, filter='songs', limit=limit + 5) # Fetch slightly more for merging
            except Exception as e_ytmusic:
                logger.error(f"YTMusic search failed: {e_ytmusic}")
        return []

    def search_ytdlp_task(q):
        try:
            # Use ytsearch prefix for yt-dlp
            ydl_opts = {
                'quiet': True,
                'extract_flat': 'discard_in_playlist', # Get full info per item
                'force_generic_extractor': False,
                'playlistend': limit + 5 # Fetch slightly more
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                 # ytsearchN: searches YouTube and returns N results
                 info = ydl.extract_info(f"ytsearch{limit + 5}:{q}", download=False)
                 if info and 'entries' in info:
                      return [extract_video_info_from_entry(entry) for entry in info['entries'] if entry]
        except Exception as e_ytdlp:
            logger.error(f"yt-dlp search failed: {e_ytdlp}")
        return []

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_ytmusic = executor.submit(search_ytmusic_task, query)
        future_ytdlp = executor.submit(search_ytdlp_task, query)
        ytmusic_results_raw = future_ytmusic.result()
        ytdlp_results_raw = future_ytdlp.result()

    # --- Process and Merge Online Results ---
    # Standardize YTMusic results format
    for item in ytmusic_results_raw:
        if item and item.get('videoId') and item.get('resultType') == 'song':
            artist_name = "Unknown Artist"
            if item.get('artists') and isinstance(item['artists'], list) and item['artists']:
                artist_name = item['artists'][0].get('name', artist_name)
            ytmusic_results.append({
                "id": item['videoId'],
                "title": item.get('title', 'Unknown Title'),
                "artist": artist_name,
                "album": item.get('album', {}).get('name', '') if item.get('album') else '',
                "duration": util.parse_duration(item.get('duration')) if item.get('duration') else 0, # util needed
                "thumbnail": get_best_thumbnail(item.get('thumbnails', [])) or util.get_default_thumbnail()
            })

    # yt-dlp results should already be in the correct format
    ytdlp_results = [item for item in ytdlp_results_raw if item] # Filter out None entries

    # Merge online results (YTMusic preferred)
    for song in ytmusic_results + ytdlp_results:
        if song["id"] not in seen_ids:
            # Ensure essential fields exist
            if not all(k in song for k in ['id', 'title', 'artist', 'thumbnail']):
                 logger.warning(f"Skipping incomplete online search result: {song.get('id')}")
                 continue
            # Add to combined list and track seen IDs
            combined_text_results.append(song)
            seen_ids.add(song["id"])

    # Apply pagination to the final combined list
    return jsonify(combined_text_results[start_index:end_index])


# --- Route: /api/song-info/<song_id> ---
@bp.route("/api/song-info/<song_id>")
@login_required
def api_song_info(song_id):
    """
    Fetch metadata for a single song ID. Checks Redis cache first,
    then local cache, then uses get_media_info (YTMusic API/fallback).
    """
    load_local_songs_from_file() # Ensure local cache is available

    # 1. Check Redis cache (if available)
    if redis_client:
        try:
            # Check if the song ID exists as a hash key in Redis
            if redis_client.exists(song_id):
                 song_data = redis_client.hgetall(song_id)
                 # Basic validation of required fields from Redis data
                 if all(k in song_data for k in ["title", "artist", "thumbnail", "duration"]):
                      logger.info(f"Returning song info for {song_id} from Redis cache.")
                      # Ensure duration is integer
                      try:
                           song_data['duration'] = int(song_data['duration'])
                      except ValueError:
                           song_data['duration'] = 0 # Default if invalid
                      song_data['id'] = song_id # Ensure 'id' is in the response
                      return jsonify(song_data)
                 else:
                      logger.warning(f"Incomplete data in Redis for {song_id}. Fetching fresh.")
            # else: No need for else, just proceed if not found in Redis
        except redis.exceptions.RedisError as e:
            logger.error(f"Redis error fetching song info for {song_id}: {e}")
            # Proceed to fetch fresh data if Redis fails

    # 2. Check if it's a local song (using the in-memory cache)
    if song_id.startswith("local-"):
        meta = local_songs.get(song_id)
        if meta:
            logger.info(f"Returning song info for local song {song_id} from memory cache.")
            return jsonify(meta)
        else:
            # If local prefix but not in cache, it's an error
            logger.error(f"Local song ID {song_id} requested but not found in cache.")
            return jsonify({"error": "Local song not found"}), 404

    # 3. Fetch using get_media_info (which includes YTMusic API and fallbacks)
    logger.info(f"Fetching song info for {song_id} using get_media_info.")
    media_info = get_media_info(song_id)

    if media_info and media_info.get('title') != 'Unknown Title': # Check if fetch was successful
         # Optionally update Redis cache here if desired
         # if redis_client:
         #    try:
         #        redis_client.hset(song_id, mapping=media_info)
         #        redis_client.expire(song_id, CACHE_DURATION) # Set expiry
         #    except redis.exceptions.RedisError as e:
         #        logger.warning(f"Failed to update Redis cache for {song_id}: {e}")
         return jsonify(media_info)
    else:
        logger.error(f"Failed to retrieve valid song info for {song_id} after all attempts.")
        return jsonify({"error": "Failed to retrieve song information"}), 404


# --- Route: /api/get-recommendations/<song_id> ---
@bp.route('/api/get-recommendations/<song_id>')
@login_required
def api_get_recommendations(song_id):
    """
    Get song recommendations based on a given song ID.
    Uses util.get_recommendations_for_song for logic.
    """
    logger.info(f"Fetching recommendations for song: {song_id}")
    try:
        recommendations = util.get_recommendations_for_song(song_id, limit=5) # Assuming util func exists
        if recommendations:
            return jsonify(recommendations)
        else:
            # If util returns empty list or None, provide fallback
            logger.warning(f"No recommendations found for {song_id}. Using fallback.")
            return jsonify(util.get_fallback_recommendations()) # Assuming util func exists
    except Exception as e:
        logger.exception(f"Error getting recommendations for {song_id}: {e}")
        # Provide fallback on any error
        return jsonify(util.get_fallback_recommendations())


# --- Route: /reset_password ---
@bp.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    """Handles the multi-step password reset process (email -> OTP -> new password)."""
    # Use request.args.get for query parameters on GET requests
    initial_error = request.args.get('error')
    initial_success = request.args.get('success')

    if request.method == 'POST':
        step = request.form.get('step', session.get('reset_step')) # Get step from form or session

        if step == 'email':
            email = request.form.get('email')
            if not email or not util.is_valid_email(email): # Add email validation
                return render_template_string(
                    var_templates.RESET_PASSWORD_HTML, step='email', error='Valid email is required'
                )

            conn = None
            try:
                conn = get_main_db()
                c = conn.cursor()
                c.execute('SELECT id FROM users WHERE email = ?', (email,))
                user = c.fetchone()
            except sqlite3.Error as e:
                 logger.error(f"Database error checking email for password reset: {e}")
                 return render_template_string(
                      var_templates.RESET_PASSWORD_HTML, step='email', error='Database error. Please try again later.'
                 )
            finally:
                 if conn: conn.close()

            if not user:
                # Show generic message to avoid revealing existing emails
                logger.warning(f"Password reset attempt for non-existent email: {email}")
                return render_template_string(
                    var_templates.RESET_PASSWORD_HTML, step='email', success='If an account exists, a reset code has been sent.' # Changed from error
                )

            # Send verification code
            try:
                otp = util.generate_otp()
                util.store_otp(email, otp, 'reset') # Ensure purpose matches verification
                # var_templates.send_forgot_password_email(email, otp) # Use appropriate template function
                util.send_email(email, "Sangeet Password Reset Code", f"Your password reset code is: {otp}\n\nThis code will expire in 10 minutes.") # Direct send
                logger.info(f"Sent password reset OTP to {email}")
            except Exception as e:
                 logger.error(f"Failed to send password reset OTP to {email}: {e}")
                 return render_template_string(
                      var_templates.RESET_PASSWORD_HTML, step='email', error='Could not send reset code. Please try again.'
                 )

            session['reset_email'] = email
            session['reset_step'] = 'verify' # Use a distinct session key for step tracking
            session.permanent = True # Make session last longer for multi-step process
            session.modified = True

            return render_template_string(
                var_templates.RESET_PASSWORD_HTML, step='verify', email=email
            )

        elif step == 'verify' and 'reset_email' in session:
            email = session['reset_email']
            otp = request.form.get('otp')
            if not otp:
                 return render_template_string(
                      var_templates.RESET_PASSWORD_HTML, step='verify', email=email, error='Verification code is required.'
                 )

            if not util.verify_otp(email, otp, 'reset'): # Verify using correct purpose
                logger.warning(f"Invalid password reset OTP entered for {email}")
                return render_template_string(
                    var_templates.RESET_PASSWORD_HTML, step='verify', email=email, error='Invalid or expired code.'
                )

            # OTP verified, proceed to password entry step
            logger.info(f"Password reset OTP verified for {email}")
            session['reset_step'] = 'new_password'
            session.modified = True

            return render_template_string(
                var_templates.RESET_PASSWORD_HTML, step='new_password' # No need to pass user_id here yet
            )

        elif step == 'new_password' and 'reset_email' in session and session.get('reset_step') == 'new_password':
            email = session['reset_email']
            new_password = request.form.get('new_password')
            confirm_password = request.form.get('confirm_password')

            if not new_password or not confirm_password:
                 return render_template_string(
                     var_templates.RESET_PASSWORD_HTML, step='new_password', error='Both password fields are required.'
                 )
            if new_password != confirm_password:
                 return render_template_string(
                     var_templates.RESET_PASSWORD_HTML, step='new_password', error='Passwords do not match.'
                 )
            if len(new_password) < 6: # Enforce minimum length
                return render_template_string(
                    var_templates.RESET_PASSWORD_HTML, step='new_password', error='Password must be at least 6 characters.'
                )

            # Hash the new password
            try:
                password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            except Exception as e:
                 logger.error(f"Password hashing failed during reset for {email}: {e}")
                 return render_template_string(
                     var_templates.RESET_PASSWORD_HTML, step='new_password', error='Password update failed. Please try again.'
                 )

            # Update the password in the database
            conn = None
            try:
                conn = get_main_db()
                c = conn.cursor()
                c.execute(
                    'UPDATE users SET password_hash = ? WHERE email = ?',
                    (password_hash, email)
                )
                conn.commit()
                logger.info(f"Password successfully reset for email {email}")
            except sqlite3.Error as e:
                logger.error(f"Database error updating password for {email}: {e}")
                if conn: conn.rollback()
                return render_template_string(
                    var_templates.RESET_PASSWORD_HTML, step='new_password', error='Database error saving new password.'
                )
            finally:
                if conn: conn.close()

            # Clear reset-related session data
            session.pop('reset_step', None)
            session.pop('reset_email', None)
            session.modified = True

            # Redirect to login with success message
            return redirect(url_for('playback.login', success='Password reset successfully. Please log in.'))

        else:
            # Invalid step or missing session data, redirect to start
             logger.warning("Invalid state reached in password reset. Redirecting to start.")
             session.pop('reset_step', None)
             session.pop('reset_email', None)
             session.modified = True
             return redirect(url_for('playback.reset_password', error='An error occurred. Please start over.'))

    # GET request: Show the appropriate step based on session
    current_step = session.get('reset_step', 'email')
    email_for_template = session.get('reset_email') if current_step != 'email' else None

    return render_template_string(
         var_templates.RESET_PASSWORD_HTML,
         step=current_step,
         email=email_for_template,
         error=initial_error, # Pass query param errors/success
         success=initial_success
    )


# --- Route: /forgot_username ---
@bp.route('/forgot_username', methods=['GET', 'POST'])
def forgot_username():
    """Handles the forgot username process (email -> send username)."""
    if request.method == 'POST':
        email = request.form.get('email')
        if not email or not util.is_valid_email(email):
            return render_template_string(
                var_templates.FORGOT_USERNAME_HTML, step='email', error='Valid email is required.'
            )

        conn = None
        username = None
        try:
            conn = get_main_db()
            c = conn.cursor()
            c.execute('SELECT username FROM users WHERE email = ?', (email,))
            user_record = c.fetchone()
            if user_record:
                 username = user_record['username']
        except sqlite3.Error as e:
            logger.error(f"Database error retrieving username for forgot username request: {e}")
            return render_template_string(
                var_templates.FORGOT_USERNAME_HTML, step='email', error='Database error. Please try again.'
            )
        finally:
            if conn: conn.close()

        if not username:
            logger.warning(f"Forgot username attempt for non-existent email: {email}")
            # Return success message regardless to avoid email enumeration
            return render_template_string(
                var_templates.LOGIN_HTML, login_step='initial', success='If an account exists, your username has been sent.'
            )

        # Send username directly to email
        try:
            # Use var_templates function if it exists and handles formatting
            # var_templates.send_forgot_username_email(email, username)
            # Or send directly:
            util.send_email(
                 email,
                 "Your Sangeet Username",
                 f"Hello,\n\nYour username for Sangeet is: {username}\n\nYou can use this to log in."
            )
            logger.info(f"Sent username reminder to {email}")
        except Exception as e:
             logger.error(f"Failed to send username reminder email to {email}: {e}")
             return render_template_string(
                  var_templates.FORGOT_USERNAME_HTML, step='email', error='Could not send username reminder. Please try again.'
             )

        return render_template_string(
            var_templates.LOGIN_HTML, login_step='initial', success='Your username has been sent to your email.'
        )

    # GET request
    return render_template_string(var_templates.FORGOT_USERNAME_HTML, step='email')


# --- Route: /logout ---
@bp.route('/logout')
def logout():
    """Logs the user out by clearing the session and removing the active session record."""
    user_id = session.get('user_id')
    session_token = session.get('session_token')

    if user_id and session_token:
        conn = None
        try:
            conn = get_main_db()
            c = conn.cursor()
            # Delete the specific session token from the database
            c.execute("""
                DELETE FROM active_sessions
                WHERE user_id = ? AND session_token = ?
            """, (user_id, session_token))
            conn.commit()
            deleted_count = c.rowcount
            logger.info(f"Removed session token for user {user_id}. Deleted count: {deleted_count}")
        except sqlite3.Error as e:
            logger.error(f"Database error removing session token during logout for user {user_id}: {e}")
            if conn: conn.rollback()
        finally:
            if conn: conn.close()

    # Clear the Flask session regardless of DB operation success
    session.clear()
    logger.info(f"User {user_id or 'Unknown'} logged out.")
    # Redirect to login page with a logout message
    return redirect(url_for('playback.login', success='You have been logged out.'))


# --- Route: /login ---
@bp.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login, including credential check, session management, and 2FA initiation."""
    # Use request.args.get for query parameters on GET requests
    initial_error = request.args.get('error')
    initial_success = request.args.get('success')

    # --- Check if already logged in with a valid session ---
    if 'user_id' in session and 'session_token' in session:
        conn_check = None
        try:
            conn_check = get_main_db()
            c_check = conn_check.cursor()
            c_check.execute("""
                SELECT 1 FROM active_sessions
                WHERE user_id = ? AND session_token = ?
                AND expires_at > CURRENT_TIMESTAMP
            """, (session['user_id'], session['session_token']))
            valid_session = c_check.fetchone()
            if valid_session:
                logger.info(f"User {session['user_id']} already logged in with valid session. Redirecting home.")
                return redirect(url_for('playback.home'))
            else:
                # Clear invalid/expired session found
                logger.warning(f"User {session['user_id']} had an invalid session cookie. Clearing.")
                session.clear()
        except sqlite3.Error as e:
            logger.error(f"Session check error during login page load: {e}")
            session.clear() # Clear session on DB error
        finally:
            if conn_check: conn_check.close()
    # --- End session check ---

    if request.method == 'POST':
        login_id = request.form.get('login_id', '').strip()
        password = request.form.get('password', '')

        if not login_id or not password:
            return render_template_string(
                var_templates.LOGIN_HTML, login_step='initial', error='Username/Email and Password are required.'
            )

        # Clear any previous temporary login attempts
        session.pop('temp_login', None)
        session.modified = True

        conn = None
        try:
            conn = get_main_db()
            c = conn.cursor()

            # Fetch user details and count existing active sessions
            c.execute("""
                SELECT u.id, u.password_hash, u.twofa_method, u.email,
                       (SELECT COUNT(*) FROM active_sessions a
                        WHERE a.user_id = u.id
                        AND a.expires_at > CURRENT_TIMESTAMP) as active_session_count
                FROM users u
                WHERE u.email = ? OR u.username = ?
            """, (login_id, login_id))
            user_record = c.fetchone() # Fetch as dict because of row_factory

            if user_record and bcrypt.checkpw(password.encode('utf-8'), user_record['password_hash'].encode('utf-8')):
                # --- Password is Correct ---
                user_id = user_record['id']
                twofa_method = user_record['twofa_method']
                email = user_record['email']
                active_session_count = user_record['active_session_count']

                logger.info(f"Password verified for user {user_id} ({login_id}). 2FA: {twofa_method}. Active sessions: {active_session_count}.")

                # Terminate other active sessions for this user if any exist
                if active_session_count > 0:
                     c.execute("DELETE FROM active_sessions WHERE user_id = ?", (user_id,))
                     conn.commit()
                     deleted_count = c.rowcount
                     logger.info(f"Terminated {deleted_count} existing active session(s) for user {user_id}.")

                # --- Handle 2FA ---
                if twofa_method and twofa_method != 'none':
                    login_token = secrets.token_urlsafe(32)
                    session['temp_login'] = {
                        'token': login_token,
                        'user_id': user_id,
                        'twofa_method': twofa_method,
                        'timestamp': time.time() # Add timestamp for expiry check later
                    }
                    session.permanent = True # Keep temp session longer
                    session.modified = True

                    if twofa_method == 'email':
                        try:
                             otp = util.generate_otp()
                             util.store_otp(email, otp, 'login') # Correct purpose
                             util.send_email(email, 'Sangeet Login Verification', f'Your login verification code is: {otp}')
                             logger.info(f"Sent login OTP to {email} for user {user_id}.")
                        except Exception as e:
                             logger.error(f"Failed to send login OTP to {email}: {e}")
                             return render_template_string(
                                  var_templates.LOGIN_HTML, login_step='initial', error='Could not send verification code. Please try again.'
                             )

                    # Render the 2FA verification step
                    return render_template_string(
                        var_templates.LOGIN_HTML,
                        login_step='2fa',
                        login_token=login_token,
                        twofa_method=twofa_method
                    )

                # --- No 2FA: Direct Login ---
                else:
                    session_token = secrets.token_urlsafe(32)
                    # Use timezone-aware datetime for expires_at
                    expires_at = datetime.now(pytz.utc) + timedelta(days=7)

                    # Insert the new active session record
                    c.execute("""
                        INSERT INTO active_sessions (user_id, session_token, expires_at)
                        VALUES (?, ?, ?)
                    """, (user_id, session_token, expires_at))
                    conn.commit()

                    # Set the final user session
                    session.clear() # Clear any temp data first
                    session['user_id'] = user_id
                    session['session_token'] = session_token
                    # session['last_session_check'] = int(time.time()) # Can be used for periodic checks
                    session.permanent = True # Make the session persistent
                    session.modified = True

                    logger.info(f"User {user_id} logged in successfully (no 2FA).")
                    return redirect(url_for('playback.home'))

            else:
                # --- Invalid Credentials ---
                logger.warning(f"Invalid login attempt for '{login_id}'.")
                # Add a small delay to mitigate timing attacks / brute-force
                time.sleep(random.uniform(0.2, 0.5))
                return render_template_string(
                    var_templates.LOGIN_HTML, login_step='initial', error='Invalid username/email or password.'
                )

        except sqlite3.Error as e:
            logger.error(f"Database error during login for '{login_id}': {e}")
            return render_template_string(
                var_templates.LOGIN_HTML, login_step='initial', error='A database error occurred. Please try again later.'
            )
        except Exception as e:
            logger.exception(f"Unexpected error during login for '{login_id}': {e}")
            return render_template_string(
                var_templates.LOGIN_HTML, login_step='initial', error='An unexpected error occurred during login.'
            )
        finally:
            if conn: conn.close()

    # --- GET Request ---
    # Show the initial login form
    return render_template_string(
        var_templates.LOGIN_HTML,
        login_step='initial',
        error=initial_error, # Pass along errors/success from redirects
        success=initial_success
    )

# --- Route: /favicon.ico ---
@bp.route("/favicon.ico")
def favicon():
    """Serves the favicon icon."""
    # It's generally better to serve static files directly via web server (Nginx/Apache)
    # or use Flask's static folder mechanism.
    # return send_file(os.path.join(os.getcwd(), "assets", "favicons", "generic", "favicon.ico"), mimetype='image/vnd.microsoft.icon')
    # For simplicity, returning 204 No Content if you don't have one set up properly
    return make_response('', 204)


# --- Route: /login_verify ---
@bp.route('/login_verify', methods=['POST'])
def login_verify():
    """Handles the verification step of 2FA during login."""
    if 'temp_login' not in session:
        logger.warning("login_verify accessed without temp_login session data.")
        return redirect(url_for('playback.login', error='Session expired. Please log in again.'))

    temp_data = session['temp_login']
    # Check if temp login data has expired (e.g., > 5 minutes old)
    if time.time() - temp_data.get('timestamp', 0) > 300: # 5 minutes expiry
         session.pop('temp_login', None)
         session.modified = True
         logger.warning(f"Temporary login session expired for user {temp_data.get('user_id')}")
         return redirect(url_for('playback.login', error='Verification timed out. Please log in again.'))

    otp_entered = request.form.get('otp')
    token_submitted = request.form.get('login_token') # Get token from form hidden field

    # Validate the submitted token against the one in the session
    if not token_submitted or token_submitted != temp_data.get('token'):
        logger.warning(f"Invalid or missing token submitted during 2FA verification for user {temp_data.get('user_id')}")
        # Potentially clear temp session here? Or just show error.
        return render_template_string(
            var_templates.LOGIN_HTML,
            login_step='2fa', # Show 2FA step again
            login_token=temp_data.get('token'), # Resubmit the same token if session still valid
            twofa_method=temp_data.get('twofa_method'),
            error='Invalid session or token. Please try again.'
        )

    user_id = temp_data['user_id']
    twofa_method = temp_data['twofa_method']

    # --- Verify the OTP based on the method ---
    verification_passed = False
    if twofa_method == 'email':
        if not otp_entered:
             return render_template_string(
                  var_templates.LOGIN_HTML, login_step='2fa', login_token=token_submitted,
                  twofa_method=twofa_method, error='Verification code is required.'
             )
        # Retrieve email associated with user_id
        conn = None
        email = None
        try:
            conn = get_main_db()
            c = conn.cursor()
            c.execute('SELECT email FROM users WHERE id = ?', (user_id,))
            user_record = c.fetchone()
            if user_record:
                 email = user_record['email']
        except sqlite3.Error as e:
             logger.error(f"Database error fetching email for 2FA verification (user {user_id}): {e}")
             # Handle DB error - maybe deny verification
             return render_template_string(
                  var_templates.LOGIN_HTML, login_step='2fa', login_token=token_submitted,
                  twofa_method=twofa_method, error='Database error during verification.'
             )
        finally:
            if conn: conn.close()

        if not email:
             logger.error(f"Could not find email for user {user_id} during 2FA verification.")
             return render_template_string(
                  var_templates.LOGIN_HTML, login_step='2fa', login_token=token_submitted,
                  twofa_method=twofa_method, error='User data error during verification.'
             )

        # Verify the OTP using the utility function
        if util.verify_otp(email, otp_entered, 'login'): # Use correct purpose
            verification_passed = True
            logger.info(f"Email OTP verified successfully for user {user_id}.")
        else:
            logger.warning(f"Invalid email OTP entered for user {user_id}.")
            # Return to 2FA page with error
            return render_template_string(
                var_templates.LOGIN_HTML,
                login_step='2fa',
                login_token=token_submitted,
                twofa_method=twofa_method,
                error='Invalid or expired verification code.'
            )

    # Add logic here for other 2FA methods (e.g., 'totp') if implemented
    # elif twofa_method == 'totp':
    #     # Fetch user's totp_secret from DB
    #     # Verify otp_entered against the secret using a TOTP library
    #     pass

    else:
        # Should not happen if temp_login data is valid
        logger.error(f"Unknown 2FA method encountered during verification: {twofa_method}")
        return redirect(url_for('playback.login', error='Configuration error. Please contact support.'))


    # --- If Verification Passed ---
    if verification_passed:
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now(pytz.utc) + timedelta(days=7) # Use timezone-aware

        conn = None
        try:
            conn = get_main_db()
            c = conn.cursor()
            # Insert the new active session
            c.execute("""
                INSERT INTO active_sessions (user_id, session_token, expires_at)
                VALUES (?, ?, ?)
            """, (user_id, session_token, expires_at))
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database error creating session after 2FA success for user {user_id}: {e}")
            if conn: conn.rollback()
            # Clear temp session and redirect to login with error
            session.pop('temp_login', None)
            session.modified = True
            return redirect(url_for('playback.login', error='Failed to create session after verification.'))
        finally:
            if conn: conn.close()

        # Clear temporary login data and set the final user session
        session.clear() # Clear all old data first
        session['user_id'] = user_id
        session['session_token'] = session_token
        session.permanent = True
        session.modified = True

        logger.info(f"User {user_id} logged in successfully after 2FA verification ({twofa_method}).")
        return redirect(url_for('playback.home'))

    # If verification didn't pass (e.g., logic error or unhandled 2FA method)
    # This path should ideally not be reached if all cases are handled above
    logger.error(f"Verification logic failed unexpectedly for user {user_id}.")
    session.pop('temp_login', None)
    session.modified = True
    return redirect(url_for('playback.login', error='An unexpected error occurred during verification.'))


# --- Route: /terms-register ---
@bp.route('/terms-register', methods=['GET'])
def terms_register():
    """Route to get terms and conditions content for the registration page."""
    terms_file_path = os.path.join(os.getcwd(), "terms", "terms_register.txt")
    try:
        with open(terms_file_path, 'r', encoding='utf-8') as file:
            terms_content = file.read()

        # Format the content with basic HTML structure for embedding
        # You might want to use Markdown conversion or more sophisticated formatting
        formatted_terms = f"""
        <div class="prose prose-sm max-w-none text-gray-300 space-y-3">
            <h4 class="text-lg font-semibold mb-2 text-gray-100">Sangeet Premium Terms of Service</h4>
            {terms_content.replace('\\n', '<br>')}
        </div>
        """
        # Using replace('\n', '<br>') for basic line breaks. Consider a Markdown library for better formatting.
        return formatted_terms
    except FileNotFoundError:
        logger.error(f"Terms file not found at: {terms_file_path}")
        # Fallback message if the terms file is missing
        return "<p class='text-red-400'>Error: Terms and conditions could not be loaded. Please contact support. Do not register if you are unsure about the terms.</p>"
    except Exception as e:
         logger.error(f"Error reading terms file {terms_file_path}: {e}")
         return "<p class='text-red-400'>Error loading terms. Please try again later.</p>"


# --- Route: /register ---
@bp.route('/register', methods=['GET', 'POST'])
def register():
    """Handles the initial step of user registration (collecting details)."""
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        username = request.form.get('username', '').strip()
        full_name = request.form.get('full_name', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        terms_accepted = request.form.get('terms_accepted') == 'on' # Check if checkbox was checked

        # --- Input Validation ---
        error = None
        if not all([email, username, full_name, password, confirm_password]):
            error = 'All fields are required.'
        elif not util.is_valid_email(email):
            error = 'Invalid email format.'
        elif not re.fullmatch(r'[a-zA-Z0-9_]+', username): # Basic username format check
            error = 'Username can only contain letters, numbers, and underscores.'
        elif len(username) < 3:
             error = 'Username must be at least 3 characters.'
        elif password != confirm_password:
            error = 'Passwords do not match.'
        elif len(password) < 6:
            error = 'Password must be at least 6 characters.'
        elif not terms_accepted:
             error = 'You must agree to the Terms of Service to register.'

        if error:
            return render_template_string(
                var_templates.REGISTER_HTML, register_step='initial', error=error,
                # Repopulate form fields on error (except passwords)
                email=email, username=username, full_name=full_name
            )
        # --- End Validation ---

        conn = None
        try:
            conn = get_main_db()
            c = conn.cursor()
            # Check if email or username already exists
            c.execute("SELECT 1 FROM users WHERE email = ? OR username = ?", (email, username))
            if c.fetchone():
                conn.close()
                logger.warning(f"Registration attempt failed: Email '{email}' or Username '{username}' already exists.")
                return render_template_string(
                    var_templates.REGISTER_HTML, register_step='initial',
                    error='Email or Username already exists. Please try logging in or use different details.',
                    email=email, username=username, full_name=full_name
                )

            # --- Proceed with OTP verification step ---
            register_token = secrets.token_urlsafe(32)
            # Store registration details temporarily in session
            session['register_data'] = {
                'token': register_token,
                'email': email,
                'username': username,
                'full_name': full_name,
                'password': password, # Store password temporarily (consider implications) - Hashing later is better.
                'timestamp': time.time()
            }
            session.permanent = True # Make session last longer
            session.modified = True

            # Send verification OTP
            try:
                 otp = util.generate_otp()
                 util.store_otp(email, otp, 'register') # Use correct purpose
                 # var_templates.send_register_otp_email(email, otp) # Use template function if available
                 util.send_email(email, "Verify Your Sangeet Account", f"Your Sangeet verification code is: {otp}")
                 logger.info(f"Sent registration OTP to {email}")
            except Exception as e:
                 logger.error(f"Failed to send registration OTP to {email}: {e}")
                 # Clear potentially sensitive session data before showing error
                 session.pop('register_data', None)
                 session.modified = True
                 return render_template_string(
                      var_templates.REGISTER_HTML, register_step='initial',
                      error='Could not send verification code. Please try registering again.',
                      email=email, username=username, full_name=full_name
                 )

            # Render the OTP verification step
            return render_template_string(
                var_templates.REGISTER_HTML,
                register_step='verify',
                email=email, # Show email being verified
                register_token=register_token # Pass token to form hidden field
            )

        except sqlite3.Error as e:
            logger.error(f"Database error during registration check for {email}/{username}: {e}")
            return render_template_string(
                var_templates.REGISTER_HTML, register_step='initial',
                error='A database error occurred. Please try again later.',
                email=email, username=username, full_name=full_name
            )
        finally:
            if conn: conn.close()

    # --- GET Request ---
    # Clear any stale registration data if visiting the page fresh
    if 'register_data' in session and request.method == 'GET':
         session.pop('register_data', None)
         session.modified = True
    return render_template_string(var_templates.REGISTER_HTML, register_step='initial')


# --- Route: /register/verify ---
@bp.route('/register/verify', methods=['POST'])
def register_verify():
    """Handles the OTP verification step for registration."""
    if 'register_data' not in session:
        logger.warning("register/verify accessed without register_data in session.")
        return redirect(url_for('playback.register', error='Registration session expired. Please start over.'))

    reg_data = session['register_data']
    # Check if registration data has expired (e.g., > 15 minutes old)
    if time.time() - reg_data.get('timestamp', 0) > 900: # 15 minutes expiry
         session.pop('register_data', None)
         session.modified = True
         logger.warning(f"Registration verification session expired for {reg_data.get('email')}")
         return redirect(url_for('playback.register', error='Verification timed out. Please start registration over.'))

    otp_entered = request.form.get('otp')
    token_submitted = request.form.get('register_token') # Get from hidden form field

    # Validate token
    if not token_submitted or token_submitted != reg_data.get('token'):
        logger.warning(f"Invalid token submitted during registration verification for {reg_data.get('email')}")
        # Keep session data for retry, but show error
        return render_template_string(
            var_templates.REGISTER_HTML,
            register_step='verify',
            email=reg_data.get('email'),
            register_token=reg_data.get('token'), # Resubmit correct token
            error='Invalid session or token. Please try again.'
        )

    if not otp_entered:
         return render_template_string(
              var_templates.REGISTER_HTML, register_step='verify',
              email=reg_data.get('email'), register_token=token_submitted,
              error='Verification code is required.'
         )

    # Verify OTP
    email = reg_data['email']
    if not util.verify_otp(email, otp_entered, 'register'): # Use correct purpose
        logger.warning(f"Invalid registration OTP entered for {email}")
        return render_template_string(
            var_templates.REGISTER_HTML,
            register_step='verify',
            email=email,
            register_token=token_submitted,
            error='Invalid or expired verification code.'
        )

    # --- OTP Verified: Create User Account ---
    logger.info(f"Registration OTP verified for {email}. Creating account.")
    conn = None
    try:
        conn = get_main_db()
        c = conn.cursor()

        # Hash the password securely before storing
        password_to_hash = reg_data['password']
        password_hash = bcrypt.hashpw(password_to_hash.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Insert the new user record
        c.execute("""
            INSERT INTO users (email, username, full_name, password_hash, twofa_method)
            VALUES (?, ?, ?, ?, 'none')
        """, (email, reg_data['username'], reg_data['full_name'], password_hash))
        user_id = c.lastrowid # Get the ID of the newly created user

        if not user_id:
             # This shouldn't happen if insert succeeds, but check anyway
             raise sqlite3.Error("Failed to get last row ID after user insertion.")

        # --- Automatically log the user in ---
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now(pytz.utc) + timedelta(days=7) # Use timezone-aware

        # Insert the active session record
        c.execute("""
            INSERT INTO active_sessions (user_id, session_token, expires_at)
            VALUES (?, ?, ?)
        """, (user_id, session_token, expires_at))

        conn.commit()
        logger.info(f"User account created successfully for {email} (ID: {user_id}).")

        # --- Set the final user session ---
        session.pop('register_data', None) # Clear temporary registration data
        session['user_id'] = user_id
        session['session_token'] = session_token
        session.permanent = True
        session.modified = True

        # Redirect to the home page after successful registration and login
        return redirect(url_for('playback.home'))

    except sqlite3.IntegrityError as e:
         # This might happen in a race condition if email/username check passed but insert failed
         logger.error(f"Database integrity error during user creation for {email}: {e}")
         if conn: conn.rollback()
         session.pop('register_data', None) # Clear session data on error
         session.modified = True
         return redirect(url_for('playback.register', error='Username or email already exists. Please try again.'))
    except sqlite3.Error as e:
        logger.error(f"Database error during user creation for {email}: {e}")
        if conn: conn.rollback()
        session.pop('register_data', None)
        session.modified = True
        return redirect(url_for('playback.register', error='Failed to create account due to database error.'))
    except Exception as e:
        logger.exception(f"Unexpected error during user creation/session setup for {email}: {e}")
        if conn: conn.rollback()
        session.pop('register_data', None)
        session.modified = True
        return redirect(url_for('playback.register', error='An unexpected error occurred during registration.'))
    finally:
        if conn: conn.close()


# --- Route: /api/insights ---
@bp.route("/api/insights")
@login_required
def get_insights():
    """Get comprehensive listening insights for the current user from SQLite history."""
    user_id = session['user_id']
    conn = None
    try:
        conn = get_main_db() # Connect to the main DB containing listening_history
        c = conn.cursor()

        # Use utility functions to fetch different insight components
        insights_data = {
            "overview": util.get_overview_stats(c, user_id),
            "recent_activity": util.get_recent_activity(c, user_id),
            "top_artists": util.get_top_artists(c, user_id),
            "listening_patterns": util.get_listening_patterns(c, user_id),
            "completion_rates": util.get_completion_rates(c, user_id)
            # Add more insight categories as needed
        }
        return jsonify(insights_data)

    except sqlite3.Error as e:
         logger.error(f"Database error fetching insights for user {user_id}: {e}")
         return jsonify({"error": "Could not retrieve listening insights due to a database error."}), 500
    except AttributeError as e:
         # Catch errors if utility functions don't exist
         logger.error(f"Missing utility function for insights: {e}")
         return jsonify({"error": "Insight calculation error (missing function)."}), 500
    except Exception as e:
        logger.exception(f"Unexpected error fetching insights for user {user_id}: {e}")
        return jsonify({"error": "An unexpected error occurred while generating insights."}), 500
    finally:
        if conn:
            conn.close()


# --- Route: /api/listen/start ---
@bp.route("/api/listen/start", methods=["POST"])
@login_required
def api_listen_start():
    """Records the start of a listening event in the SQLite database."""
    user_id = session['user_id']
    data = request.json

    # Validate input data
    if not data or not data.get("songId") or not data.get("title") or not data.get("artist"):
        return jsonify({"error": "Missing required song information (songId, title, artist)"}), 400

    song_id = data["songId"]
    title = data["title"]
    artist = data["artist"]
    # Optional fields
    duration = data.get("duration") # Total song duration if available
    album = data.get("album")

    conn = None
    try:
        session_id = util.generate_session_id() # Generate a unique ID for this listening session/group

        conn = get_main_db()
        c = conn.cursor()

        # Insert the initial listening record
        c.execute("""
            INSERT INTO listening_history
                (user_id, song_id, title, artist, album, duration, started_at, session_id, listen_type)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, 'partial')
        """, (user_id, song_id, title, artist, album, duration, session_id))

        listen_id = c.lastrowid # Get the ID of the newly inserted record
        conn.commit()

        logger.info(f"Listen start recorded for user {user_id}, song {song_id}. Listen ID: {listen_id}, Session ID: {session_id}")
        return jsonify({
            "status": "success",
            "listenId": listen_id,
            "sessionId": session_id
        }), 201 # 201 Created

    except sqlite3.Error as e:
        logger.error(f"Database error recording listen start for user {user_id}, song {song_id}: {e}")
        if conn: conn.rollback()
        return jsonify({"error": "Database error recording listen start."}), 500
    except Exception as e:
        logger.exception(f"Unexpected error recording listen start for user {user_id}, song {song_id}: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500
    finally:
        if conn: conn.close()


# --- Route: /api/listen/end ---
@bp.route("/api/listen/end", methods=["POST"])
@login_required
def api_listen_end():
    """Updates the listening record with end time, duration, and completion rate."""
    user_id = session['user_id']
    data = request.json

    # Validate input data
    if not data or 'listenId' not in data:
        return jsonify({"error": "Missing 'listenId'"}), 400

    try:
        listen_id = int(data["listenId"])
    except ValueError:
        return jsonify({"error": "'listenId' must be an integer"}), 400

    # Optional fields from client
    listened_duration_sec = data.get("listenedDuration") # Actual time listened in seconds
    client_duration_sec = data.get("duration") # Total duration known by client

    conn = None
    try:
        conn = get_main_db()
        c = conn.cursor()

        # --- Verify listenId belongs to the current user ---
        c.execute("SELECT user_id, duration, started_at FROM listening_history WHERE id = ?", (listen_id,))
        record = c.fetchone()

        if not record:
             return jsonify({"error": "Invalid listen ID"}), 404
        if record['user_id'] != user_id:
             logger.warning(f"User {user_id} attempted to end listen ID {listen_id} belonging to user {record['user_id']}")
             return jsonify({"error": "Unauthorized action"}), 403
        # --- Verification End ---

        # Use duration from DB if client didn't provide or if DB is more reliable
        db_duration_sec = record['duration']
        final_duration = client_duration_sec if client_duration_sec is not None else db_duration_sec

        # Calculate listened duration if not provided by client (less accurate)
        if listened_duration_sec is None:
             try:
                  start_time = datetime.strptime(record['started_at'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.utc) # Assuming stored as UTC string
                  end_time = datetime.now(pytz.utc)
                  listened_duration_sec = max(0, (end_time - start_time).total_seconds())
             except (ValueError, TypeError) as e:
                  logger.error(f"Error calculating listened duration from timestamps for listen ID {listen_id}: {e}")
                  listened_duration_sec = 0 # Default if calculation fails

        # Ensure listened duration is not negative or larger than total duration
        listened_duration_sec = max(0, listened_duration_sec)
        if final_duration and final_duration > 0:
             listened_duration_sec = min(listened_duration_sec, final_duration)

        # Calculate completion rate
        completion_rate = 0.0
        if final_duration and final_duration > 0:
            completion_rate = round((listened_duration_sec / final_duration), 4)

        # Determine listen type based on completion rate
        listen_type = 'partial'
        if completion_rate >= 0.95: # Threshold for 'full' listen
            listen_type = 'full'
        elif completion_rate <= 0.10: # Threshold for 'skip' (adjust as needed)
             # Could also check if listened_duration_sec is very short (e.g., < 5s)
             listen_type = 'skip'

        # Update the listening history record
        c.execute("""
            UPDATE listening_history SET
                ended_at = CURRENT_TIMESTAMP,
                listened_duration = ?,
                completion_rate = ?,
                listen_type = ?,
                duration = ? -- Update duration if client provided a potentially more accurate one
            WHERE id = ?
        """, (listened_duration_sec, completion_rate, listen_type, final_duration, listen_id))
        conn.commit()

        logger.info(f"Listen end recorded for listen ID {listen_id}. Listened: {listened_duration_sec}s, Rate: {completion_rate}, Type: {listen_type}")
        return jsonify({"status": "success"})

    except sqlite3.Error as e:
        logger.error(f"Database error recording listen end for listen ID {listen_id}: {e}")
        if conn: conn.rollback()
        return jsonify({"error": "Database error recording listen end."}), 500
    except Exception as e:
        logger.exception(f"Unexpected error recording listen end for listen ID {listen_id}: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500
    finally:
        if conn: conn.close()


# --- Route: /api/proxy/image ---
@bp.route("/api/proxy/image")
@login_required
def proxy_image():
    """Proxies image requests from allowed domains to add CORS headers and cache control."""
    image_url = request.args.get('url')
    if not image_url:
        return jsonify({"error": "No URL provided"}), 400

    # --- Validate URL ---
    try:
        parsed_url = urlparse(image_url)
        # Allow specific YouTube image domains
        allowed_domains = {'i.ytimg.com', 'yt3.ggpht.com', 'lh3.googleusercontent.com'} # Add more if needed (e.g., YT Music domains)
        if not parsed_url.scheme in ['http', 'https'] or parsed_url.netloc not in allowed_domains:
             logger.warning(f"Image proxy request blocked for invalid domain: {parsed_url.netloc} (URL: {image_url})")
             return jsonify({"error": "Image source not allowed"}), 403
    except Exception as e:
        logger.error(f"URL parsing error in image proxy: {e} (URL: {image_url})")
        return jsonify({"error": "Invalid image URL format"}), 400
    # --- End Validation ---

    # Fetch the image using the utility function
    try:
        content, content_type = util.fetch_image(image_url) # Assuming util handles requests/errors
        if content and content_type:
            response = make_response(content)
            response.headers['Content-Type'] = content_type
            # Allow requests from any origin (adjust if needed for security)
            response.headers['Access-Control-Allow-Origin'] = '*'
            # Cache publicly for a long time (e.g., 1 year)
            response.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
            return response
        else:
             # util.fetch_image should ideally log the error
             return jsonify({"error": "Failed to fetch image from source"}), 502 # 502 Bad Gateway
    except Exception as e:
        logger.exception(f"Unexpected error in image proxy for URL {image_url}: {e}")
        return jsonify({"error": "Internal proxy error"}), 500


# --- Route: /get-extension ---
@bp.route("/get-extension")
def get_extension():
    """Renders the page providing the browser extension."""
    # Add any dynamic data needed for the template here
    return render_template("extension.html")


# --- Route: /download/extension ---
@bp.route('/download/extension')
def download_extension():
    """Provides the browser extension zip file as a download."""
    extension_dir = os.path.join(os.getcwd(), "payloads", "extension")
    extension_filename = "sangeet-premium.zip"
    extension_path = os.path.join(extension_dir, extension_filename)

    if not os.path.exists(extension_path):
        logger.error(f"Extension file not found at: {extension_path}")
        return "Error: Extension file not found.", 404

    try:
        return send_file(
            extension_path,
            as_attachment=True,
            download_name=extension_filename, # Use the actual filename
            mimetype='application/zip'
        )
    except Exception as e:
        logger.exception(f"Error sending extension file {extension_path}: {e}")
        return "Error serving the extension file.", 500


# --- Route: /sangeet-download/<video_id> ---
@bp.route('/sangeet-download/<video_id>')
@login_required # Ensure user is logged in to access this page
def sangeet_download(video_id):
    """Renders a dedicated download page for a specific video ID."""
    user_id = session['user_id'] # Already confirmed by @login_required

    if not util.is_potential_video_id(video_id):
         return render_template('error.html', message="Invalid Video ID format."), 400

    try:
        # Get metadata using the standard helper function
        info = get_media_info(video_id) # Handles API/fallback

        if not info or info.get('title') == 'Unknown Title':
             logger.warning(f"Could not retrieve info for download page: {video_id}")
             return render_template('error.html', message=f"Could not find information for video ID: {video_id}"), 404

        # Prepare data for the template
        title = info.get("title", "Unknown Title")
        artist = info.get("artist", "Unknown Artist")
        album = info.get("album", "") # Get album if available
        thumbnail = info.get("thumbnail", util.get_default_thumbnail())

        # Construct potential download filename (consistent with download logic)
        safe_title = util.sanitize_filename(title) or "Track"
        dl_name = f"{safe_title}.flac" # Assume FLAC download

        return render_template(
            'download.html',
            title=title,
            artist=artist,
            album=album,
            thumbnail=thumbnail,
            dl_name=dl_name,
            video_id=video_id
        )

    except Exception as e:
        logger.exception(f"Error generating download page for {video_id}: {e}")
        return render_template('error.html', message="An error occurred while preparing the download page."), 500


# --- Route: /download-file/<video_id> ---
@bp.route('/download-file/<video_id>')
@login_required # Ensure user is logged in to trigger download
def download_file(video_id):
    """Handles the actual file download initiated from the /sangeet-download page."""
    user_id = session['user_id']

    if not util.is_potential_video_id(video_id):
         return jsonify({"error": "Invalid Video ID format"}), 400

    # --- Check if file already exists before potentially re-downloading ---
    # Use the consistent MUSIC_PATH_ENV
    flac_path_check = os.path.join(MUSIC_PATH_ENV, f"{video_id}.flac")
    existing_download_info = None
    if os.path.exists(flac_path_check):
        logger.info(f"File {flac_path_check} exists. Fetching DB info for filename.")
        # Fetch info from DB to get potentially better title/artist for filename
        try:
             existing_download_info = util.get_download_info_for_user(video_id, user_id)
        except Exception as db_err:
             logger.error(f"Error fetching DB info for existing file {video_id}: {db_err}")
             # Proceed, but filename might just use video ID
    # --- End check ---

    # --- Download or retrieve existing path ---
    flac_path = None
    download_metadata = {} # To store title/artist from download/DB/API

    if existing_download_info and os.path.exists(existing_download_info["path"]):
        flac_path = existing_download_info["path"]
        download_metadata['title'] = existing_download_info.get("title")
        download_metadata['artist'] = existing_download_info.get("artist")
    else:
        # File doesn't exist or DB info wasn't found, trigger download process
        logger.info(f"File for {video_id} not found or info missing. Triggering download_flac.")
        try:
             download_result = util.download_flac(video_id, user_id)
             if not download_result or not download_result.get("path"):
                  raise Exception("download_flac utility failed.")
             flac_path = download_result["path"]
             download_metadata['title'] = download_result.get("title")
             download_metadata['artist'] = download_result.get("artist")
        except Exception as download_err:
             logger.error(f"download_flac failed for {video_id}: {download_err}", exc_info=True)
             return jsonify({"error": f"Failed to download or process audio: {download_err}"}), 500

    if not flac_path or not os.path.exists(flac_path):
         logger.error(f"FLAC path is invalid or file missing after download attempt: {flac_path}")
         return jsonify({"error": "Audio file not found after processing."}), 404

    # --- Determine Filename ---
    # Prioritize metadata obtained during download/DB check
    title = download_metadata.get('title')
    artist = download_metadata.get('artist')

    # If metadata is still missing, fetch it now
    if not title or not artist:
        logger.warning(f"Fetching metadata for {video_id} again for filename.")
        info = get_video_info(video_id)
        title = info.get('title') if not title else title
        artist = info.get('artist') if not artist else artist

    safe_title = util.sanitize_filename(title) or "Track"
    safe_artist = util.sanitize_filename(artist) if artist else None
    dl_name = f"{safe_artist} - {safe_title}.flac" if safe_artist else f"{safe_title}.flac"

    # --- Send the File ---
    try:
        logger.info(f"Sending file {flac_path} as attachment {dl_name}")
        return send_file(flac_path, as_attachment=True, download_name=dl_name, mimetype='audio/flac')
    except Exception as send_err:
        logger.exception(f"Error sending file {flac_path}: {send_err}")
        return jsonify({"error": "Failed to send the file."}), 500


# --- Route: /data/download/icons/<type> ---
@bp.route("/data/download/icons/<type>")
def icons(type):
    """Serves specific asset files (icons, GIFs) based on type."""
    # Define base directories for assets
    assets_base = os.path.join(os.getcwd(), "assets")
    favicons_base = os.path.join(assets_base, "favicons")
    gifs_base = os.path.join(assets_base, "gifs")

    # Define mappings for icon types to their file paths (using .txt for base64)
    icon_map = {
        "download": os.path.join(favicons_base, "download", "fav.txt"),
        "get-extension": os.path.join(favicons_base, "get-extension", "fav.txt"),
        "login-system-login": os.path.join(favicons_base, "login-system", "login.txt"),
        "login-system-register": os.path.join(favicons_base, "login-system", "register.txt"),
        "login-system-forgot": os.path.join(favicons_base, "login-system", "forgot.txt"),
        "generic": os.path.join(favicons_base, "generic", "fav.txt") # Default icon
    }

    # Define mappings for GIF types
    gif_map = {
        "sangeet-home": os.path.join(gifs_base, "sangeet", "index.gif")
    }

    file_path = None
    is_base64_txt = False
    is_gif = False

    if type in icon_map:
        file_path = icon_map[type]
        is_base64_txt = True
    elif type in gif_map:
         file_path = gif_map[type]
         is_gif = True
    else:
        # Fallback to generic icon if type is unknown
        file_path = icon_map["generic"]
        is_base64_txt = True
        logger.debug(f"Unknown icon type '{type}', falling back to generic.")

    if not file_path or not os.path.exists(file_path):
        logger.error(f"Asset file not found for type '{type}' at path: {file_path}")
        return jsonify({"error": "Asset not found"}), 404

    try:
        if is_base64_txt:
            with open(file_path, "r", encoding="utf-8") as fav:
                data = fav.read().strip()
            return jsonify({"base64": data})
        elif is_gif:
             return send_file(file_path, mimetype='image/gif')
        else:
             # Should not happen based on current logic
             return jsonify({"error": "Unknown asset type configuration"}), 500

    except Exception as e:
        logger.exception(f"Error serving asset for type '{type}': {e}")
        return jsonify({"error": "Failed to serve asset"}), 500


# --- Route: /embed/<song_id> ---
@bp.route("/embed/<song_id>")
def embed_player(song_id):
    """Serves an embeddable HTML player for a specific song."""
    load_local_songs_from_file() # Ensure local song cache is fresh

    try:
        # Get customization options from query parameters
        size = request.args.get("size", "normal").lower() # small, normal, large
        theme = request.args.get("theme", "default").lower() # default, purple, blue, dark
        autoplay = request.args.get("autoplay", "false").lower() == "true"

        valid_sizes = ["small", "normal", "large"]
        valid_themes = ["default", "dark", "purple", "blue"] # Add more themes as needed
        size = size if size in valid_sizes else "normal"
        theme = theme if theme in valid_themes else "default"

        # --- Get Song Info and Stream URL ---
        song_info = None
        stream_url = None

        if song_id.startswith("local-"):
            meta = local_songs.get(song_id)
            if not meta:
                 logger.error(f"Embed requested for non-existent local song: {song_id}")
                 # Render an error template or return 404
                 return render_template("embed_error.html", message="Song not found."), 404
            song_info = {
                "id": song_id,
                "title": meta.get("title", "Unknown Local Song"),
                "artist": meta.get("artist", "Unknown Artist"),
                "thumbnail": meta.get("thumbnail") or url_for('static', filename='images/default-cover.png'), # Use a default image
                "duration": meta.get("duration", 0)
            }
            # URL for the local streaming endpoint
            stream_url = url_for('playback.api_stream_local', song_id=song_id, _external=True) # Use _external=True if needed depending on setup
        else:
            # Assume YouTube ID
            if not util.is_potential_video_id(song_id):
                 return render_template("embed_error.html", message="Invalid song ID format."), 400

            media_info = get_media_info(song_id) # Use the standard info getter
            if not media_info or media_info.get('title') == 'Unknown Title':
                logger.error(f"Embed requested for non-existent/unknown song ID: {song_id}")
                return render_template("embed_error.html", message="Song information not found."), 404

            song_info = media_info # Already has needed keys (id, title, artist, thumbnail, duration)
            # URL for the FLAC streaming endpoint
            # Note: Streaming FLAC might require user login depending on the endpoint's decorators
            stream_url = url_for('playback.stream_file', song_id=song_id, _external=True) # Use _external=True

        # --- Render Embed Template ---
        return render_template(
            "embed.html", # Assumes embed.html exists in templates folder
            song=song_info,
            size=size,
            theme=theme,
            autoplay=autoplay,
            stream_url=stream_url,
            host_url=request.host_url.rstrip('/') # Pass the base URL of the host
        )

    except Exception as e:
        logger.exception(f"Error generating embed player for {song_id}: {e}")
        # Render a generic error within the embed if possible, or return 500
        return render_template("embed_error.html", message="An internal error occurred."), 500


# --- Route: /play/<song_id> ---
@bp.route("/play/<song_id>")
def play_song(song_id):
    """Redirects to the main player page, adding the song_id as a query parameter."""
    # This allows the frontend JS to potentially auto-play the requested song on load.
    logger.info(f"/play route hit for song_id: {song_id}. Redirecting to home.")
    # Ensure the song_id is url-safe, though typical IDs should be fine.
    from urllib.parse import quote
    safe_song_id = quote(song_id)
    return redirect(url_for('playback.home', song=safe_song_id))


# --- Flask Hook: before_request ---
@bp.before_request
def before_request():
    """Actions to perform before each request within this blueprint."""
    # 1. Cleanup expired sessions periodically (can be intensive, consider alternatives)
    # This might be better run as a background task or less frequently.
    # util.cleanup_expired_sessions() # Assuming util function exists

    # 2. Check if the *current* session is valid (if it exists)
    # This duplicates logic in @login_required but catches cases for non-login-required routes
    # that might still benefit from knowing session status.
    # Optimization: Only check if session cookies exist.
    if 'user_id' in session and 'session_token' in session:
        # Check validity less frequently to reduce DB load? e.g., store last check time in session
        last_check = session.get('last_session_check', 0)
        check_interval = 300 # Check every 5 minutes
        now = int(time.time())

        if now - last_check > check_interval:
             conn = None
             try:
                  conn = get_main_db()
                  c = conn.cursor()
                  c.execute("""
                        SELECT 1 FROM active_sessions
                        WHERE user_id = ? AND session_token = ?
                        AND expires_at > CURRENT_TIMESTAMP
                  """, (session['user_id'], session['session_token']))
                  valid_session = c.fetchone()
                  if not valid_session:
                       logger.warning(f"Invalid session detected during before_request check for user {session['user_id']}. Clearing.")
                       session.clear()
                  # Update last check time regardless of validity to avoid rapid checks
                  session['last_session_check'] = now
                  session.modified = True # Ensure session changes are saved
             except sqlite3.Error as e:
                  logger.error(f"Database error during session validity check in before_request: {e}")
                  # Decide action: maybe clear session? or just log?
                  # Clearing might log out users unnecessarily if DB is temporarily down.
             finally:
                  if conn: conn.close()

# --- Route: /api/embed-code/<song_id> ---
@bp.route("/api/embed-code/<song_id>")
def get_embed_code(song_id):
    """Generates the HTML iframe embed code for a given song ID and options."""
    try:
        # Validate song_id format if needed (e.g., check if it's local or YT ID)
        is_local = song_id.startswith("local-")
        if not is_local and not util.is_potential_video_id(song_id):
             return jsonify({"error": "Invalid song ID format"}), 400

        # Get customization options from query parameters
        size = request.args.get("size", "normal").lower()
        theme = request.args.get("theme", "default").lower()
        autoplay = request.args.get("autoplay", "false").lower()

        valid_sizes = ["small", "normal", "large"]
        valid_themes = ["default", "dark", "purple", "blue"]
        size = size if size in valid_sizes else "normal"
        theme = theme if theme in valid_themes else "default"
        autoplay_bool = autoplay == "true" # Convert to boolean for clarity if needed

        # Define dimensions based on size (adjust as needed)
        dimensions = {
            "small": (300, 150),
            "normal": (380, 190),
            "large": (480, 220)
        }
        width, height = dimensions.get(size, dimensions["normal"])

        # Construct the embed URL using url_for for robustness
        embed_url = url_for(
             'playback.embed_player',
             song_id=song_id,
             size=size,
             theme=theme,
             autoplay=autoplay, # Pass string 'true'/'false'
             _external=True # Generate absolute URL
        )

        # Generate the iframe HTML code
        iframe_code = (
            f'<iframe src="{embed_url}" '
            f'width="{width}" height="{height}" '
            f'style="border:none; overflow:hidden;" ' # Added basic styling
            f'allow="autoplay; encrypted-media" ' # Standard permissions
            f'loading="lazy">' # Add lazy loading
            f'</iframe>'
        )

        return jsonify({
            "code": iframe_code,
            "url": embed_url,
            "width": width,
            "height": height,
            "options": {"size": size, "theme": theme, "autoplay": autoplay_bool}
        })

    except Exception as e:
        logger.exception(f"Error generating embed code for {song_id}: {e}")
        return jsonify({"error": "Internal server error generating embed code"}), 500


# --- Route: /api/queue ---
@bp.route('/api/queue', methods=['GET'])
@login_required
def api_queue():
    """
    Returns the user's play history from Redis, acting as a playback queue/history.
    Uses pagination and fetches metadata for each song.
    """
    if not redis_client:
        return jsonify({"error": "History service unavailable"}), 503

    user_id = session['user_id']
    try:
        limit = int(request.args.get('limit', 15)) # Default limit
        offset = int(request.args.get('offset', 0))
        if limit <= 0 or offset < 0:
             raise ValueError("Invalid limit or offset")
    except ValueError:
        return jsonify({"error": "Invalid pagination parameters (limit/offset)"}), 400

    history_key = f"user_history:{user_id}"
    history_list = []
    load_local_songs_from_file() # Ensure local songs are loaded for metadata lookup

    try:
        # Fetch history entry IDs (JSON strings) from Redis list with pagination
        # LRANGE end index is inclusive, so use offset + limit - 1
        history_entries_raw = redis_client.lrange(history_key, offset, offset + limit - 1)

        # Process each entry to fetch metadata
        for entry_raw in history_entries_raw:
            try:
                entry_data = json.loads(entry_raw)
                song_id = entry_data.get('song_id')
                if not song_id:
                    logger.warning(f"Skipping history entry with missing song_id for user {user_id}")
                    continue

                # Fetch metadata using get_media_info (checks local cache, then API)
                media_info = get_media_info(song_id)

                # Append structured data to the history list
                history_list.append({
                    "id": song_id,
                    "title": media_info.get("title", "Unknown Title"),
                    "artist": media_info.get("artist", "Unknown Artist"),
                    "album": media_info.get("album", ""),
                    "thumbnail": media_info.get("thumbnail", util.get_default_thumbnail()),
                    "duration": media_info.get("duration", 0),
                    # Include original history data if needed
                    "played_at": entry_data.get("played_at"),
                    "session_id": entry_data.get("session_id"),
                    "sequence_number": entry_data.get("sequence_number")
                })

            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON entry in history queue for user {user_id}")
            except Exception as meta_err:
                logger.error(f"Error fetching metadata for history song {song_id}: {meta_err}", exc_info=False)
                # Optionally append with default data or skip
                history_list.append({
                    "id": song_id, "title": "Error Loading Info", "artist": "", "thumbnail": util.get_default_thumbnail(),
                     # Include original history data if needed
                    "played_at": entry_data.get("played_at"), "session_id": entry_data.get("session_id"), "sequence_number": entry_data.get("sequence_number")
                })

        return jsonify(history_list)

    except redis.exceptions.RedisError as e:
        logger.error(f"Redis error fetching history queue for user {user_id}: {e}")
        return jsonify({"error": "History service error"}), 500
    except Exception as e:
        logger.exception(f"Unexpected error fetching history queue for user {user_id}: {e}")
        return jsonify({"error": "An internal error occurred"}), 500

# --- Route: /api/stats ---
@bp.route("/api/stats")
@login_required
def api_stats():
    """
    Returns various usage statistics for the logged-in user.
    Combines data from Redis (history-based stats) and SQLite (downloads).
    """
    user_id = session['user_id']
    load_local_songs_from_file() # Ensure local song count is accurate

    stats_data = {
        "total_plays": 0,
        "total_listened_time": 0, # Seconds
        "unique_songs_played": 0,
        "total_downloads": 0,
        "download_size_bytes": 0,
        "top_songs": [], # List of {id, title, artist, plays}
        "top_artists": [], # List of {name, plays} - calculated from history
        "favorite_song_id": None, # Most played song ID
        "favorite_artist": None, # Most played artist name
        "local_songs_count": len(local_songs)
    }

    # --- Fetch stats from Redis (History-based) ---
    if redis_client:
        history_key = f"user_history:{user_id}"
        try:
            history_entries_raw = redis_client.lrange(history_key, 0, -1)
            history_entries = [json.loads(entry) for entry in history_entries_raw if entry]

            stats_data["total_plays"] = len(history_entries)

            song_counts = {}
            artist_counts = {}
            unique_song_ids = set()

            # Process history to calculate counts and unique songs
            for entry in history_entries:
                 song_id = entry.get('song_id')
                 if not song_id: continue

                 unique_song_ids.add(song_id)
                 song_counts[song_id] = song_counts.get(song_id, 0) + 1

                 # Attempt to get artist info for artist stats
                 # This requires fetching metadata which can be slow here.
                 # Alternative: Store artist in history record? Or calculate async?
                 # For simplicity now, fetch metadata on demand. Caching in get_media_info helps.
                 media_info = get_media_info(song_id) # Uses cache
                 artist_name = media_info.get("artist", "Unknown Artist")
                 if artist_name != "Unknown Artist": # Avoid counting unknowns
                      artist_counts[artist_name] = artist_counts.get(artist_name, 0) + 1

            stats_data["unique_songs_played"] = len(unique_song_ids)

            # --- Calculate Top Songs ---
            sorted_songs = sorted(song_counts.items(), key=lambda item: item[1], reverse=True)
            if sorted_songs:
                 stats_data["favorite_song_id"] = sorted_songs[0][0]

            for sid, count in sorted_songs[:5]: # Get top 5
                 media_info = get_media_info(sid) # Fetch metadata again (cached)
                 stats_data["top_songs"].append({
                      "id": sid,
                      "title": media_info.get("title", "Unknown Title"),
                      "artist": media_info.get("artist", "Unknown Artist"),
                      "plays": count
                 })

            # --- Calculate Top Artists ---
            sorted_artists = sorted(artist_counts.items(), key=lambda item: item[1], reverse=True)
            if sorted_artists:
                 stats_data["favorite_artist"] = sorted_artists[0][0]

            stats_data["top_artists"] = [{"name": name, "plays": count} for name, count in sorted_artists[:5]] # Top 5

            # Total listened time - requires storing duration in history or fetching and summing.
            # This is likely better handled by the /api/insights endpoint which uses the dedicated history table.
            # Placeholder:
            # stats_data["total_listened_time"] = redis_client.hget(f"user_aggregate_stats:{user_id}", "total_time") or 0

        except redis.exceptions.RedisError as e:
            logger.error(f"Redis error fetching stats for user {user_id}: {e}")
        except json.JSONDecodeError as e:
             logger.error(f"Error decoding JSON from Redis history for user {user_id}: {e}")
        except Exception as e:
            logger.exception(f"Error calculating stats from Redis history for user {user_id}: {e}")

    # --- Fetch Download Stats from SQLite ---
    conn_main = None
    try:
        conn_main = get_main_db()
        c = conn_main.cursor()
        c.execute("SELECT COUNT(*), path FROM user_downloads WHERE user_id = ?", (user_id,))
        download_records = c.fetchall() # Fetch all download records for the user

        stats_data["total_downloads"] = len(download_records)

        total_size = 0
        for record in download_records:
             path = record["path"] # Access by column name due to row_factory
             if path and os.path.exists(path):
                  try:
                       total_size += os.path.getsize(path)
                  except OSError as e:
                       logger.warning(f"Could not get size for download path {path}: {e}")
             elif path:
                  logger.warning(f"Download path {path} for user {user_id} not found on disk.")

        stats_data["download_size_bytes"] = total_size

    except sqlite3.Error as e:
        logger.error(f"Database error fetching download stats for user {user_id}: {e}")
    except Exception as e:
        logger.exception(f"Error calculating download stats for user {user_id}: {e}")
    finally:
        if conn_main: conn_main.close()

    return jsonify(stats_data)


# --- Flask Error Handlers ---
@bp.errorhandler(404)
def not_found(e):
    """Custom 404 error handler."""
    # Check if the request expects JSON
    if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
        return jsonify({"error": "Not Found", "message": str(e)}), 404
    # Otherwise, render an HTML error page
    return render_template("error.html", status_code=404, message="Page Not Found"), 404

@bp.errorhandler(500)
def internal_error(e):
    """Custom 500 error handler."""
    logger.error(f"Internal Server Error: {e}", exc_info=True) # Log the full error
    if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
        return jsonify({"error": "Internal Server Error", "message": "An unexpected error occurred on the server."}), 500
    return render_template("error.html", status_code=500, message="Internal Server Error"), 500

@bp.errorhandler(400)
def bad_request_error(e):
    """Custom 400 error handler."""
    logger.warning(f"Bad Request Error: {e.description}") # Log the specific reason if available
    if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
        return jsonify({"error": "Bad Request", "message": str(e.description or "Invalid request parameters.")}), 400
    return render_template("error.html", status_code=400, message=str(e.description or "Bad Request")), 400

@bp.errorhandler(401)
def unauthorized_error(e):
    """Custom 401 error handler."""
    logger.warning(f"Unauthorized Access Attempt: {request.path}")
    if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
        return jsonify({"error": "Unauthorized", "message": str(e.description or "Authentication is required.")}), 401
    # Redirect to login for HTML requests? Or show error page?
    # return redirect(url_for('playback.login', error='Please log in to access this page.'))
    return render_template("error.html", status_code=401, message=str(e.description or "Authentication Required")), 401

@bp.errorhandler(403)
def forbidden_error(e):
    """Custom 403 error handler."""
    logger.warning(f"Forbidden Access Attempt: User {session.get('user_id')} to {request.path}")
    if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
        return jsonify({"error": "Forbidden", "message": str(e.description or "You do not have permission to access this resource.")}), 403
    return render_template("error.html", status_code=403, message=str(e.description or "Access Denied")), 403


# --- Route: /api/stream/<song_id> ---
@bp.route("/api/stream/<song_id>")
@login_required
def api_stream(song_id):
    """
    Provides the appropriate streaming URL for a song (local or FLAC file).
    Records the song play event in Redis history.
    """
    user_id = session['user_id']
    client_timestamp = request.args.get('timestamp') # Get timestamp if provided by client

    # Record the play event in Redis history using the utility function
    try:
        util.record_song(song_id, user_id, client_timestamp) # Assumes util handles Redis interaction
    except Exception as e:
         # Log error but continue trying to provide stream URL
         logger.error(f"Failed to record song play for {song_id}, user {user_id}: {e}")

    # Determine stream type and URL
    if song_id.startswith("local-"):
        load_local_songs_from_file() # Ensure local cache is populated
        if song_id in local_songs and local_songs[song_id].get("path"):
            logger.info(f"Providing local stream URL for {song_id}")
            # URL points to the endpoint that serves local files directly
            return jsonify({
                "local": True,
                "url": url_for('playback.api_stream_local', song_id=song_id) # Relative URL usually sufficient
            })
        else:
             logger.error(f"Local song {song_id} requested for streaming not found in cache or has no path.")
             return jsonify({"error": "Local song not found or inaccessible"}), 404
    else:
        # Assume YouTube ID - need to ensure FLAC file exists or download it
        logger.info(f"Providing FLAC stream URL for {song_id}")
        # The download_flac utility should handle the download/check logic
        try:
            # download_flac should ideally just return the path if exists, or download then return path
            # It shouldn't *require* download every time if file exists.
            download_result = util.download_flac(song_id, user_id) # Re-check logic in util.download_flac
            if download_result and download_result.get("path"):
                 flac_path = download_result["path"]
                 if os.path.exists(flac_path):
                      # URL points to the endpoint that serves downloaded FLAC files
                      return jsonify({
                           "url": url_for('playback.stream_file', song_id=song_id), # Relative URL
                           "local": False # Indicates it's a processed file, not original local
                      })
                 else:
                      logger.error(f"FLAC path returned by download_flac does not exist: {flac_path}")
                      raise FileNotFoundError("FLAC file missing after download process.")
            else:
                 raise Exception("Failed to obtain FLAC path from download utility.")
        except Exception as e:
             logger.error(f"Failed to get/prepare FLAC stream for {song_id}: {e}", exc_info=True)
             return jsonify({"error": f"Failed to prepare audio stream: {e}"}), 500


# --- Route: /api/download/<song_id> (Duplicate? Keep api_download2 or rename) ---
# This seems like a duplicate of the consolidated `/api/download/<song_id>` (named api_download2).
# It's generally better to have one route. Let's comment this one out and rely on `api_download2`.
# If this was intended for a different purpose, it needs clarification.

# @bp.route("/api/download/<song_id>")
# @login_required
# def api_download(song_id):
#     """Provide the FLAC file as a downloadable attachment."""
#     # ... (Logic is similar to api_download2) ...
#     pass


# --- Route: /api/similar/<song_id> ---
@bp.route("/api/similar/<song_id>")
@login_required
def api_similar(song_id):
    """Get similar songs using YTMusic API, with fallbacks."""
    logger.info(f"Fetching similar songs for: {song_id}")

    if not ytmusic:
         logger.warning("YTMusic client not available for similar songs request.")
         return jsonify(util.get_fallback_recommendations())

    # Don't provide similar songs for local tracks directly
    if song_id.startswith("local-"):
        logger.info("Similar song request for local track, returning fallback.")
        return jsonify(util.get_fallback_recommendations())

    if not util.is_potential_video_id(song_id):
         return jsonify({"error": "Invalid song ID format"}), 400

    try:
        # Use the centralized recommendation function
        recommendations = util.get_recommendations_for_song(song_id, limit=5) # Get 5 similar songs
        if recommendations:
            return jsonify(recommendations)
        else:
            logger.warning(f"No similar songs found via utility for {song_id}. Returning fallback.")
            return jsonify(util.get_fallback_recommendations())

    except Exception as e:
        logger.exception(f"Error getting similar songs for {song_id}: {e}")
        return jsonify(util.get_fallback_recommendations()) # Fallback on error


# --- Route: /api/history/clear ---
@bp.route("/api/history/clear", methods=["POST"])
@login_required
def api_clear_history():
    """Clears the user's listening history from both Redis and SQLite."""
    user_id = session['user_id']
    cleared_redis = False
    cleared_sqlite = False
    errors = []

    # Clear Redis history
    if redis_client:
        history_key = f"user_history:{user_id}"
        try:
            deleted_count = redis_client.delete(history_key)
            logger.info(f"Cleared Redis history for user {user_id}. Keys deleted: {deleted_count}")
            cleared_redis = True
        except redis.exceptions.RedisError as e:
            logger.error(f"Redis error clearing history for user {user_id}: {e}")
            errors.append("Failed to clear Redis history.")
    else:
        logger.warning("Redis client unavailable, skipping Redis history clear.")
        errors.append("History service (Redis) unavailable.")

    # Clear SQLite history
    conn = None
    try:
        conn = get_main_db()
        c = conn.cursor()
        c.execute("DELETE FROM listening_history WHERE user_id = ?", (user_id,))
        conn.commit()
        deleted_rows = c.rowcount
        logger.info(f"Cleared SQLite listening history for user {user_id}. Rows deleted: {deleted_rows}")
        cleared_sqlite = True
    except sqlite3.Error as e:
        logger.error(f"Database error clearing SQLite history for user {user_id}: {e}")
        if conn: conn.rollback()
        errors.append("Failed to clear database history.")
    finally:
        if conn: conn.close()

    if cleared_redis and cleared_sqlite:
        return jsonify({"status": "success", "message": "Listening history cleared."})
    else:
        return jsonify({"status": "partial_error", "errors": errors}), 500


# --- Route: /api/downloads/clear ---
@bp.route("/api/downloads/clear", methods=["POST"])
@login_required
def api_clear_downloads():
    """Clears user's download records from DB and attempts to delete associated files."""
    user_id = session['user_id']
    deleted_files_count = 0
    failed_deletions = []
    db_cleared = False

    conn = None
    try:
        conn = get_main_db()
        c = conn.cursor()

        # Get list of download paths for the user
        c.execute("SELECT id, path FROM user_downloads WHERE user_id = ?", (user_id,))
        downloads_to_delete = c.fetchall() # Fetch as dicts due to row_factory

        logger.info(f"Attempting to clear {len(downloads_to_delete)} download records and files for user {user_id}.")

        # Delete files from disk first
        for download in downloads_to_delete:
             path = download['path']
             if path and os.path.exists(path):
                  try:
                       os.remove(path)
                       logger.info(f"Deleted download file: {path}")
                       deleted_files_count += 1
                  except OSError as e:
                       logger.error(f"Failed to delete download file {path}: {e}")
                       failed_deletions.append(path)
             elif path:
                  logger.warning(f"Download path {path} record exists but file not found on disk.")
                  # Should this path be considered a failed deletion? Maybe not, file is already gone.

        # Delete records from the database
        c.execute("DELETE FROM user_downloads WHERE user_id = ?", (user_id,))
        conn.commit()
        deleted_db_rows = c.rowcount
        db_cleared = True
        logger.info(f"Deleted {deleted_db_rows} download records from DB for user {user_id}.")

    except sqlite3.Error as e:
        logger.error(f"Database error clearing downloads for user {user_id}: {e}")
        if conn: conn.rollback()
        return jsonify({"error": "Database error during download clearing."}), 500
    except Exception as e:
        logger.exception(f"Unexpected error clearing downloads for user {user_id}: {e}")
        return jsonify({"error": "An unexpected error occurred during download clearing."}), 500
    finally:
        if conn: conn.close()

    response = {
        "status": "success" if db_cleared and not failed_deletions else "partial_success",
        "message": f"Downloads cleared. {deleted_files_count} files deleted.",
        "failed_deletions": failed_deletions
    }
    status_code = 200 if db_cleared and not failed_deletions else 207 # 207 Multi-Status

    return jsonify(response), status_code


# --- Route: /api/stream-file/<song_id> ---
@bp.route("/api/stream-file/<song_id>")
# Removed @login_required to allow potentially public embeds/shares to stream
# Add authentication/authorization checks within the function if needed based on context.
def stream_file(song_id):
    """
    Serves a downloaded FLAC file for streaming. Supports range requests for seeking.
    Uses the configured MUSIC_PATH_ENV.
    """
    if not util.is_potential_video_id(song_id): # Basic validation
        return jsonify({"error": "Invalid file identifier"}), 400

    # Construct the full path using the environment variable
    flac_path = os.path.join(MUSIC_PATH_ENV, f"{song_id}.flac")

    if not os.path.exists(flac_path):
        logger.warning(f"Stream request failed: File not found at {flac_path}")
        # Could optionally try to trigger download here if file is missing?
        # Or just return 404. For simplicity, return 404.
        return jsonify({"error": "Audio file not found"}), 404

    try:
        logger.debug(f"Serving FLAC file for streaming: {flac_path}")
        # send_file automatically handles range requests for seeking if possible
        return send_file(
            flac_path,
            mimetype='audio/flac',
            as_attachment=False # Serve inline for streaming
            # conditional=True, # Enable ETag/Last-Modified handling for caching
            # etag=True # Force ETag generation (can be heavy)
        )
    except FileNotFoundError: # Should be caught by os.path.exists, but as safety
         logger.error(f"File not found error during send_file for {flac_path}")
         return jsonify({"error": "Audio file not found"}), 404
    except Exception as e:
        logger.exception(f"Error streaming file {flac_path}: {e}")
        return jsonify({"error": "Failed to stream audio file"}), 500


# --- Route: /api/stream-local/<song_id> ---
@bp.route("/api/stream-local/<song_id>")
# Removed @login_required - Check implications for security if local paths are sensitive.
# Consider adding checks if only logged-in users should stream local files.
def api_stream_local(song_id):
    """Serves a local music file directly for streaming. Supports range requests."""
    if not song_id.startswith("local-"):
        return jsonify({"error": "Invalid identifier for local stream"}), 400

    load_local_songs_from_file() # Ensure cache is fresh
    meta = local_songs.get(song_id)

    if not meta or not meta.get("path"):
        logger.warning(f"Local stream request failed: Metadata or path not found for {song_id}")
        return jsonify({"error": "Local file metadata not found"}), 404

    local_file_path = meta["path"]

    if not os.path.isfile(local_file_path):
        logger.error(f"Local stream request failed: File not found on disk at {local_file_path}")
        # Maybe remove from local_songs cache here if file is missing?
        return jsonify({"error": "Local file not found on disk"}), 404

    try:
        logger.debug(f"Serving local file for streaming: {local_file_path}")
        # Let send_file determine mimetype if possible, provide fallback?
        # Mimetypes library can help: import mimetypes; mimetype, _ = mimetypes.guess_type(local_file_path)
        return send_file(
            local_file_path,
            as_attachment=False # Serve inline
            # conditional=True
        )
    except FileNotFoundError:
         logger.error(f"File not found error during send_file for {local_file_path}")
         return jsonify({"error": "Local file not found"}), 404
    except Exception as e:
        logger.exception(f"Error streaming local file {local_file_path}: {e}")
        return jsonify({"error": "Failed to stream local file"}), 500


# --- Route: /api/lyrics/<song_id> ---
@bp.route("/api/lyrics/<song_id>")
def api_lyrics(song_id):
    """
    Fetches lyrics for a song ID. Checks SQLite cache first, then YTMusic API.
    Handles local song IDs by attempting to extract the original video ID.
    """
    original_song_id = song_id # Keep original for logging/cache key

    # --- Handle Local Song ID ---
    if song_id.startswith("local-"):
        load_local_songs_from_file() # Ensure local cache is populated
        meta = local_songs.get(song_id)
        if meta and meta.get("path"):
            # Attempt to extract video ID from the path/filename stored in Redis/local_songs
            filename = os.path.basename(meta["path"])
            # Assuming filename format is often 'videoId.ext' or 'Artist - Title (videoId).ext'
            # Try extracting 11-char ID from filename base
            base_name, _ = os.path.splitext(filename)
            match = re.search(r'([a-zA-Z0-9_-]{11})', base_name) # Find potential 11-char ID
            if match:
                potential_vid = match.group(1)
                # Basic check if it looks like a valid ID
                if util.is_potential_video_id(potential_vid):
                     song_id = potential_vid # Use the extracted video ID for lyrics search
                     logger.info(f"Extracted potential video ID '{song_id}' for local song '{original_song_id}' for lyrics lookup.")
                else:
                     logger.info(f"Extracted string '{potential_vid}' doesn't look like video ID. No lyrics for local song '{original_song_id}'.")
                     return jsonify([]) # Cannot get lyrics without a valid video ID
            else:
                logger.info(f"Could not extract video ID from filename '{filename}' for local song '{original_song_id}'. No lyrics available.")
                return jsonify([]) # No lyrics if ID extraction fails
        else:
            logger.warning(f"Metadata or path missing for local song '{original_song_id}'. Cannot fetch lyrics.")
            return jsonify([]) # Cannot proceed without path/metadata

    # --- Check SQLite Cache ---
    # Use the potentially updated song_id (extracted video ID or original)
    cached_lyrics = get_cached_lyrics(song_id)
    if cached_lyrics is not None: # Check for None explicitly, empty list is valid cache hit (no lyrics found previously)
        logger.info(f"Returning lyrics for '{song_id}' (original: '{original_song_id}') from cache.")
        return jsonify(cached_lyrics)

    # --- Fetch from YTMusic API ---
    if not ytmusic:
         logger.error("YTMusic client not available for lyrics request.")
         # Cache that lyrics aren't available?
         cache_lyrics(song_id, []) # Cache empty list to prevent repeated API calls
         return jsonify([])

    logger.info(f"Fetching lyrics for '{song_id}' (original: '{original_song_id}') from YTMusic API.")
    try:
        # Step 1: Get watch playlist to find the lyrics browseId
        watch_playlist_data = ytmusic.get_watch_playlist(videoId=song_id)
        lyrics_browse_id = watch_playlist_data.get('lyrics') if watch_playlist_data else None

        if not lyrics_browse_id:
            logger.info(f"No lyrics browseId found in watch playlist for {song_id}.")
            cache_lyrics(song_id, []) # Cache empty list: no lyrics available
            return jsonify([])

        # Step 2: Get lyrics using the browseId
        lyrics_data = ytmusic.get_lyrics(lyrics_browse_id)

        if lyrics_data and 'lyrics' in lyrics_data and lyrics_data['lyrics']:
            lyrics_text = lyrics_data['lyrics']
            # Split into lines, handle potential empty lines
            lines = [line for line in lyrics_text.split('\n') if line.strip()]
            # Optionally add attribution
            lines.append("") # Add a blank line before attribution
            lines.append("Lyrics provided by Sangeet Premium")

            cache_lyrics(song_id, lines) # Cache the fetched lyrics
            logger.info(f"Successfully fetched and cached lyrics for {song_id}.")
            return jsonify(lines)
        else:
            # Lyrics browseId existed, but get_lyrics returned no actual lyrics content
            logger.info(f"get_lyrics call for browseId {lyrics_browse_id} (song {song_id}) returned no lyrics content.")
            cache_lyrics(song_id, []) # Cache empty list: no lyrics content found
            return jsonify([])

    except Exception as e:
        # Catch specific YTMusic errors if possible, e.g., song not found
        logger.error(f"Error fetching lyrics from YTMusic API for {song_id}: {e}", exc_info=True)
        # Don't cache empty on general error, could be temporary issue
        return jsonify({"error": "Failed to fetch lyrics"}), 500


# --- Route: /api/downloads ---
@bp.route("/api/downloads")
@login_required
def api_downloads():
    """Returns a list of the user's downloaded songs from the database, checking file existence."""
    user_id = session['user_id']
    downloaded_items = []
    conn = None

    try:
        conn = get_main_db()
        c = conn.cursor()
        # Fetch download records for the specific user
        c.execute("""
            SELECT video_id, title, artist, album, path, downloaded_at
            FROM user_downloads
            WHERE user_id = ?
            ORDER BY downloaded_at DESC
        """, (user_id,))
        download_records = c.fetchall() # Fetch as dicts

        logger.info(f"Found {len(download_records)} download records for user {user_id}.")

        # Process each record and check file existence
        for record in download_records:
             video_id = record['video_id']
             file_path = record['path']

             # Check if the path stored in DB matches the expected path based on MUSIC_PATH_ENV
             expected_path = os.path.join(MUSIC_PATH_ENV, f"{video_id}.flac")
             actual_path_to_check = file_path if file_path else expected_path # Use DB path if available

             if os.path.exists(actual_path_to_check):
                  # Use metadata from DB record
                  title = record.get('title', 'Unknown Title')
                  artist = record.get('artist', 'Unknown Artist')
                  album = record.get('album', '')
                  # Use standard YT thumbnail URL structure
                  thumbnail = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg" # Or use get_best_thumbnail if needed

                  downloaded_items.append({
                       "id": video_id,
                       "title": title,
                       "artist": artist,
                       "album": album,
                       "downloaded_at": record.get('downloaded_at'),
                       "thumbnail": thumbnail,
                       "path": actual_path_to_check # Optionally include path if needed by frontend
                  })
             else:
                  logger.warning(f"Download record exists for {video_id} (user {user_id}) but file not found at: {actual_path_to_check}. Skipping.")
                  # Optionally, could trigger a cleanup task here to remove the DB record.

        return jsonify(downloaded_items)

    except sqlite3.Error as e:
        logger.error(f"Database error fetching downloads for user {user_id}: {e}")
        return jsonify({"error": "Database error retrieving downloads."}), 500
    except Exception as e:
        logger.exception(f"Unexpected error fetching downloads for user {user_id}: {e}")
        return jsonify({"error": "An unexpected error occurred while fetching downloads."}), 500
    finally:
        if conn: conn.close()


# --- Route: /api/resend-otp ---
@bp.route('/api/resend-otp', methods=['POST'])
def resend_otp():
    """Resends an OTP code for login or registration verification based on session tokens."""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Invalid request data'}), 400

        purpose = None
        email = None
        token_type = None # 'login' or 'register'

        # Check for login verification resend request
        if 'login_token' in data:
            if 'temp_login' not in session:
                return jsonify({'error': 'Login session not found or expired'}), 400
            temp = session['temp_login']
            # Validate token and expiry
            if data['login_token'] != temp.get('token'):
                return jsonify({'error': 'Invalid login token'}), 400
            if time.time() - temp.get('timestamp', 0) > 300: # Check expiry again
                 session.pop('temp_login', None); session.modified = True
                 return jsonify({'error': 'Login verification timed out'}), 400

            # Get email from DB using user_id in temp session
            user_id = temp['user_id']
            conn = None
            try:
                 conn = get_main_db()
                 c = conn.cursor()
                 c.execute('SELECT email FROM users WHERE id = ?', (user_id,))
                 user_record = c.fetchone()
                 if not user_record:
                      raise ValueError(f"User ID {user_id} not found for OTP resend.")
                 email = user_record['email']
            except Exception as e:
                 logger.error(f"Error fetching email for login OTP resend (user {user_id}): {e}")
                 raise # Re-raise to be caught by outer handler
            finally:
                 if conn: conn.close()

            purpose = 'login'
            token_type = 'login'

        # Check for registration verification resend request
        elif 'register_token' in data:
            if 'register_data' not in session:
                return jsonify({'error': 'Registration session not found or expired'}), 400
            reg_data = session['register_data']
            # Validate token and expiry
            if data['register_token'] != reg_data.get('token'):
                return jsonify({'error': 'Invalid registration token'}), 400
            if time.time() - reg_data.get('timestamp', 0) > 900: # Check expiry again
                 session.pop('register_data', None); session.modified = True
                 return jsonify({'error': 'Registration verification timed out'}), 400

            email = reg_data['email']
            purpose = 'register'
            token_type = 'register'

        # Check for password reset resend request (if applicable, based on reset_password logic)
        elif 'reset_token' in data: # Assuming reset flow uses a token passed to template
             if 'reset_email' not in session or session.get('reset_step') != 'verify':
                  return jsonify({'error': 'Password reset session not found or invalid step'}), 400
             # Add token validation if reset flow includes one
             # if data['reset_token'] != session.get('reset_token'): ...

             email = session['reset_email']
             purpose = 'reset'
             token_type = 'reset' # Use purpose directly

        else:
            return jsonify({'error': 'Invalid request type. Missing required token.'}), 400

        # --- Generate and send new OTP ---
        if not email: # Should have been set above if logic is correct
            logger.error(f"Email address missing for OTP resend (purpose: {purpose}).")
            return jsonify({'error': 'Could not determine email address for resend.'}), 500

        try:
            otp = util.generate_otp()
            util.store_otp(email, otp, purpose) # Store with correct purpose
            # Customize subject/body based on purpose
            subject = f"Sangeet New {purpose.capitalize()} Code"
            body = f"Your new Sangeet {purpose} verification code is: {otp}"
            util.send_email(email, subject, body)
            logger.info(f"Resent {purpose} OTP to {email}")
            return jsonify({'status': 'success', 'message': f'New {purpose} code sent.'})
        except Exception as e:
            logger.error(f"Failed to resend {purpose} OTP to {email}: {e}")
            return jsonify({'error': f'Could not resend {purpose} code. Please try again.'}), 500

    except Exception as e:
        # Catch unexpected errors in the main try block
        logger.exception(f"Unexpected error in resend_otp: {e}")
        return jsonify({'error': "An unexpected error occurred during OTP resend."}), 500

# --- Route: /api/session-status ---
@bp.route("/api/session-status")
def check_session_status():
    """
    Checks if the current Flask session corresponds to a valid, active session
    in the database. Used by frontend to detect logouts elsewhere or expirations.
    """
    user_id = session.get('user_id')
    session_token = session.get('session_token')

    if not user_id or not session_token:
        # No session cookies present
        return jsonify({"valid": False, "reason": "no_session"}), 200 # Return 200 OK, but indicate invalid

    conn = None
    try:
        conn = get_main_db()
        c = conn.cursor()

        # Check if this specific session token is valid and not expired
        c.execute("""
            SELECT 1 FROM active_sessions
            WHERE user_id = ? AND session_token = ?
            AND expires_at > CURRENT_TIMESTAMP
        """, (user_id, session_token))
        session_record = c.fetchone()

        if session_record:
            # Session is valid
            return jsonify({"valid": True})
        else:
            # Session token not found or expired in DB
            # Check if *any* other sessions exist for this user to determine reason
            c.execute("""
                SELECT 1 FROM active_sessions
                WHERE user_id = ? AND expires_at > CURRENT_TIMESTAMP
                LIMIT 1
            """, (user_id,))
            other_session_exists = c.fetchone()

            reason = "logged_out_elsewhere" if other_session_exists else "expired_or_logged_out"
            logger.info(f"Session status check for user {user_id}: Invalid ({reason})")
            # Clear the potentially invalid Flask session cookies
            session.clear()
            session.modified = True
            return jsonify({"valid": False, "reason": reason}), 200 # 200 OK, but indicate invalid

    except sqlite3.Error as e:
        logger.error(f"Database error during session status check for user {user_id}: {e}")
        # Return valid=False on DB error, frontend might retry or prompt login
        return jsonify({"valid": False, "reason": "error"}), 503 # 503 Service Unavailable might be appropriate
    except Exception as e:
        logger.exception(f"Unexpected error during session status check for user {user_id}: {e}")
        return jsonify({"valid": False, "reason": "error"}), 500
    finally:
        if conn: conn.close()

# --- Route: /design/<type> ---
@bp.route("/design/<type>")
def design(type):
    """Serves CSS files from the design directory."""
    css_dir = os.path.join(os.getcwd(), "design", "css")
    allowed_files = {
        "index": "index.css",
        "embed": "embed.css",
        "settings" : "settings.css",
        "share" : "share.css",
        "admin-issues": "admin_issues.css"   # Admin issue dashboard style
        # Add other CSS files here as needed
    }

    if type in allowed_files:
        file_path = os.path.join(css_dir, allowed_files[type])
        if os.path.exists(file_path):
            return send_file(file_path, mimetype="text/css")
        else:
            logger.error(f"CSS file not found for type '{type}' at path: {file_path}")
            abort(404) # Use Flask's abort for standard error handling
    else:
        logger.warning(f"Request for unknown design type: {type}")
        abort(404)


# --- Route: /share/open/<media_id> ---
@bp.route('/share/open/<media_id>')
def share(media_id):
    """Renders a share page for a media item if its corresponding file exists."""
    # Determine the expected path (assuming FLAC for non-local)
    if media_id.startswith("local-"):
         load_local_songs_from_file()
         meta = local_songs.get(media_id)
         file_path = meta.get("path") if meta else None
         is_local_file = True
    else:
         file_path = os.path.join(MUSIC_PATH_ENV, f"{media_id}.flac")
         is_local_file = False # It's a downloaded YT file

    if file_path and os.path.exists(file_path):
        # File exists, get metadata to display on the share page
        info = get_media_info(media_id) # Handles both local and YT IDs
        # Construct the share URL pointing back to the main player with the song ID
        # Use url_for for robustness
        player_url = url_for('playback.home', song=media_id, _external=True)

        return render_template(
            'share.html', # Assumes share.html template exists
            share_url=player_url, # URL to share/embed that opens the player
            song_info=info, # Pass the fetched metadata dictionary
            media_id=media_id
            )
    else:
        # File doesn't exist, render a "not found" page
        logger.warning(f"Share request for non-existent media file: {media_id} (Path checked: {file_path})")
        return render_template('share_not_found.html', media_id=media_id), 404


# --- Route: /stream2/open/<media_id> ---
# Note: The name 'stream2' is potentially confusing alongside '/api/stream'. Consider renaming for clarity.
@bp.route('/stream2/open/<media_id>')
def stream2(media_id):
    """Directly streams a local file or redirects to the standard FLAC stream endpoint."""
    if media_id.startswith("local-"):
        # Stream local file directly using the dedicated local stream endpoint
        logger.info(f"Redirecting local stream request from /stream2 to /api/stream-local for {media_id}")
        # Using redirect is often cleaner than duplicating send_file logic
        return redirect(url_for('playback.api_stream_local', song_id=media_id))
        # --- Alternative: Duplicate send_file logic (less DRY) ---
        # load_local_songs_from_file()
        # details = local_songs.get(media_id)
        # if details and details.get("path"):
        #     file_path = details["path"]
        #     if os.path.exists(file_path):
        #         try:
        #             return send_file(file_path)
        #         except Exception as e:
        #              logger.error(f"Error in stream2 sending local file {file_path}: {e}")
        #              return "Error streaming file", 500
        # return "Local file not found", 404
        # --- End Alternative ---
    else:
        # For non-local IDs (YouTube), redirect to the standard FLAC streaming endpoint
        logger.info(f"Redirecting non-local stream request from /stream2 to /api/stream-file for {media_id}")
        return redirect(url_for('playback.stream_file', song_id=media_id))


# ==============================================================================
# Playlist Routes
# ==============================================================================

# --- Helper: Get Playlist DB Connection ---
def get_playlist_db():
     """Gets a connection to the playlist database."""
     conn = sqlite3.connect(PLAYLIST_DB_PATH)
     conn.row_factory = sqlite3.Row # Return rows as dicts
     # Optionally ensure tables exist here if not done at app init
     # cursor = conn.cursor()
     # cursor.execute('''CREATE TABLE IF NOT EXISTS playlists (...)''')
     # cursor.execute('''CREATE TABLE IF NOT EXISTS playlist_songs (...)''')
     # conn.commit()
     return conn

# --- Route: /api/playlists ---
@bp.route('/api/playlists', methods=['GET'])
@login_required
def get_playlists():
    """Fetches all playlists owned by the logged-in user."""
    user_id = session['user_id']
    conn = None
    try:
        conn = get_playlist_db()
        c = conn.cursor()
        # Query playlists and count songs in each using a LEFT JOIN
        c.execute('''
            SELECT p.id, p.name, COUNT(ps.song_id) as song_count
            FROM playlists p
            LEFT JOIN playlist_songs ps ON p.id = ps.playlist_id
            WHERE p.user_id = ?
            GROUP BY p.id, p.name
            ORDER BY p.name ASC
        ''', (user_id,))
        # Convert results to list of dictionaries
        playlists = [dict(row) for row in c.fetchall()]
        return jsonify(playlists)
    except sqlite3.Error as e:
        logger.error(f"Database error fetching playlists for user {user_id}: {e}")
        return jsonify({"error": "Failed to retrieve playlists"}), 500
    finally:
        if conn: conn.close()

# --- Route: /api/playlists/create ---
@bp.route('/api/playlists/create', methods=['POST'])
@login_required
def create_playlist():
    """Creates a new playlist for the logged-in user."""
    user_id = session['user_id']
    data = request.json
    name = data.get('name', '').strip()

    if not name:
        return jsonify({'error': 'Playlist name cannot be empty'}), 400
    # Add length validation?

    conn = None
    try:
        conn = get_playlist_db()
        c = conn.cursor()
        # Generate a unique share_id for potential future sharing
        share_id = secrets.token_urlsafe(16)
        c.execute('''
            INSERT INTO playlists (user_id, name, is_public, share_id, created_at)
            VALUES (?, ?, 0, ?, CURRENT_TIMESTAMP)
        ''', (user_id, name, share_id))
        new_playlist_id = c.lastrowid
        conn.commit()
        logger.info(f"Created new playlist '{name}' (ID: {new_playlist_id}) for user {user_id}")
        # Return the new playlist info
        return jsonify({
             'status': 'success',
             'message': 'Playlist created successfully',
             'playlist': {'id': new_playlist_id, 'name': name, 'song_count': 0} # Include initial info
             }), 201 # 201 Created
    except sqlite3.IntegrityError as e:
         # Handle potential future constraints (e.g., unique name per user?)
         logger.warning(f"Playlist creation failed for user {user_id}, name '{name}': {e}")
         if conn: conn.rollback()
         return jsonify({'error': 'Playlist with this name might already exist or invalid data.'}), 409 # 409 Conflict
    except sqlite3.Error as e:
        logger.error(f"Database error creating playlist for user {user_id}, name '{name}': {e}")
        if conn: conn.rollback()
        return jsonify({'error': 'Failed to create playlist due to database error.'}), 500
    finally:
        if conn: conn.close()

# --- Route: /api/playlists/add_song ---
@bp.route('/api/playlists/add_song', methods=['POST'])
@login_required
def add_song_to_playlist():
    """Adds a song (by ID) to a specified playlist owned by the user."""
    user_id = session['user_id']
    data = request.json
    try:
        playlist_id = int(data.get('playlist_id'))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid playlist ID format'}), 400
    song_id = data.get('song_id') # Can be local- or YT ID

    if not song_id:
        return jsonify({'error': 'Song ID is required'}), 400

    conn = None
    try:
        conn = get_playlist_db()
        c = conn.cursor()

        # --- Verify user owns the playlist ---
        c.execute('SELECT user_id FROM playlists WHERE id = ?', (playlist_id,))
        result = c.fetchone()
        if not result:
             return jsonify({'error': 'Playlist not found'}), 404
        if result['user_id'] != user_id:
             logger.warning(f"User {user_id} attempted to add song to playlist {playlist_id} owned by user {result['user_id']}")
             return jsonify({'error': 'Unauthorized to modify this playlist'}), 403
        # --- End Verification ---

        # Insert the song into the playlist, ignoring if it already exists
        c.execute('''
            INSERT OR IGNORE INTO playlist_songs (playlist_id, song_id, added_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (playlist_id, song_id))
        conn.commit()
        added = c.rowcount > 0 # Check if a row was actually inserted

        logger.info(f"Song '{song_id}' {'added to' if added else 'already exists in'} playlist {playlist_id} for user {user_id}")
        return jsonify({'status': 'success', 'added': added})

    except sqlite3.Error as e:
        logger.error(f"Database error adding song '{song_id}' to playlist {playlist_id}: {e}")
        if conn: conn.rollback()
        return jsonify({'error': 'Failed to add song due to database error.'}), 500
    finally:
        if conn: conn.close()


# --- Route: /api/playlists/<int:playlist_id>/songs ---
@bp.route('/api/playlists/<int:playlist_id>/songs', methods=['GET'])
@login_required
def get_playlist_songs(playlist_id):
    """Fetches all songs within a specific playlist owned by the user."""
    user_id = session['user_id']
    load_local_songs_from_file() # Ensure local cache is available for metadata
    conn = None

    try:
        conn = get_playlist_db()
        c = conn.cursor()

        # --- Verify user owns the playlist ---
        c.execute('SELECT user_id, name FROM playlists WHERE id = ?', (playlist_id,))
        playlist_owner = c.fetchone()
        if not playlist_owner:
            return jsonify({'error': 'Playlist not found'}), 404
        if playlist_owner['user_id'] != user_id:
            logger.warning(f"User {user_id} attempted to access songs in playlist {playlist_id} owned by user {playlist_owner['user_id']}")
            return jsonify({'error': 'Unauthorized to view this playlist'}), 403
        # --- End Verification ---

        # Fetch all song IDs in the playlist, ordered by when they were added
        c.execute('''
            SELECT song_id FROM playlist_songs
            WHERE playlist_id = ?
            ORDER BY added_at ASC
        ''', (playlist_id,))
        song_ids = [row['song_id'] for row in c.fetchall()]

        # Fetch metadata for each song ID
        songs_with_metadata = []
        for song_id in song_ids:
             media_info = get_media_info(song_id) # Uses cache/API/fallback
             songs_with_metadata.append(media_info) # Add the full metadata dict

        return jsonify({
             "playlist_id": playlist_id,
             "playlist_name": playlist_owner['name'],
             "songs": songs_with_metadata
             })

    except sqlite3.Error as e:
        logger.error(f"Database error fetching songs for playlist {playlist_id}: {e}")
        return jsonify({'error': 'Failed to retrieve playlist songs due to database error.'}), 500
    except Exception as e:
        logger.exception(f"Unexpected error fetching songs for playlist {playlist_id}: {e}")
        return jsonify({'error': 'An unexpected error occurred.'}), 500
    finally:
        if conn: conn.close()

# --- Route: /api/playlists/<int:playlist_id>/share ---
@bp.route('/api/playlists/<int:playlist_id>/share', methods=['POST'])
@login_required
def share_playlist(playlist_id):
    """Makes a user's playlist public and returns its shareable ID."""
    user_id = session['user_id']
    conn = None
    try:
        conn = get_playlist_db()
        c = conn.cursor()

        # --- Verify user owns the playlist ---
        c.execute('SELECT share_id, is_public FROM playlists WHERE id = ? AND user_id = ?', (playlist_id, user_id))
        playlist_data = c.fetchone()
        if not playlist_data:
            logger.warning(f"User {user_id} attempted to share non-existent or unowned playlist {playlist_id}")
            return jsonify({'error': 'Playlist not found or you do not own it.'}), 404 # Or 403
        # --- End Verification ---

        share_id = playlist_data['share_id']
        is_already_public = playlist_data['is_public'] == 1

        # Update playlist to be public if it's not already
        if not is_already_public:
             c.execute('UPDATE playlists SET is_public = 1 WHERE id = ?', (playlist_id,))
             conn.commit()
             logger.info(f"Made playlist {playlist_id} public for user {user_id}. Share ID: {share_id}")
        else:
             logger.info(f"Playlist {playlist_id} was already public. Share ID: {share_id}")

        # Generate the full shareable URL
        share_url = url_for('playback.import_shared_playlist', share_id=share_id, _external=True)

        return jsonify({
            'status': 'success',
            'share_id': share_id,
            'share_url': share_url, # Provide the full URL
            'message': f"Playlist is {'now' if not is_already_public else 'already'} public."
            })

    except sqlite3.Error as e:
        logger.error(f"Database error sharing playlist {playlist_id} for user {user_id}: {e}")
        if conn: conn.rollback()
        return jsonify({'error': 'Failed to share playlist due to database error.'}), 500
    finally:
        if conn: conn.close()


# --- Route: /playlists/share/<share_id> ---
@bp.route('/playlists/share/<share_id>', methods=['GET'])
@login_required # User must be logged in to import a playlist
def import_shared_playlist(share_id):
    """Imports a shared public playlist into the logged-in user's account."""
    user_id = session['user_id']
    conn = None

    try:
        conn = get_playlist_db()
        c = conn.cursor()

        # --- Find the public playlist by share_id ---
        c.execute('''
            SELECT id, name, user_id as owner_id
            FROM playlists
            WHERE share_id = ? AND is_public = 1
        ''', (share_id,))
        shared_playlist = c.fetchone()

        if not shared_playlist:
            logger.warning(f"Attempt to import non-existent or private playlist with share_id: {share_id}")
            # Redirect back with an error message
            return redirect(url_for('playback.home', error='Playlist not found or is not public.'))

        original_playlist_id = shared_playlist['id']
        original_name = shared_playlist['name']
        owner_id = shared_playlist['owner_id']

        # Prevent importing own playlist (optional)
        if owner_id == user_id:
             logger.info(f"User {user_id} attempted to import their own playlist {original_playlist_id}. Skipping.")
             return redirect(url_for('playback.home', message='You already own this playlist.'))

        # --- Fetch songs from the original playlist ---
        c.execute('SELECT song_id FROM playlist_songs WHERE playlist_id = ?', (original_playlist_id,))
        song_ids = [row['song_id'] for row in c.fetchall()]

        if not song_ids:
             logger.info(f"Shared playlist {original_playlist_id} ('{original_name}') is empty. Importing empty playlist.")
             # Proceed to create an empty playlist

        # --- Create a new playlist for the importing user ---
        new_playlist_name = f"{original_name} (Imported)" # Add suffix to distinguish
        # Check if user already has a playlist with this exact imported name? Optional.
        new_share_id = secrets.token_urlsafe(16) # New playlist gets its own share_id
        c.execute('''
            INSERT INTO playlists (user_id, name, is_public, share_id, created_at)
            VALUES (?, ?, 0, ?, CURRENT_TIMESTAMP)
        ''', (user_id, new_playlist_name, new_share_id))
        new_playlist_id = c.lastrowid

        # --- Add songs to the new playlist ---
        if song_ids:
            # Prepare list of tuples for executemany
            song_data_to_insert = [(new_playlist_id, song_id) for song_id in song_ids]
            c.executemany('''
                INSERT INTO playlist_songs (playlist_id, song_id, added_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', song_data_to_insert)

        conn.commit()
        logger.info(f"User {user_id} successfully imported playlist {original_playlist_id} ('{original_name}') as new playlist {new_playlist_id} ('{new_playlist_name}') with {len(song_ids)} songs.")

        # Redirect back to home page with success message
        return redirect(url_for('playback.home', success=f'Playlist "{original_name}" imported successfully!'))

    except sqlite3.Error as e:
        logger.error(f"Database error importing shared playlist {share_id} for user {user_id}: {e}")
        if conn: conn.rollback()
        return redirect(url_for('playback.home', error='Failed to import playlist due to a database error.'))
    except Exception as e:
        logger.exception(f"Unexpected error importing shared playlist {share_id} for user {user_id}: {e}")
        return redirect(url_for('playback.home', error='An unexpected error occurred during playlist import.'))
    finally:
        if conn: conn.close()


# --- Route: /api/random-song ---
@bp.route("/api/random-song")
@login_required
def api_random_song():
    """
    Returns metadata for a randomly selected song.
    Prioritizes user's history, falls back to local songs, then hardcoded defaults.
    """
    user_id = session['user_id']
    load_local_songs_from_file() # Ensure local songs are loaded

    # --- Strategy 1: Get a recent song from user's history (Redis) ---
    if redis_client:
        history_key = f"user_history:{user_id}"
        try:
            # Get a few recent entries to pick from, avoiding just the very last one
            recent_entries_raw = redis_client.lrange(history_key, 0, 4) # Get up to 5 recent
            if recent_entries_raw:
                 # Pick a random one from the recent list
                 random_entry_raw = random.choice(recent_entries_raw)
                 entry_data = json.loads(random_entry_raw)
                 song_id = entry_data.get('song_id')
                 if song_id:
                      logger.info(f"Returning random song from recent history for user {user_id}: {song_id}")
                      media_info = get_media_info(song_id)
                      if media_info and media_info.get('title') != 'Unknown Title':
                           return jsonify(media_info)
                      else:
                           logger.warning(f"Failed to get metadata for history song {song_id}. Falling back.")
                 else:
                      logger.warning("History entry missing song_id.")
            # else: No history found, proceed to next strategy
        except redis.exceptions.RedisError as e:
            logger.error(f"Redis error fetching history for random song (user {user_id}): {e}")
        except json.JSONDecodeError as e:
             logger.error(f"Error decoding history entry for random song (user {user_id}): {e}")
        except Exception as e:
             logger.exception(f"Error processing history for random song (user {user_id}): {e}")
             # Fall through on error

    # --- Strategy 2: Get a random song from loaded local songs ---
    if local_songs:
        try:
            random_local_id = random.choice(list(local_songs.keys()))
            logger.info(f"Returning random local song: {random_local_id}")
            return jsonify(local_songs[random_local_id])
        except IndexError:
            logger.warning("Local songs cache is unexpectedly empty.") # Should not happen if local_songs is true
        except Exception as e:
             logger.exception(f"Error selecting random local song: {e}")
             # Fall through

    # --- Strategy 3: Fallback to hardcoded default YouTube IDs ---
    logger.info("Falling back to hardcoded default song for random request.")
    default_song_ids = ["dQw4w9WgXcQ", "kJQP7kiw5Fk", "9bZkp7q19f0", "xvFZjo5PgG0"] # Add a few diverse defaults
    try:
        random_default_id = random.choice(default_song_ids)
        media_info = get_media_info(random_default_id)
        if media_info and media_info.get('title') != 'Unknown Title':
            return jsonify(media_info)
        else:
             # If even fetching default fails, return error
             logger.error(f"Failed to get metadata for fallback default song: {random_default_id}")
             return jsonify({"error": "Failed to retrieve any song information."}), 500
    except Exception as e:
        logger.exception(f"Error fetching hardcoded default song: {e}")
        return jsonify({"error": "Failed to retrieve fallback song."}), 500


# ==============================================================================
# Issue Reporting Routes (User Facing)
# ==============================================================================

# --- Route: /api/report-issue ---
@bp.route('/api/report-issue', methods=['POST'])
@login_required
def report_issue():
    """Allows a logged-in user to submit a new issue report."""
    user_id = session['user_id']
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data provided"}), 400

    topic = data.get('topic', '').strip()
    details = data.get('details', '').strip()

    # Basic validation
    if not topic or not details:
        return jsonify({"error": "Topic and details are required fields."}), 400
    if len(topic) > 100: # Add length limits
        return jsonify({"error": "Topic is too long (max 100 characters)."}), 400
    if len(details) > 2000:
        return jsonify({"error": "Details are too long (max 2000 characters)."}), 400

    conn = None
    try:
        conn = get_issues_db() # Uses helper to get connection and ensure tables
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO user_issues (user_id, topic, details, status, created_at, updated_at)
            VALUES (?, ?, ?, 'Open', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """, (user_id, topic, details))

        issue_id = cursor.lastrowid # Get the ID of the newly created issue
        conn.commit()

        logger.info(f"User {user_id} reported new issue (ID: {issue_id}): '{topic}'")
        return jsonify({
            "success": True,
            "message": "Issue reported successfully.",
            "issue_id": issue_id
        }), 201 # 201 Created

    except sqlite3.Error as e:
        logger.error(f"Database error reporting issue for user {user_id}: {e}")
        if conn: conn.rollback()
        return jsonify({"error": "Failed to report issue due to a database error."}), 500
    except Exception as e:
        logger.exception(f"Unexpected error reporting issue for user {user_id}: {e}")
        return jsonify({"error": "An unexpected error occurred while reporting the issue."}), 500
    finally:
        if conn: conn.close()

# --- Route: /api/user-issues ---
@bp.route('/api/user-issues', methods=['GET'])
@login_required
def get_user_issues():
    """Fetches all issues submitted by the currently logged-in user, including comments."""
    user_id = session['user_id']
    conn = None

    try:
        conn = get_issues_db() # Helper ensures tables exist
        cursor = conn.cursor()

        # --- Fetch issues AND their comments using a LEFT JOIN ---
        # Order by issue creation date (newest first), then comment date (oldest first within issue)
        cursor.execute("""
            SELECT
                i.id, i.topic, i.details, i.status, i.created_at, i.updated_at,
                c.id as comment_id, c.comment, c.is_admin, c.user_id as comment_user_id, c.created_at as comment_created_at
            FROM user_issues i
            LEFT JOIN issue_comments c ON i.id = c.issue_id
            WHERE i.user_id = ?
            ORDER BY i.created_at DESC, c.created_at ASC
        """, (user_id,))

        rows = cursor.fetchall() # Fetch all rows (as dicts due to row_factory)

        # --- Process rows into a structured list of issues with comments ---
        issues_dict = {}
        for row in rows:
            issue_id = row['id']

            # Initialize issue entry if first time seeing it
            if issue_id not in issues_dict:
                issues_dict[issue_id] = {
                    'id': issue_id,
                    'topic': row['topic'],
                    'details': row['details'],
                    'status': row['status'],
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at'],
                    'comments': [] # Initialize empty comments list
                }

            # Add comment if present in the current row
            if row['comment_id'] is not None:
                 comment_data = {
                     'id': row['comment_id'],
                     'content': row['comment'],
                     'is_admin': bool(row['is_admin']), # Ensure boolean type
                     'is_own_comment': row['comment_user_id'] == user_id and not row['is_admin'], # Flag if user wrote this comment
                     'created_at': row['comment_created_at']
                 }
                 # Avoid duplicates if JOIN behaves unexpectedly (though ORDER BY should prevent it)
                 if not any(c['id'] == comment_data['id'] for c in issues_dict[issue_id]['comments']):
                      issues_dict[issue_id]['comments'].append(comment_data)

        # Convert the dictionary back to a list, maintaining original sort order (newest issues first)
        issues_list = list(issues_dict.values())

        return jsonify(issues_list)

    except sqlite3.Error as e:
         logger.error(f"Database error retrieving issues for user {user_id}: {e}")
         return jsonify({"error": "Database error fetching your reported issues."}), 500
    except Exception as e:
        logger.exception(f"Error retrieving issues for user {user_id}: {e}")
        return jsonify({"error": "An unexpected error occurred fetching your issues."}), 500
    finally:
        if conn: conn.close()

# --- Route: /api/issues/<int:issue_id>/reply ---
@bp.route('/api/issues/<int:issue_id>/reply', methods=['POST'])
@login_required
def reply_to_issue(issue_id):
    """Allows a logged-in user to add a comment to their own issue report."""
    user_id = session['user_id']
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data provided"}), 400

    comment_text = data.get('comment', '').strip()
    if not comment_text:
        return jsonify({"error": "Comment text cannot be empty"}), 400
    if len(comment_text) > 1000: # Add length limit
         return jsonify({"error": "Comment is too long (max 1000 characters)."}), 400

    conn = None
    try:
        conn = get_issues_db()
        cursor = conn.cursor()

        # --- Security Check: Verify the user owns this issue ---
        cursor.execute("SELECT user_id FROM user_issues WHERE id = ?", (issue_id,))
        issue_owner = cursor.fetchone()

        if not issue_owner:
            return jsonify({"error": "Issue not found"}), 404

        if issue_owner['user_id'] != user_id:
            logger.warning(f"User {user_id} attempted to reply to issue {issue_id} owned by user {issue_owner['user_id']}")
            return jsonify({"error": "Unauthorized to comment on this issue"}), 403
        # --- End Security Check ---

        # Insert the user's comment (is_admin is explicitly 0)
        cursor.execute("""
            INSERT INTO issue_comments (issue_id, user_id, is_admin, comment, created_at)
            VALUES (?, ?, 0, ?, CURRENT_TIMESTAMP)
        """, (issue_id, user_id, comment_text))
        new_comment_id = cursor.lastrowid

        # Update the issue's last updated timestamp
        cursor.execute("UPDATE user_issues SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (issue_id,))

        conn.commit()
        logger.info(f"User {user_id} added reply (ID: {new_comment_id}) to their issue {issue_id}")

        # Return success and details of the new comment for frontend update
        new_comment_data = {
             'id': new_comment_id,
             'content': comment_text,
             'is_admin': False,
             'is_own_comment': True,
             'created_at': datetime.now(pytz.utc).isoformat() # Provide timestamp
        }
        return jsonify({
            "success": True,
            "message": "Reply submitted successfully.",
            "comment": new_comment_data
            }), 201 # 201 Created

    except sqlite3.Error as e:
        logger.error(f"Database error adding user reply to issue {issue_id}: {e}")
        if conn: conn.rollback()
        return jsonify({"error": "Database error submitting reply."}), 500
    except Exception as e:
        logger.exception(f"Error adding user reply to issue {issue_id}: {e}")
        return jsonify({"error": "An unexpected error occurred submitting reply."}), 500
    finally:
        if conn: conn.close()


# ==============================================================================
# Admin Issue Management Routes
# ==============================================================================

# --- Route: /view/issues (Admin Login/Dashboard) ---
@bp.route('/view/issues', methods=['GET', 'POST'])
def view_admin_issues():
    """Handles admin login via password and displays the admin issue dashboard."""
    error = request.args.get('error') # Get potential errors from redirects (e.g., auth_required)
    session_expired = request.args.get('session_expired') # Check if redirected due to expired session

    # Handle Session Expiry Message
    if session_expired:
         error = "Admin session expired or logged out. Please re-enter password."
         session.pop('admin_authed', None) # Ensure flag is cleared

    # Handle POST request (Password Submission)
    if request.method == 'POST':
        password_attempt = request.form.get('password')
        if password_attempt == ADMIN_PASSWORD:
            session.permanent = True # Make admin session persistent
            session['admin_authed'] = True
            session.modified = True
            logger.info("Admin login successful.")
            # Redirect to the GET version of this route to display the dashboard
            return redirect(url_for('playback.view_admin_issues'))
        else:
            error = "Invalid Admin Password"
            session.pop('admin_authed', None) # Ensure flag is removed on failure
            logger.warning("Failed admin login attempt.")

    # Handle GET request
    # If admin is authenticated in the session, show the dashboard template
    if session.get('admin_authed'):
        return render_template('admin_issues.html') # Assumes admin_issues.html exists

    # --- If not authenticated (GET or failed POST), show the login prompt ---
    password_prompt_html = """
    <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Admin Login - Sangeet Issues</title>
    <style>
        body { font-family: system-ui, sans-serif; background-color: #111827; color: #d1d5db; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
        .login-container { background-color: #1f2937; padding: 2rem; border-radius: 0.5rem; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05); text-align: center; width: 90%; max-width: 350px; border: 1px solid #374151; }
        h2 { margin-top: 0; color: #f9fafb; font-size: 1.5rem; margin-bottom: 1.5rem; }
        label { display: block; margin-bottom: 0.5rem; font-weight: 500; color: #9ca3af; text-align: left; }
        input[type='password'] { width: calc(100% - 24px); padding: 0.75rem; margin-bottom: 1rem; border: 1px solid #4b5563; border-radius: 0.375rem; background-color: #374151; color: #e5e7eb; font-size: 1rem; }
        input[type='submit'] { background-color: #2563eb; color: white; padding: 0.75rem 1.5rem; border: none; border-radius: 0.375rem; cursor: pointer; font-size: 1rem; font-weight: 500; transition: background-color 0.2s ease; width: 100%; }
        input[type='submit']:hover { background-color: #1d4ed8; }
        .error { color: #f87171; margin-top: 1rem; font-weight: 500; }
    </style>
    </head><body>
    <div class="login-container">
        <h2>Admin Issue Dashboard</h2>
        <form method="post">
            <label for="password">Enter Admin Password:</label>
            <input type="password" id="password" name="password" required autofocus>
            <input type="submit" value="Login">
            {% if error %}
                <p class="error">{{ error }}</p>
            {% endif %}
        </form>
    </div>
    </body></html>
    """
    return render_template_string(password_prompt_html, error=error)


# --- API: Get All Issues (Admin) ---
@bp.route('/api/admin/issues', methods=['GET'])
@admin_required # Apply the admin authentication decorator
def api_get_admin_issues():
    """Fetches all reported issues, joining with user info for display."""
    issues = []
    users_cache = {} # Simple cache for usernames {user_id: username}

    conn_issues = None
    conn_main = None
    try:
        # Fetch all issues from issues DB
        conn_issues = get_issues_db()
        cursor_issues = conn_issues.cursor()
        cursor_issues.execute("""
            SELECT id, user_id, topic, details, status, created_at, updated_at
            FROM user_issues
            ORDER BY status ASC, created_at DESC
        """) # Order by status then date
        all_issues_raw = cursor_issues.fetchall()

        # Get unique user IDs from the fetched issues
        user_ids = list(set(issue['user_id'] for issue in all_issues_raw if issue['user_id']))

        # Fetch corresponding usernames from main DB if any user IDs exist
        if user_ids:
             conn_main = get_main_db()
             cursor_main = conn_main.cursor()
             placeholders = ','.join('?' for _ in user_ids)
             query = f"SELECT id, username FROM users WHERE id IN ({placeholders})"
             cursor_main.execute(query, user_ids)
             user_rows = cursor_main.fetchall()
             users_cache = {user['id']: user['username'] for user in user_rows}

        # Combine issue data with username
        for issue_raw in all_issues_raw:
            issue_dict = dict(issue_raw) # Convert SqLite3.Row to dict
            issue_dict['username'] = users_cache.get(issue_raw['user_id'], 'Unknown User') # Add username
            issues.append(issue_dict)

        return jsonify(issues)

    except sqlite3.Error as e:
         logger.error(f"Database error fetching admin issues: {e}")
         return jsonify({"error": f"Database error: {e}"}), 500
    except Exception as e:
        logger.exception(f"Error fetching admin issues: {e}")
        return jsonify({"error": "An unexpected error occurred fetching issues."}), 500
    finally:
        if conn_issues: conn_issues.close()
        if conn_main: conn_main.close()


# --- API: Get Specific Issue Details (Admin) ---
@bp.route('/api/admin/issues/<int:issue_id>', methods=['GET'])
@admin_required
def api_get_admin_issue_details(issue_id):
    """Fetches full details for a specific issue, including comments and usernames."""
    conn_issues = None
    conn_main = None
    try:
        conn_issues = get_issues_db()
        cursor_issues = conn_issues.cursor()

        # Fetch the specific issue
        cursor_issues.execute("SELECT * FROM user_issues WHERE id = ?", (issue_id,))
        issue = cursor_issues.fetchone()
        if not issue:
            return jsonify({"error": "Issue not found"}), 404

        issue_dict = dict(issue) # Convert to dict

        # Fetch associated comments
        cursor_issues.execute("""
            SELECT id, user_id, is_admin, comment, created_at
            FROM issue_comments
            WHERE issue_id = ?
            ORDER BY created_at ASC
        """, (issue_id,))
        comments_raw = cursor_issues.fetchall()

        # --- Gather User IDs to Fetch Usernames ---
        user_ids_to_fetch = set()
        if issue_dict.get('user_id'):
            user_ids_to_fetch.add(issue_dict['user_id'])
        for comment in comments_raw:
            if comment['user_id'] and not comment['is_admin']:
                 user_ids_to_fetch.add(comment['user_id'])

        # --- Fetch Usernames from Main DB ---
        users_cache = {}
        if user_ids_to_fetch:
            try:
                 conn_main = get_main_db()
                 cursor_main = conn_main.cursor()
                 placeholders = ','.join('?' for _ in user_ids_to_fetch)
                 query = f"SELECT id, username FROM users WHERE id IN ({placeholders})"
                 cursor_main.execute(query, list(user_ids_to_fetch))
                 user_rows = cursor_main.fetchall()
                 users_cache = {user['id']: user['username'] for user in user_rows}
            except sqlite3.Error as e:
                 logger.error(f"Database error fetching usernames for issue details ({issue_id}): {e}")
                 # Proceed without usernames if main DB fails, use IDs or 'Unknown'

        # --- Combine Data ---
        issue_dict['username'] = users_cache.get(issue_dict.get('user_id'), f"User ID: {issue_dict.get('user_id')}" if issue_dict.get('user_id') else 'Unknown')

        comments_list = []
        for comment_raw in comments_raw:
             comment_dict = dict(comment_raw)
             if comment_dict['is_admin']:
                 comment_dict['commenter_display'] = "Admin" # Display name for admin comments
             else:
                 comment_dict['commenter_display'] = users_cache.get(comment_dict.get('user_id'), f"User ID: {comment_dict.get('user_id')}" if comment_dict.get('user_id') else 'Unknown')
             comments_list.append(comment_dict)

        issue_dict['comments'] = comments_list

        return jsonify(issue_dict)

    except sqlite3.Error as e:
        logger.error(f"Database error fetching admin issue details for {issue_id}: {e}")
        return jsonify({"error": f"Database error: {e}"}), 500
    except Exception as e:
        logger.exception(f"Error fetching admin issue details for {issue_id}: {e}")
        return jsonify({"error": "An unexpected error occurred fetching issue details."}), 500
    finally:
        if conn_issues: conn_issues.close()
        if conn_main: conn_main.close()


# --- API: Add Admin Comment ---
@bp.route('/api/admin/issues/<int:issue_id>/comment', methods=['POST'])
@admin_required
def api_add_admin_comment(issue_id):
    """Allows an authenticated admin to add a comment to an issue."""
    data = request.get_json()
    comment_text = data.get('comment', '').strip()

    if not comment_text:
        return jsonify({"error": "Comment text cannot be empty"}), 400
    if len(comment_text) > 1000: # Length limit
         return jsonify({"error": "Comment is too long (max 1000 characters)."}), 400

    conn = None
    try:
        conn = get_issues_db()
        cursor = conn.cursor()

        # Verify issue exists first
        cursor.execute("SELECT id FROM user_issues WHERE id = ?", (issue_id,))
        if not cursor.fetchone():
            return jsonify({"error": "Issue not found"}), 404

        # Insert the admin comment (user_id is NULL, is_admin is 1)
        cursor.execute("""
            INSERT INTO issue_comments (issue_id, user_id, is_admin, comment, created_at)
            VALUES (?, NULL, 1, ?, CURRENT_TIMESTAMP)
        """, (issue_id, comment_text))
        new_comment_id = cursor.lastrowid

        # Update the issue's last updated timestamp
        cursor.execute("UPDATE user_issues SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (issue_id,))

        conn.commit()
        logger.info(f"Admin added comment (ID: {new_comment_id}) to issue {issue_id}")

        # Return the newly added comment data for frontend update
        new_comment_data = {
             'id': new_comment_id,
             'issue_id': issue_id,
             'user_id': None,
             'is_admin': True,
             'comment': comment_text,
             'created_at': datetime.now(pytz.utc).isoformat(),
             'commenter_display': 'Admin'
        }
        return jsonify({"success": True, "message": "Admin comment added.", "comment": new_comment_data}), 201

    except sqlite3.Error as e:
        logger.error(f"Database error adding admin comment to issue {issue_id}: {e}")
        if conn: conn.rollback()
        return jsonify({"error": "Database error adding comment."}), 500
    except Exception as e:
        logger.exception(f"Error adding admin comment to issue {issue_id}: {e}")
        return jsonify({"error": "An unexpected error occurred adding comment."}), 500
    finally:
        if conn: conn.close()


# --- API: Update Issue Status (Admin) ---
@bp.route('/api/admin/issues/<int:issue_id>/update_status', methods=['POST'])
@admin_required
def api_update_issue_status(issue_id):
    """Allows an authenticated admin to update the status of an issue."""
    data = request.get_json()
    new_status = data.get('status', '').strip()

    # Define allowed statuses
    allowed_statuses = ["Open", "In Progress", "Resolved", "Closed"]
    if not new_status or new_status not in allowed_statuses:
        logger.warning(f"Invalid status update attempted for issue {issue_id}: '{new_status}'")
        return jsonify({"error": f"Invalid status. Must be one of: {', '.join(allowed_statuses)}"}), 400

    conn = None
    try:
        conn = get_issues_db()
        cursor = conn.cursor()

        # Update the issue status and timestamp
        cursor.execute("""
            UPDATE user_issues
            SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (new_status, issue_id))

        # Check if the update affected any row (i.e., if the issue exists)
        if cursor.rowcount == 0:
            return jsonify({"error": "Issue not found"}), 404

        conn.commit()
        logger.info(f"Admin updated status of issue {issue_id} to '{new_status}'")
        return jsonify({"success": True, "message": f"Issue status updated to {new_status}"})

    except sqlite3.Error as e:
        logger.error(f"Database error updating status for issue {issue_id}: {e}")
        if conn: conn.rollback()
        return jsonify({"error": "Database error updating status."}), 500
    except Exception as e:
        logger.exception(f"Error updating status for issue {issue_id}: {e}")
        return jsonify({"error": "An unexpected error occurred updating status."}), 500
    finally:
        if conn: conn.close()


# --- API: Get Admin Stats ---
@bp.route('/api/admin/stats', methods=['GET'])
@admin_required
def api_get_admin_stats():
    """Fetches summary statistics about issues for the admin dashboard."""
    conn_issues = None
    conn_main = None
    stats = {
        "total_issues": 0,
        "status_counts": { # Initialize all statuses
             "Open": 0,
             "In Progress": 0,
             "Resolved": 0,
             "Closed": 0
        },
        "total_users": 0,
        # Add more stats as needed: avg resolution time, new issues today, etc.
    }

    try:
        # Get issue stats from issues DB
        conn_issues = get_issues_db()
        cursor_issues = conn_issues.cursor()
        cursor_issues.execute("SELECT status, COUNT(*) as count FROM user_issues GROUP BY status")
        status_rows = cursor_issues.fetchall()
        for row in status_rows:
             if row['status'] in stats["status_counts"]:
                  stats["status_counts"][row['status']] = row['count']
                  stats["total_issues"] += row['count']

        # Get total user count from main DB
        conn_main = get_main_db()
        cursor_main = conn_main.cursor()
        cursor_main.execute("SELECT COUNT(*) as count FROM users")
        user_count_row = cursor_main.fetchone()
        if user_count_row:
            stats["total_users"] = user_count_row['count']

        return jsonify(stats)

    except sqlite3.Error as e:
        logger.error(f"Database error fetching admin stats: {e}")
        # Return partial stats if possible, or error
        return jsonify({"error": f"Database error fetching stats: {e}"}), 500
    except Exception as e:
        logger.exception(f"Error fetching admin stats: {e}")
        return jsonify({"error": "An unexpected error occurred fetching stats."}), 500
    finally:
        if conn_issues: conn_issues.close()
        if conn_main: conn_main.close()


# ==============================================================================
# End of File
# ==============================================================================

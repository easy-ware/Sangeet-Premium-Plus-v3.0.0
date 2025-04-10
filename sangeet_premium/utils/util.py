# util.py
# --- Top Imports ---
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
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Dict, List
import requests
import redis
from dotenv import load_dotenv
from flask import jsonify, redirect, session
from mutagen import File
from mutagen.flac import FLAC
from yt_dlp import YoutubeDL
from ytmusicapi import YTMusic

try:
    from ..helpers import time_helper
    time_sync = time_helper.TimeSync()
except ImportError:
    print("time_helper module not found. Time synchronization features will be limited.")
    class TimeSync:
        def get_current_time(self):
            return datetime.now()
        def parse_datetime(self, time_str):
            return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        def format_time(self, dt, relative=False):
            if relative:
                return "relative time" # Placeholder for relative time formatting
            return dt.strftime('%Y-%m-%d %H:%M:%S')
    time_sync = TimeSync()


# --- Variables ---
load_dotenv(dotenv_path=os.path.join(os.getcwd() , "config" , ".env"))
DB_PATH = os.path.join(os.getcwd() , "database_files" , "sangeet_database_main.db")
MUSIC_DIR = os.getenv("MUSIC_PATH")
FFMPEG_BIN_DIR = os.path.join(os.getcwd(), "ffmpeg", "bin")
LOCAL_SONGS_PATHS = os.getenv("LOCAL_SONGS_PATHS", "")
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
CACHE_DURATION = 3600
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
ytmusic = YTMusic()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
song_cache = {}
search_cache = {}
lyrics_cache = {}
local_songs = {}



# --- Functions ---
def setup_ytdlp():
    """
    Set up yt-dlp executable and return its paths.
    Creates system-specific subdirectories in res folder.
    Checks existing version before downloading.

    Returns:
        tuple: (executable_path: str, version_path: str)
    """
    try:
        # Detect system
        system = platform.system().lower()
        machine = platform.machine().lower()

        print(f"Detected system: {system}")
        print(f"Detected machine architecture: {machine}")

        # More precise architecture mapping
        download_patterns = {
            'aarch64': 'yt-dlp_linux_aarch64',
            'armv7l': 'yt-dlp_linux_armv7l',
            'armv6l': 'yt-dlp_linux_armv7l',
            'x86_64': 'yt-dlp_linux',
            'amd64': 'yt-dlp_linux',
            'i386': 'yt-dlp_linux_x86',
            'i686': 'yt-dlp_linux_x86'
        }

        if system == "windows":
            download_pattern = "yt-dlp.exe"
        elif system == "darwin":
            download_pattern = "yt-dlp_macos"
        else:  # linux
            download_pattern = download_patterns.get(machine)
            if not download_pattern:
                raise Exception(f"Unsupported architecture: {machine}")

        print(f"Selected download pattern: {download_pattern}")

        # Create system-specific directory
        res_dir = Path('res') / system / machine
        res_dir.mkdir(parents=True, exist_ok=True)

        # Set paths
        executable = "yt-dlp.exe" if system == "windows" else "yt-dlp"
        executable_path = res_dir / executable
        version_path = res_dir / "version.txt"

        # Get latest release info
        api_url = "https://api.github.com/repos/yt-dlp/yt-dlp/releases/latest"
        response = requests.get(api_url)
        release_data = response.json()
        latest_version = release_data['tag_name']

        # Check if version file exists and compare versions
        should_download = True
        if executable_path.exists() and version_path.exists():
            with open(version_path, 'r') as f:
                current_version = f.read().strip()
                if current_version == latest_version:
                    print(f"Current version {current_version} matches latest version. Skipping download.")
                    should_download = False
                else:
                    print(f"Update needed: Current version {current_version}, latest version {latest_version}")

        if should_download:
            # Find download URL
            download_url = None
            for asset in release_data['assets']:
                if asset['name'] == download_pattern:
                    download_url = asset['browser_download_url']
                    print(f"Found matching asset: {asset['name']}")
                    break

            if not download_url:
                print("\nAvailable files in release:")
                for asset in release_data['assets']:
                    print(f"- {asset['name']} ({asset['size'] / 1024 / 1024:.1f} MB)")
                raise Exception(f"No executable found for pattern: {download_pattern}")

            print(f"\nDownloading from: {download_url}")

            # Download executable
            response = requests.get(download_url)
            with open(executable_path, 'wb') as f:
                f.write(response.content)

            # Save version information
            with open(version_path, 'w') as f:
                f.write(latest_version)

            # Set executable permissions for non-Windows systems
            if system != "windows":
                executable_path.chmod(executable_path.stat().st_mode | stat.S_IEXEC)

            print(f"Successfully installed version {latest_version} to: {executable_path}")

        return str(executable_path), str(version_path)

    except Exception as e:
        print(f"Error setting up yt-dlp: {e}")
        return None, None

YTDLP_PATH , version_path = setup_ytdlp()

def generate_session_id():
    """Generate a unique session ID for grouping played songs."""
    return f"session_{int(time.time())}"

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or 'session_token' not in session:
            return redirect('/login')

        # Verify session is still valid in database
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute("""
            SELECT 1 FROM active_sessions
            WHERE user_id = ? AND session_token = ?
            AND expires_at > CURRENT_TIMESTAMP
        """, (session['user_id'], session['session_token']))

        valid_session = c.fetchone()
        conn.close()

        if not valid_session:
            # Clear invalid session
            session.clear()
            return redirect("/login")

        return f(*args, **kwargs)
    return decorated_function

@login_required
def record_song(song_id, user_id, client_timestamp=None):
    """Record song play with user association using Redis and client timestamp."""
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

        # Use client-provided timestamp or fallback to server time
        if client_timestamp:
            try:
                played_at = datetime.strptime(client_timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')
            except ValueError:
                played_at = datetime.now()
        else:
            played_at = datetime.now()

        current_time = played_at.strftime('%Y-%m-%d %H:%M:%S')
        history_key = f"user_history:{user_id}"

        # Session continuity logic
        last_entry = redis_client.lindex(history_key, 0)
        if last_entry:
            last_data = json.loads(last_entry)
            last_time = datetime.strptime(last_data['played_at'], '%Y-%m-%d %H:%M:%S')
            if (played_at - last_time).total_seconds() < 3600:
                session_id = last_data['session_id']
                sequence_number = last_data['sequence_number'] + 1
            else:
                session_id = generate_session_id()
                sequence_number = 1
        else:
            session_id = generate_session_id()
            sequence_number = 1

        # Fetch song metadata for thumbnail
        song_data = redis_client.hgetall(song_id)
        thumbnail = song_data.get('thumbnail', f"https://i.ytimg.com/vi/{song_id}/hqdefault.jpg")

        entry = {
            "song_id": song_id,
            "played_at": current_time,
            "session_id": session_id,
            "sequence_number": sequence_number,
            "thumbnail": thumbnail
        }

        redis_client.lpush(history_key, json.dumps(entry))
        redis_client.ltrim(history_key, 0, 999)

        stats_key = f"user_stats:{user_id}"
        redis_client.hincrby(stats_key, "total_plays", 1)
        redis_client.hset(stats_key, "last_played", current_time)

    except Exception as e:
        logger.error(f"Error recording song play: {e}")
        raise
@login_required
def record_download(video_id, title, artist, album, path, user_id):
    """Store a downloaded track with user association in Redis."""
    download_key = f"download:{user_id}:{video_id}"
    redis_client.hset(download_key, mapping={
        "title": title,
        "artist": artist,
        "album": album,
        "path": path,
        "downloaded_at": datetime.now().isoformat()
    })
    redis_client.sadd(f"user_downloads:{user_id}", video_id)

@login_required
def get_play_history(user_id, limit=5, offset=0):
    """Get user's play history optimized for speed using Redis."""
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        history_key = f"user_history:{user_id}"

        # Get paginated history from Redis list
        start = offset
        end = offset + limit - 1
        history_entries = redis_client.lrange(history_key, start, end)

        history = []
        song_ids = []
        for entry in history_entries:
            data = json.loads(entry)
            song_ids.append(data['song_id'])
            history.append({
                "id": data['song_id'],
                "played_at": data['played_at'],
                "session_id": data['session_id'],
                "sequence_number": data['sequence_number']
            })

        if not song_ids:
            return history

        # Fetch metadata from Redis cache
        metadata_cache = {}
        for song_id in song_ids:
            meta_key = f"song_metadata:{song_id}"
            meta = redis_client.hgetall(meta_key)
            if meta:
                metadata_cache[song_id] = {'title': meta['title'], 'artist': meta['artist']}

        # Process each history item
        for item in history:
            song_id = item['id']
            if song_id.startswith("local-"):
                # Handle local songs
                meta = local_songs.get(song_id, {})
                item['title'] = meta.get('title', 'Unknown Title')
                item['artist'] = meta.get('artist', 'Unknown Artist')
                item['thumbnail'] = meta.get('thumbnail', '')
            else:
                # Handle YouTube songs
                if song_id in metadata_cache:
                    meta = metadata_cache[song_id]
                    item['title'] = meta['title']
                    item['artist'] = meta['artist']
                else:
                    # Fetch from API if not cached and store it
                    try:
                        data = ytmusic.get_song(song_id)
                        vd = data.get("videoDetails", {})
                        title = vd.get("title", "Unknown")
                        artist = vd.get("author", "Unknown Artist")
                        redis_client.hset(f"song_metadata:{song_id}", mapping={
                            "title": title,
                            "artist": artist
                        })
                        item['title'] = title
                        item['artist'] = artist
                    except Exception as e:
                        logger.error(f"Error fetching song info for {song_id}: {e}")
                        item['title'] = "Unknown Title"
                        item['artist'] = "Unknown Artist"
                item['thumbnail'] = f"https://i.ytimg.com/vi/{song_id}/default.jpg"

            # Convert UTC to IST
            item['played_at'] = time_sync.format_time(item['played_at'])

        return history

    except Exception as e:
        logger.error(f"Error getting play history: {e}")
        return []


def init_db_local():
    """
    Initialize (or create if needed) the SQLite database.
    The database file is stored in a subfolder called "locals".
    """
    db_dir = os.path.join(os.getcwd(), "database_files")
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    db_path = os.path.join(db_dir, "local_songs.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Create the table if it doesn’t exist.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS songs (
            id TEXT PRIMARY KEY,
            title TEXT,
            artist TEXT,
            album TEXT,
            path TEXT UNIQUE,
            thumbnail TEXT,
            duration INTEGER
        )
    ''')
    conn.commit()
    return conn

def get_new_local_id(cursor):
    """
    Look through the current song IDs (which are of the form "local-<number>")
    and generate a new one with an incremented number.
    """
    cursor.execute("SELECT id FROM songs")
    rows = cursor.fetchall()
    max_num = 0
    for (song_id,) in rows:
        if song_id.startswith("local-"):
            try:
                num = int(song_id.split("-")[1])
                if num > max_num:
                    max_num = num
            except ValueError:
                # In case the format is not as expected, ignore it.
                pass
    return "local-" + str(max_num + 1)


def load_local_songs():
    """Scan local directories for music files, update Redis,
    and load the current list of songs into local_songs."""
    if not LOCAL_SONGS_PATHS:
        return {}

    # Split the directories and log what we're scanning
    dirs = [os.path.abspath(d.strip()) for d in LOCAL_SONGS_PATHS.split(";") if d.strip()]
    logger.info(f"Loading local songs from: {dirs}")

    # Set of file extensions to consider
    file_exts = {".mp3", ".flac", ".m4a", ".wav", ".ogg", ".wma", ".aac", ".aiff", ".alac"}

    # Connect to Redis with string decoding for ease of use
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

    # Walk through each provided directory
    for d in dirs:
        if not os.path.isdir(d):
            logger.warning(f"Local path is not a directory: {d}")
            continue

        for root, _, files in os.walk(d):
            for fname in files:
                _, ext = os.path.splitext(fname)
                if ext.lower() not in file_exts:
                    continue

                # Use absolute path to ensure consistency
                full_path = os.path.abspath(os.path.join(root, fname))

                # Skip if the file doesn't exist
                if not os.path.exists(full_path):
                    continue

                # Check if this song file is already in Redis using the path
                song_id = redis_client.hget("path_to_id", full_path)
                if not song_id:
                    # Generate a new unique ID based on filename
                    base_name, _ = os.path.splitext(fname)
                    desired_id = base_name
                    suffix = 0
                    while redis_client.exists(desired_id):
                        suffix += 1
                        desired_id = f"{base_name}_{suffix}"
                    song_id = desired_id
                    # Map the path to the new song ID
                    redis_client.hset("path_to_id", full_path, song_id)
                    # Add to the set of song IDs
                    redis_client.sadd("local_songs_ids", song_id)

                # Default metadata values
                title = fname
                artist = "Unknown Artist"
                album = "Unknown Album"
                duration = 0
                thumbnail = ""

                # Extract metadata using Mutagen
                try:
                    audio = File(full_path, easy=True)
                    if audio and hasattr(audio, "info") and hasattr(audio.info, "length"):
                        duration = int(audio.info.length)
                    if audio:
                        title = audio.get("title", [fname])[0]
                        artist = audio.get("artist", ["Unknown Artist"])[0]
                        album = audio.get("album", ["Unknown Album"])[0]

                    # Extract thumbnail if available
                    if isinstance(audio, FLAC) or hasattr(audio, "pictures"):
                        pictures = getattr(audio, "pictures", [])
                        if pictures:
                            pic = pictures[0]
                            thumbnail = f"data:{pic.mime};base64,{base64.b64encode(pic.data).decode()}"
                    elif hasattr(audio, "tags"):
                        tags = audio.tags
                        if "APIC:" in tags:
                            pic = tags["APIC:"]
                            thumbnail = f"data:{pic.mime};base64,{base64.b64encode(pic.data).decode()}"
                except Exception as e:
                    logger.error(f"Error reading metadata from {fname}: {e}")

                # Store or update the song's metadata in Redis
                redis_client.hset(song_id, mapping={
                    "id": song_id,
                    "title": title,
                    "artist": artist,
                    "album": album,
                    "path": full_path,
                    "thumbnail": thumbnail,
                    "duration": str(duration)  # Redis stores values as strings
                })

    # Load all songs from Redis into local_songs, cleaning up invalid entries
    global local_songs
    local_songs = {}
    song_ids = redis_client.smembers("local_songs_ids") or set()
    for song_id in song_ids:
        song_data = redis_client.hgetall(song_id)
        if song_data and "path" in song_data and os.path.exists(song_data["path"]):
            local_songs[song_id] = {
                "id": song_data["id"],
                "title": song_data["title"],
                "artist": song_data["artist"],
                "album": song_data["album"],
                "path": song_data["path"],
                "thumbnail": song_data["thumbnail"],
                "duration": int(song_data["duration"])  # Convert back to int
            }
        else:
            # Remove song if file no longer exists
            redis_client.delete(song_id)
            redis_client.srem("local_songs_ids", song_id)
            if song_data and "path" in song_data:
                redis_client.hdel("path_to_id", song_data["path"])

    logger.info(f"Loaded {len(local_songs)} local songs with metadata.")
    return local_songs

def get_song_info(song_id):
    """Get song metadata with error handling."""
    try:
        info = ytmusic.get_song(song_id)
        if not info:
            return jsonify({"error": "Song not found"}), 404

        vd = info.get("videoDetails", {})
        return jsonify({
            "id": song_id,
            "title": vd.get("title", "Unknown"),
            "artist": vd.get("author", "Unknown Artist"),
            "thumbnail": f"https://i.ytimg.com/vi/{song_id}/hqdefault.jpg",
            "duration": int(vd.get("lengthSeconds", 0))
        })
    except Exception as e:
        logger.error(f"Song info error: {e}")
        return jsonify({"error": str(e)}), 500

def get_fallback_recommendations():
    """Get fallback recommendations when normal methods fail."""
    try:
        # Try popular music first
        results = ytmusic.search("popular music", filter="songs", limit=5)
        if not results:
            # Fallback to trending
            results = ytmusic.search("trending songs", filter="songs", limit=5)

        if not results:
            return jsonify([]), 404

        recs = []
        for track in results:
            if not track.get("videoId"):
                continue

            recs.append({
                "id": track["videoId"],
                "title": track["title"],
                "artist": track["artists"][0]["name"] if track.get("artists") else "Unknown",
                "thumbnail": f"https://i.ytimg.com/vi/{track['videoId']}/hqdefault.jpg"
            })

            if len(recs) >= 5:
                break

        return jsonify(recs)

    except Exception as e:
        logger.error(f"Fallback recommendations error: {e}")
        return jsonify([])

def is_potential_video_id(filename: str) -> bool:
    """Check if a filename might be a YouTube video ID."""
    # Remove local- prefix if present
    if filename.startswith("local-"):
        filename = filename[6:]

    # YouTube video IDs are typically 11 characters
    # and contain alphanumeric chars plus - and _
    return bool(re.match(r'^[A-Za-z0-9_-]{11}$', filename))


def sanitize_filename(s: str) -> str:
    """Remove invalid characters from filename."""
    # Remove invalid chars
    s = re.sub(r'[<>:"/\\|?*]', '', s)
    # Limit length
    s = s[:200]
    # Remove leading/trailing spaces and dots
    s = s.strip('. ')
    # Replace remaining dots except last extension
    s = s.replace('.', '_')
    return s if s else 'unknown'

def process_description(description):
    """Process and clean artist description"""
    if isinstance(description, list):
        description = ' '.join(description)
    return description.strip() or 'No description available'

def get_best_thumbnail(thumbnails):
    """Get highest quality thumbnail URL"""
    try:
        if not thumbnails or not isinstance(thumbnails, list):
            return ''
        thumb = thumbnails[-1].get('url', '')
        if thumb.startswith('//'):
            return f"https:{thumb}"
        return thumb
    except:
        return ''

def process_genres(artist_data):
    """Extract and process genres"""
    try:
        if 'genres' not in artist_data:
            return []

        if isinstance(artist_data['genres'], list):
            return artist_data['genres']
        elif artist_data['genres']:
            return [artist_data['genres']]
        return []
    except:
        return []

@login_required
def get_artist_stats(artist_data):
    """Extract all artist statistics"""
    try:
        # Get monthly listeners with fallbacks
        monthly_listeners = get_monthly_listeners(artist_data)

        # Extract and format stats
        stats = {
            'subscribers': safe_format_count(artist_data.get('subscribers', '0')),
            'views': safe_format_count(artist_data.get('views', '0')),
            'monthlyListeners': safe_format_count(monthly_listeners)
        }

        # Try to get additional stats if available
        if 'stats' in artist_data and isinstance(artist_data['stats'], dict):
            extra_stats = artist_data['stats']
            if 'totalPlays' in extra_stats:
                stats['totalPlays'] = safe_format_count(extra_stats['totalPlays'])
            if 'avgDailyPlays' in extra_stats:
                stats['avgDailyPlays'] = safe_format_count(extra_stats['avgDailyPlays'])

        return stats
    except Exception as e:
        logger.warning(f"Error processing stats: {e}")
        return {
            'subscribers': '0',
            'views': '0',
            'monthlyListeners': '0'
        }

def get_monthly_listeners(artist_data):
    """Extract monthly listeners with multiple fallback methods"""
    try:
        # Try direct stats object first
        if 'stats' in artist_data:
            stats = artist_data['stats']
            if isinstance(stats, dict):
                for key in ['monthlyListeners', 'monthly_listeners', 'listeners']:
                    if key in stats and stats[key]:
                        return stats[key]

        # Try top level fields
        for key in ['monthlyListeners', 'monthly_listeners', 'listeners']:
            if key in artist_data and artist_data[key]:
                return artist_data[key]

        # Get from subscriptionButton if available
        if 'subscriptionButton' in artist_data:
            sub_text = artist_data['subscriptionButton'].get('text', '')
            if isinstance(sub_text, str):
                match = re.search(r'(\d[\d,.]*[KMB]?)\s*(?:monthly listeners|listeners)',
                                sub_text, re.IGNORECASE)
                if match:
                    return match.group(1)

        # Try to extract from header or subscription count
        return (artist_data.get('header', {}).get('subscriberCount') or
                artist_data.get('subscribers', '0'))

    except Exception as e:
        logger.warning(f"Error extracting monthly listeners: {e}")
        return '0'

def process_top_songs(artist_data):
    """Process and extract top songs information"""
    top_songs = []
    try:
        songs_data = artist_data.get('songs', [])
        if not isinstance(songs_data, list):
            return []

        for song in songs_data[:10]:  # Limit to top 10
            if not isinstance(song, dict):
                continue

            try:
                song_info = {
                    'title': song.get('title', 'Unknown'),
                    'videoId': song.get('videoId', ''),
                    'plays': safe_format_count(song.get('plays', '0')),
                    'duration': song.get('duration', ''),
                    'thumbnail': get_best_thumbnail(song.get('thumbnails', [])),
                    'album': (song.get('album', {}) or {}).get('name', ''),
                    'year': song.get('year', '')
                }

                if song_info['videoId']:  # Only add if we have a valid video ID
                    top_songs.append(song_info)
            except Exception as e:
                logger.warning(f"Error processing song: {e}")
                continue

    except Exception as e:
        logger.warning(f"Error processing top songs: {e}")

    return top_songs

def process_artist_links(artist_data, artist_id):
    """Process and extract all artist links"""
    links = {
        'youtube': f"https://music.youtube.com/channel/{artist_id}" if artist_id else None,
        'official': artist_data.get('officialWebsite')
    }

    # Add social media links if available
    try:
        if 'links' in artist_data:
            for link in artist_data['links']:
                if isinstance(link, dict):
                    link_type = link.get('type', '').lower()
                    if link_type in ['instagram', 'twitter', 'facebook']:
                        links[link_type] = link.get('url', '')
    except Exception as e:
        logger.warning(f"Error processing social links: {e}")

    return links

def extract_video_id(url):
    """Extract video ID from various YouTube/YouTube Music URL formats"""
    try:
        # Common YouTube URL patterns
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|music\.youtube\.com/watch\?v=|music\.youtube\.com/playlist\?list=)([a-zA-Z0-9_-]{11})',
            r'(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})',
            r'(?:youtube\.com/playlist\?list=)([a-zA-Z0-9_-]{34})',
            r'(?:music\.youtube\.com/browse/)([a-zA-Z0-9_-]{11})'
        ]

        # Try each pattern
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None
    except:
        return None

def extract_year(artist_data):
    """Extract artist's formation/debut year"""
    try:
        # Try direct year field
        if 'yearFormed' in artist_data:
            return str(artist_data['yearFormed'])

        # Try years active
        if 'years_active' in artist_data:
            years = artist_data['years_active']
            if isinstance(years, list) and years:
                return str(years[0])

        # Try to find year in description
        if 'description' in artist_data:
            desc = str(artist_data['description'])
            match = re.search(r'\b(19|20)\d{2}\b', desc)
            if match:
                return match.group(0)

        return None
    except Exception as e:
        logger.warning(f"Error extracting year: {e}")
        return None

def safe_format_count(count):
    """Safely format numerical counts with K/M/B suffixes"""
    try:
        if not count or str(count).strip() in ['0', '', 'None', 'null']:
            return '0'

        # Remove any commas and spaces
        count_str = str(count).replace(',', '').replace(' ', '')

        # Handle if count is already formatted
        if any(suffix in count_str.upper() for suffix in ['K', 'M', 'B']):
            return count_str.upper()  # Normalize suffix to uppercase

        num = float(count_str)
        if num >= 1_000_000_000:
            return f"{num/1_000_000_000:.1f}B"
        if num >= 1_000_000:
            return f"{num/1_000_000:.1f}M"
        if num >= 1_000:
            return f"{num/1_000:.1f}K"
        return str(int(num))
    except Exception as e:
        logger.warning(f"Error formatting count {count}: {e}")
        return str(count)

def get_download_info(video_id):
    """Return file path if already downloaded, else None."""
    # Assuming the user_id is available in the context or session
    # If not, you might need to modify this to include user_id as parameter
    try:
        # Get user_id from Flask session if available
        user_id = session.get('user_id') if 'user_id' in session else 'default_user'

        # Create the Redis key for this video download
        download_key = f"download:{user_id}:{video_id}"

        # Get the path from Redis hash
        path = redis_client.hget(download_key, "path")

        # Verify the file still exists
        if path and os.path.exists(path):
            return path
        elif path:
            # Clean up Redis if file doesn't exist
            redis_client.delete(download_key)
            redis_client.srem(f"user_downloads:{user_id}", video_id)
            return None
        return None

    except Exception as e:
        logger.error(f"Error checking download info for {video_id}: {e}")
        return None

def get_recent_plays(limit=10):
    """Get user's recent play history for context."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT DISTINCT song_id
        FROM history
        ORDER BY played_at DESC
        LIMIT ?
    """, (limit,))
    recent = [row[0] for row in c.fetchall()]
    conn.close()
    return recent


def filter_local_songs(query: str):
    """Return deduplicated local songs with title/artist matching the query."""
    qlow = query.lower()
    seen_titles = set()  # Track seen title+artist combinations
    out = []

    for sid, meta in local_songs.items():
        title_artist = (meta["title"].lower(), meta["artist"].lower())

        if (qlow in meta["title"].lower() or qlow in meta["artist"].lower()) and \
           title_artist not in seen_titles:
            out.append(meta)
            seen_titles.add(title_artist)

    return out

@lru_cache(maxsize=100)
def search_songs(query: str):
    """YTMusic search (songs) with deduplication, cached for 1 hour."""
    now_ts = datetime.now().timestamp()
    if query in search_cache:
        old_data, old_ts = search_cache[query]
        if now_ts - old_ts < CACHE_DURATION:
            return old_data

    try:
        raw = ytmusic.search(query, filter="songs")
        seen_titles = set()  # Track seen title+artist combinations
        results = []

        for item in raw:
            vid = item.get("videoId")
            if not vid:
                continue

            artist = "Unknown Artist"
            if item.get("artists"):
                artist = item["artists"][0].get("name", "Unknown Artist")

            title = item.get("title", "Unknown")
            title_artist = (title.lower(), artist.lower())

            # Skip if we've seen this title+artist combination
            if title_artist in seen_titles:
                continue

            dur = item.get("duration_seconds", 0)
            thumb = f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"

            results.append({
                "id": vid,
                "title": title,
                "artist": artist,
                "album": "",
                "duration": dur,
                "thumbnail": thumb
            })

            seen_titles.add(title_artist)

        search_cache[query] = (results, now_ts)
        return results

    except Exception as e:
        logger.error(f"search_songs error: {e}")
        return []

def fallback_recommendations():
    """Simplified fallback using search instead of unavailable methods."""
    try:
        categories = [
            "top hits",
            "popular music",
            "trending songs",
            "new releases",
            "viral hits"
        ]

        recommendations = []
        seen_songs = set()

        # Try 2 random categories
        selected_cats = random.sample(categories, 2)

        for query in selected_cats:
            results = ytmusic.search(query, filter="songs", limit=10)
            if results:
                selected = random.sample(results, min(3, len(results)))
                for track in selected:
                    add_recommendation(track, recommendations, seen_songs)
                    if len(recommendations) >= 5:
                        break

        random.shuffle(recommendations)
        return jsonify(recommendations[:5])

    except Exception as e:
        logger.error(f"Fallback recommendations error: {e}")
        return jsonify([])


def get_local_song_recommendations(local_song_id):
    """Get recommendations for local songs using title/artist search."""
    try:
        local_meta = local_songs.get(local_song_id)
        if not local_meta:
            return fallback_recommendations()

        recommendations = []
        seen_songs = set()

        # Search using song title and artist
        query = f"{local_meta['title']} {local_meta['artist']}"
        search_results = ytmusic.search(query, filter="songs", limit=15)

        for track in search_results:
            if add_recommendation(track, recommendations, seen_songs):
                if len(recommendations) >= 5:
                    break

        # If we need more, add some popular songs
        if len(recommendations) < 5:
            popular = ytmusic.search("popular music", filter="songs", limit=10)
            for track in popular:
                if add_recommendation(track, recommendations, seen_songs):
                    if len(recommendations) >= 5:
                        break

        random.shuffle(recommendations)
        return jsonify(recommendations[:5])

    except Exception as e:
        logger.error(f"Local recommendations error: {e}")
        return fallback_recommendations()

def add_recommendation(track, recommendations, seen_songs, current_song_id=None):
    """Enhanced track processing with better validation."""
    try:
        video_id = track.get("videoId")
        if not video_id:
            return False

        # Skip if song is current or recently played
        if video_id in seen_songs or video_id == current_song_id:
            return False

        # Skip unavailable or private videos
        if track.get("isAvailable") == False or track.get("isPrivate") == True:
            return False

        # Get track details with validation
        title = track.get("title", "").strip()
        if not title or title == "Unknown":
            return False

        artist = "Unknown Artist"
        if "artists" in track and track["artists"]:
            artist = track["artists"][0].get("name", "").strip()
        elif track.get("artist"):
            artist = track["artist"].strip()

        # Skip if we couldn't get valid title/artist
        if not title or not artist or artist == "Unknown Artist":
            return False

        album = ""
        if "album" in track and isinstance(track["album"], dict):
            album = track["album"].get("name", "").strip()

        thumb = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
        duration = int(track.get("duration_seconds", 0))

        # Skip very short or very long tracks
        if duration < 30 or duration > 1800:  # Between 30s and 30m
            return False

        # Add to recommendations and mark as seen
        recommendations.append({
            "id": video_id,
            "title": title,
            "artist": artist,
            "album": album,
            "thumbnail": thumb,
            "duration": duration
        })
        seen_songs.add(video_id)
        return True

    except Exception as e:
        logger.warning(f"Error processing track: {e}")
        return False

def cleanup_expired_sessions():
    """Remove expired sessions from database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM active_sessions WHERE expires_at <= CURRENT_TIMESTAMP")
    conn.commit()
    conn.close()

def get_fallback_tracks(seen_songs):
    """Get fallback tracks from trending/popular songs or charts."""
    try:
        fallback_tracks = []

        # Try trending
        trending = ytmusic.get_trending_music()
        if trending:
            for track in trending:
                add_recommendation(track, fallback_tracks, seen_songs)

        # If still need more, use charts
        if len(fallback_tracks) < 5:
            charts = ytmusic.get_charts()
            if charts and "items" in charts:
                for track in charts["items"]:
                    add_recommendation(track, fallback_tracks, seen_songs)
        return fallback_tracks
    except Exception as e:
        logger.warning(f"Fallback tracks error: {e}")
        return []

@lru_cache(maxsize=1000)
@login_required
def fetch_image(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.content, response.headers.get('Content-Type', 'image/jpeg')
    except Exception as e:
        logger.error(f"Error fetching image {url}: {e}")
    return None, None


def safe_int(value, default=0):
    """Safely convert value to int."""
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def get_overview_stats(c, user_id: int) -> Dict[str, Any]:
    """Get overall listening statistics for a specific user."""
    # Total listening time
    c.execute("SELECT SUM(listened_duration) FROM listening_history WHERE user_id = ?", (user_id,))
    total_time = c.fetchone()[0] or 0

    # Total songs played
    c.execute("SELECT COUNT(*) FROM listening_history WHERE user_id = ?", (user_id,))
    total_songs = c.fetchone()[0] or 0

    # Unique artists
    c.execute("SELECT COUNT(DISTINCT artist) FROM listening_history WHERE user_id = ?", (user_id,))
    unique_artists = c.fetchone()[0] or 0

    return {
        "total_time": total_time,
        "total_songs": total_songs,
        "unique_artists": unique_artists,
        "average_daily": total_songs / max(1, (datetime.now() - datetime.strptime(get_first_listen_date(c, user_id), '%Y-%m-%d')).days)
    }

def get_first_listen_date(c, user_id: int) -> str:
    """Get the date of first listen for a specific user."""
    c.execute("SELECT MIN(started_at) FROM listening_history WHERE user_id = ?", (user_id,))
    first_listen = c.fetchone()[0]
    if first_listen:
        return datetime.strptime(first_listen, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
    return datetime.now().strftime('%Y-%m-%d')

def get_recent_activity(c, user_id: int) -> List[Dict[str, Any]]:
    """Get recent activity with proper IST timestamps for a specific user."""
    c.execute("""
        SELECT
            title, artist, started_at, listened_duration, completion_rate
        FROM listening_history
        WHERE user_id = ?
        ORDER BY started_at DESC
        LIMIT 50
    """, (user_id,))

    activities = []
    for row in c.fetchall():
        try:
            started_at = time_sync.parse_datetime(row[2])
            activities.append({
                "title": row[0],
                "artist": row[1],
                "started_at": time_sync.format_time(started_at),
                "started_at_relative": time_sync.format_time(started_at, relative=True),
                "duration": row[3],
                "completion": row[4]
            })
        except Exception as e:
            logger.error(f"Error processing activity: {e}")
            continue
    return activities

def get_top_artists(c, user_id: int) -> List[Dict[str, Any]]:
    """Get top artists by listening time for a specific user."""
    c.execute("""
        SELECT
            artist,
            COUNT(*) as plays,
            SUM(listened_duration) as total_time
        FROM listening_history
        WHERE user_id = ?
        GROUP BY artist
        ORDER BY total_time DESC
        LIMIT 10
    """, (user_id,))

    return [{
        "name": row[0],
        "plays": row[1],
        "time": row[2]
    } for row in c.fetchall()]

def get_listening_patterns(c, user_id: int) -> Dict[str, Any]:
    """Analyze listening patterns with proper data validation for a specific user."""
    try:
        hourly_pattern = {str(h).zfill(2): 0 for h in range(24)}
        daily_pattern = {str(d): 0 for d in range(7)}

        c.execute("""
            SELECT
                strftime('%H', started_at) as hour,
                COUNT(*) as plays
            FROM listening_history
            WHERE user_id = ? AND started_at IS NOT NULL
            GROUP BY hour
            ORDER BY hour
        """, (user_id,))

        for hour, plays in c.fetchall():
            if hour in hourly_pattern:
                hourly_pattern[hour] = plays

        c.execute("""
            SELECT
                strftime('%w', started_at) as day,
                COUNT(*) as plays
            FROM listening_history
            WHERE user_id = ? AND started_at IS NOT NULL
            GROUP BY day
            ORDER BY day
        """, (user_id,))

        for day, plays in c.fetchall():
            if day in daily_pattern:
                daily_pattern[day] = plays

        return {
            "hourly": hourly_pattern,
            "daily": daily_pattern
        }
    except Exception as e:
        logger.error(f"Error getting listening patterns: {e}")
        return {
            "hourly": {str(h).zfill(2): 0 for h in range(24)},
            "daily": {str(d): 0 for d in range(7)}
        }

def get_completion_rates(c, user_id: int) -> Dict[str, Any]:
    """Analyze song completion rates with better error handling for a specific user."""
    try:
        c.execute("""
            SELECT
                COALESCE(listen_type, 'partial') as type,
                COUNT(*) as count
            FROM listening_history
            WHERE user_id = ? AND listen_type IS NOT NULL
            GROUP BY type
            ORDER BY count DESC
        """, (user_id,))

        completion_stats = {'full': 0, 'partial': 0, 'skip': 0}
        for listen_type, count in c.fetchall():
            if listen_type in completion_stats:
                completion_stats[listen_type] = count

        c.execute("""
            SELECT AVG(completion_rate)
            FROM listening_history
            WHERE user_id = ? AND completion_rate IS NOT NULL AND completion_rate BETWEEN 0 AND 100
        """, (user_id,))

        avg_completion = round(float(c.fetchone()[0] or 0), 2)

        return {
            "completion_distribution": completion_stats,
            "average_completion": avg_completion
        }
    except Exception as e:
        logger.error(f"Error getting completion rates: {e}")
        return {
            "completion_distribution": {'full': 0, 'partial': 0, 'skip': 0},
            "average_completion": 0
        }

def get_average_completion(c) -> float:
    """Calculate average completion rate."""
    c.execute("SELECT AVG(completion_rate) FROM listening_history")
    return round(c.fetchone()[0] or 0, 2)

def record_listen_start(user_id: int, song_id: str, title: str, artist: str, session_id: str) -> int:
    """Record listen start with accurate IST timestamp for a specific user."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        current_time = time_sync.get_current_time()
        song_id = str(song_id).strip()
        title = str(title).strip()
        artist = str(artist).strip()
        session_id = str(session_id).strip()

        c.execute("""
            INSERT INTO listening_history
            (user_id, song_id, title, artist, session_id, started_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, song_id, title, artist, session_id, current_time.strftime('%Y-%m-%d %H:%M:%S')))

        listen_id = c.lastrowid
        conn.commit()
        return listen_id
    finally:
        conn.close()

def record_listen_end(listen_id: int, duration: int, listened_duration: int):
    """Record the end of a song listen with analytics."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        # Handle null/invalid duration
        try:
            duration = int(float(duration or 0))
            listened_duration = int(float(listened_duration or 0))
        except (TypeError, ValueError):
            duration = 0
            listened_duration = 0

        # Ensure we don't divide by zero
        completion_rate = (listened_duration / duration) * 100 if duration > 0 else 0
        listen_type = 'full' if completion_rate >= 85 else 'skip' if completion_rate <= 20 else 'partial'

        c.execute("""
            UPDATE listening_history
            SET ended_at = CURRENT_TIMESTAMP,
                duration = ?,
                listened_duration = ?,
                completion_rate = ?,
                listen_type = ?
            WHERE id = ?
        """, (duration, listened_duration, completion_rate, listen_type, listen_id))

        conn.commit()
    finally:
        conn.close()

@login_required
def update_artist_stats(c, artist: str, duration: int, listened_duration: int):
    """Update artist listening statistics."""
    c.execute("""
        INSERT INTO artist_stats (
            artist, total_plays, total_time, first_played, last_played
        )
        VALUES (?, 1, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT(artist) DO UPDATE SET
            total_plays = total_plays + 1,
            total_time = total_time + ?,
            last_played = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
    """, (artist, listened_duration, listened_duration))

@login_required
def update_daily_stats(c, duration: int, listened_duration: int):
    """Update aggregated daily statistics."""
    today = datetime.now().strftime('%Y-%m-%d')

    c.execute("""
        INSERT INTO daily_stats (date, total_songs, total_time)
        VALUES (?, 1, ?)
        ON CONFLICT(date) DO UPDATE SET
            total_songs = total_songs + 1,
            total_time = total_time + ?,
            updated_at = CURRENT_TIMESTAMP
    """, (today, listened_duration, listened_duration))


def generate_otp():
    """Generate a 6-digit OTP"""
    return ''.join([str(secrets.randbelow(10)) for _ in range(6)])

def store_otp(email, otp, purpose):
    """Store OTP in database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    expires_at = datetime.now() + timedelta(minutes=10)

    # Clear any existing OTPs for this email/purpose
    c.execute("""
        DELETE FROM pending_otps
        WHERE email = ? AND purpose = ?
    """, (email, purpose))

    c.execute("""
        INSERT INTO pending_otps (email, otp, purpose, expires_at)
        VALUES (?, ?, ?, ?)
    """, (email, otp, purpose, expires_at))

    conn.commit()
    conn.close()

def verify_otp(email, otp, purpose):
    """Verify OTP from database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        SELECT otp FROM pending_otps
        WHERE email = ? AND purpose = ?
        AND expires_at > CURRENT_TIMESTAMP
        ORDER BY created_at DESC LIMIT 1
    """, (email, purpose))

    row = c.fetchone()
    valid = row and row[0] == otp

    if valid:
        c.execute("""
            DELETE FROM pending_otps
            WHERE email = ? AND purpose = ?
        """, (email, purpose))

    conn.commit()
    conn.close()
    return valid

def download_default_songs():
    default_songs = ["dQw4w9WgXcQ"]
    for song_id in default_songs:
        flac_path = os.path.join(MUSIC_DIR, f"{song_id}.flac")
        if not os.path.exists(flac_path):
            download_flac_init(song_id)


def send_email(to_email, subject, body):
    """Send HTML email using SMTP"""
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = SMTP_USER
        msg['To'] = to_email
        msg['Subject'] = subject
        html_part = MIMEText(body, 'html')
        msg.attach(html_part)

        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False


def download_flac(video_id: str, user_id: int) -> str:
    """
    Download song with metadata and thumbnail using yt-dlp.exe with fallback to yt-dlp module.

    Args:
        video_id: str - The YouTube video ID
        user_id: int - The ID of the user downloading the song

    Returns:
        str: Path to the FLAC file if successful or existing, None if failed
    """
    try:
        flac_path = os.path.join(MUSIC_DIR, f"{video_id}.flac")

        if os.path.exists(flac_path):
            return flac_path

        logger.info(f"Downloading new song: {video_id}")
        yt_music_url = f"https://music.youtube.com/watch?v={video_id}"

        # Try executable first
        try:
            return download_with_executable(video_id, user_id, yt_music_url, flac_path, is_init=False)
        except Exception as exe_error:
            logger.warning(f"Executable download failed, falling back to module: {str(exe_error)}")
            return download_with_module(video_id, user_id, yt_music_url, flac_path, is_init=False)

    except Exception as e:
        logger.error(f"Error downloading song {video_id}: {str(e)}")
        return None

def download_flac_init(video_id: str) -> str:
    """
    Version of download_flac that works during initialization with fallback to yt-dlp module.

    Args:
        video_id: str - The YouTube video ID

    Returns:
        str: Path to the FLAC file if successful or existing, None if failed
    """
    try:
        flac_path = os.path.join(MUSIC_DIR, f"{video_id}.flac")

        if os.path.exists(flac_path):
            return flac_path

        logger.info(f"Downloading new song during initialization: {video_id}")
        yt_music_url = f"https://music.youtube.com/watch?v={video_id}"

        # Try executable first
        try:
            return download_with_executable(video_id, None, yt_music_url, flac_path, is_init=True)
        except Exception as exe_error:
            logger.warning(f"Executable download failed during init, falling back to module: {str(exe_error)}")
            return download_with_module(video_id, None, yt_music_url, flac_path, is_init=True)

    except Exception as e:
        logger.error(f"Error downloading song {video_id} during initialization: {str(e)}")
        return None

def download_with_executable(video_id: str, user_id: int | None, url: str, flac_path: str, is_init: bool) -> str:
    """Helper function to download using yt-dlp executable"""
    # Get metadata first
    result = subprocess.run([
        YTDLP_PATH,
        "--quiet",
        "--print", "%(title)s",
        "--print", "%(artist)s",
        "--print", "%(album)s",
        url
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception(f"Metadata extraction failed: {result.stderr}")

    title, artist, album = result.stdout.strip().split("\n")

    command = [
        YTDLP_PATH,
        "--quiet",
        "--no-warnings",
        "--extract-audio",
        "--audio-format", "flac",
        "--audio-quality", "0",
        "--embed-metadata",
        "--embed-thumbnail",
        "-o", os.path.join(MUSIC_DIR, "%(id)s.%(ext)s"),
    ]

    # Only add ffmpeg path on Windows
    if platform.system().lower() == "windows":
        command.extend(["--ffmpeg-location", FFMPEG_BIN_DIR])

    command.append(url)

    download_result = subprocess.run(command)

    if download_result.returncode == 0 and os.path.exists(flac_path):
        if not is_init:
            record_download(video_id, title, artist, album, flac_path, user_id)
            load_local_songs()
        logger.info(f"Successfully downloaded {video_id} using executable")
        return flac_path

    raise Exception(f"Executable download failed with code {download_result.returncode}")

def download_with_module(video_id: str, user_id: int | None, url: str, flac_path: str, is_init: bool) -> str:
    """Helper function to download using yt-dlp Python module with optimized settings"""
    ydl_opts = {
        # Basic options
        'format': 'bestaudio/best',  # Get best quality audio
        'outtmpl': os.path.join(MUSIC_DIR, '%(id)s.%(ext)s'),

        # Audio processing
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'flac',
            'preferredquality': '0',  # Best quality
            'nopostoverwrites': False,  # Allow overwriting postprocessed files
        }, {
            # Add metadata
            'key': 'FFmpegMetadata',
            'add_metadata': True,
            'add_chapters': True,
        }, {
            # Add thumbnail
            'key': 'EmbedThumbnail',
            'already_have_thumbnail': False,
        }],

        # Additional options
        'writethumbnail': True,  # Write thumbnail to disk before embedding
        'embedthumbnail': True,  # Embed thumbnail in audio file
        'addmetadata': True,     # Write metadata to file
        'prefer_ffmpeg': True,   # Prefer ffmpeg for post-processing

        # Quality settings
        'audioformat': 'flac',   # Force FLAC format
        'audioquality': '0',     # Best quality

        # Optimization settings
        'concurrent_fragment_downloads': 1,  # Download fragments concurrently
        'retries': 10,           # Retry on download errors
        'fragment_retries': 10,  # Retry on fragment download errors

        # Output settings
        'extract_flat': False,   # Extract audio
        'keepvideo': False,      # Don't keep video file after extraction
        'clean_infojson': True,  # Remove info json after download

        # Progress settings
        'progress_hooks': [],    # Can add progress tracking if needed
        'postprocessor_hooks': [], # Can add post-processing tracking
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            # Extract metadata first
            info = ydl.extract_info(url, download=False)
            title = info.get('title', '')
            artist = info.get('artist', '')
            album = info.get('album', '')

            # Get additional metadata if available
            track = info.get('track', '')
            release_year = info.get('release_year', '')
            release_date = info.get('release_date', '')
            genre = info.get('genre', '')

            # Download and process the file
            ydl.download([url])

            if os.path.exists(flac_path):
                if not is_init:
                    # Record download with extended metadata
                    record_download(
                        video_id=video_id,
                        title=title or track or "Unknown Title",
                        artist=artist or "Unknown Artist",
                        album=album or "Unknown Album",
                        path=flac_path,
                        user_id=user_id
                    )
                    load_local_songs()
                logger.info(f"Successfully downloaded {video_id} using module")
                return flac_path

            raise Exception("Module download failed - file not found")

    except Exception as e:
        logger.error(f"Download error for {video_id}: {str(e)}")
        raise Exception(f"Download failed: {str(e)}")

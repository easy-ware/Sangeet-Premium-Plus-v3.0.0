# Flask Imports
from flask import (
    Blueprint, session, jsonify, send_file, url_for, render_template,
    request, redirect, render_template_string, make_response, current_app
)

# External Libraries
import bcrypt
import concurrent.futures
import json
from pytz import timezone as pytz_timezone, UnknownTimeZoneError # Import specific items needed
import logging
import random
import re
import redis
import requests
import sqlite3
import time
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from functools import wraps, partial, lru_cache
from threading import Thread
from urllib.parse import urlparse, parse_qs
import yt_dlp
from ytmusicapi import YTMusic

# Internal Imports
import pytz
from ..utils import util
from sangeet_premium import var_templates

# Standard Library
import os
import secrets
from concurrent.futures import ThreadPoolExecutor

# Global variables and constants
local_songs = {}
search_cache = {}
song_cache = {}
lyrics_cache = {}
CACHE_DURATION = 3600
DB_PATH = os.path.join(os.getcwd(), "database_files", "sangeet_database_main.db")
DB_PATH_MAIN = os.path.join(os.getcwd(), "database_files", "sangeet_database_main.db")
DB_PATH_ISSUES = os.path.join(os.getcwd(), "database_files", "issues.db")
PLAYLIST_DB_PATH = os.path.join(os.getcwd(), "database_files", "playlists.db")
LOCAL_JSON_PATH = os.path.join(os.getcwd(), "locals", "local.json")
SERVER_DOMAIN = os.getenv('SANGEET_BACKEND', f'http://127.0.0.1:{os.getenv("PORT")}')
ADMIN_PASSWORD = "Sk124" # Keep this secure in a real application (e.g., environment variable)


# Blueprint and external service initialization
bp = Blueprint('playback', __name__)
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
ytmusic = YTMusic()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load local JSON data at startup
try:
    with open(LOCAL_JSON_PATH, 'r', encoding='utf-8') as f:
        local_data = json.load(f)
except Exception as e:
    print(f"Error reading the JSON file: {e}")
    local_data = {}

# Decorators and utility functions
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
            session.clear()
            return redirect("/login")
            
        return f(*args, **kwargs)
    return decorated_function

def load_local_songs_from_file():
    """Load songs from Redis into the local_songs dictionary, ensuring all required keys are present."""
    global local_songs
    local_songs = {}
    song_ids = redis_client.smembers("local_songs_ids") or set()
    required_keys = ["id", "title", "artist", "album", "path", "thumbnail", "duration"]

    for song_id in song_ids:
        song_data = redis_client.hgetall(song_id)
        if not all(key in song_data for key in required_keys):
            missing_keys = set(required_keys) - set(song_data.keys())
            logger.warning(f"Song {song_id} is missing keys: {missing_keys}. Skipping or removing.")
            redis_client.delete(song_id)
            redis_client.srem("local_songs_ids", song_id)
            if "path" in song_data:
                redis_client.hdel("path_to_id", song_data["path"])
            continue

        if "path" in song_data and os.path.exists(song_data["path"]):
            try:
                local_songs[song_id] = {
                    "id": song_data["id"],
                    "title": song_data["title"],
                    "artist": song_data["artist"],
                    "album": song_data["album"],
                    "path": song_data["path"],
                    "thumbnail": song_data["thumbnail"],
                    "duration": int(song_data["duration"])
                }
            except (ValueError, TypeError) as e:
                logger.error(f"Error processing song {song_id}: {e}. Skipping.")
                continue
        else:
            redis_client.delete(song_id)
            redis_client.srem("local_songs_ids", song_id)
            if "path" in song_data:
                redis_client.hdel("path_to_id", song_data["path"])

    logger.info(f"Loaded {len(local_songs)} songs from Redis.")
    return local_songs

@lru_cache(maxsize=1)
def get_default_songs():
    """Generate a cached list of default songs for empty queries."""
    local_songs = load_local_songs_from_file()
    combined = []
    seen_ids = set()

    for song in local_songs.values():
        if song["id"] not in seen_ids:
            combined.append(song)
            seen_ids.add(song["id"])

    return combined


# First add this helper function to extract playlist info
def extract_playlist_info(url, max_workers=4):
    """Extract playlist information using yt-dlp"""
    try:
        ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'force_generic_extractor': False
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info and 'entries' in info:
                # Process entries in parallel
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Create a partial function with ydl options
                    extract_func = partial(extract_video_info, ydl_opts=ydl_opts)
                    # Map the extraction function over all video URLs
                    video_urls = [entry['url'] if 'url' in entry else f"https://youtube.com/watch?v={entry['id']}" 
                                for entry in info['entries'] if entry]
                    results = list(executor.map(extract_func, video_urls))
                    return [r for r in results if r]  # Filter out None results
            return []
    except Exception as e:
        logger.error(f"Error extracting playlist info: {e}")
        return []

def extract_video_info(url, ydl_opts):
    """Extract single video information"""
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info:
                return {
                    "id": info.get('id', ''),
                    "title": info.get('title', 'Unknown'),
                    "artist": info.get('artist', info.get('uploader', 'Unknown Artist')),
                    "album": info.get('album', ''),
                    "duration": int(info.get('duration', 0)),
                    "thumbnail": get_best_thumbnail(info.get('thumbnails', [])),
                }
    except Exception as e:
        logger.error(f"Error extracting video info: {e}")
    return None

def get_best_thumbnail(thumbnails):
    """Get the best quality thumbnail URL"""
    if not thumbnails:
        return ""
    # Sort by resolution if available
    sorted_thumbs = sorted(thumbnails, 
                         key=lambda x: x.get('height', 0) * x.get('width', 0),
                         reverse=True)
    return sorted_thumbs[0].get('url', '')


# Function to get lyrics from cache
def get_cached_lyrics(song_id):
    conn = sqlite3.connect(os.path.join(os.getcwd() , "database_files" , "lyrics_cache.db"))
    cursor = conn.cursor()
    cursor.execute('SELECT lyrics FROM lyrics_cache WHERE song_id = ?', (song_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0].split('\n')
    return None

# Function to cache lyrics
def cache_lyrics(song_id, lyrics_lines):
    lyrics_text = '\n'.join(lyrics_lines)
    conn = sqlite3.connect(os.path.join(os.getcwd() , "database_files" , "lyrics_cache.db"))
    cursor = conn.cursor()
    cursor.execute('''
    INSERT OR REPLACE INTO lyrics_cache (song_id, lyrics, timestamp) 
    VALUES (?, ?, CURRENT_TIMESTAMP)
    ''', (song_id, lyrics_text))
    conn.commit()
    conn.close()




def get_video_info(video_id):
    try:
        # Retrieve song details via ytmusicapi
        song_info = ytmusic.get_song(video_id)
        details = song_info.get("videoDetails", {})
        title = details.get("title", "Unknown Title")
        artist = details.get("author", "Unknown Artist")
        thumbnails = details.get("thumbnail", {}).get("thumbnails", [])
        thumbnail = (max(thumbnails, key=lambda x: x.get("width", 0))["url"]
                     if thumbnails else f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg")
        return {"title": title, "artist": artist, "thumbnail": thumbnail, "video_id": video_id}
    except Exception:
        # Fallback: minimal extraction from YouTube page metadata
        url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            og_title = soup.find("meta", property="og:title")
            title = og_title["content"] if og_title else "Unknown Title"
            artist = "Unknown Artist"
            thumbnail = f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg"
            return {"title": title, "artist": artist, "thumbnail": thumbnail, "video_id": video_id}
    return {"title": "Unknown Title", "artist": "Unknown Artist",
            "thumbnail": f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg", "video_id": video_id}

def get_media_info(media_id):
    """
    Returns media details based on the given media_id.
    If media_id starts with 'local-', it fetches details from the local JSON file.
    Otherwise, it retrieves details using the YouTube Music API.
    """
    local_data = load_local_songs_from_file()
    if media_id.startswith("local-"):
        details = local_data.get(media_id)
        if details:
            # Ensure the returned dictionary has the keys expected by the template
            details["video_id"] = media_id
            return details
        else:
            return {"title": "Unknown Title", "artist": "Unknown Artist",
                    "thumbnail": "", "video_id": media_id}
    else:
        return get_video_info(media_id)



@bp.route('/')
@login_required
def home():
    return render_template("index.html")



@bp.route("/api/play-sequence/<song_id>/<action>")
@login_required
def api_play_sequence(song_id, action):
    """Enhanced previous/next handling with proper sequence tracking using Redis."""
    local_songs = load_local_songs_from_file()
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    user_id = session['user_id']
    history_key = f"user_history:{user_id}"
    
    try:
        # Get all history entries
        history_entries = redis_client.lrange(history_key, 0, -1)
        if not history_entries:
            return jsonify({"error": "No history found"}), 404
        
        # Find current song's position
        current_session = None
        current_seq = None
        for i, entry in enumerate(history_entries):
            data = json.loads(entry)
            if data['song_id'] == song_id:
                current_session = data['session_id']
                current_seq = data['sequence_number']
                break
        
        if not current_session:
            return jsonify({"error": "Current song not found"}), 404
        
        if action == "previous":
            # Find previous song in same session
            for entry in reversed(history_entries):
                data = json.loads(entry)
                if data['session_id'] == current_session and data['sequence_number'] < current_seq:
                    prev_id = data['song_id']
                    if prev_id.startswith("local-"):
                        meta = local_songs.get(prev_id)
                        if meta:
                            return jsonify(meta)
                        return jsonify({"error": "Local song not found"}), 404
                    return util.get_song_info(prev_id)
            return jsonify({"error": "No previous song"}), 404
        
        elif action == "next":
            # Find next song in same session
            for entry in history_entries:
                data = json.loads(entry)
                if data['session_id'] == current_session and data['sequence_number'] > current_seq:
                    next_id = data['song_id']
                    if next_id.startswith("local-"):
                        meta = local_songs.get(next_id)
                        if meta:
                            return jsonify(meta)
                        return jsonify({"error": "Local song not found"}), 404
                    return util.get_song_info(next_id)
            
            # If no next song, get recommendations
            try:
                song_info = local_songs.get(song_id) if song_id.startswith("local-") else ytmusic.get_song(song_id)
                if not song_info:
                    return jsonify({"error": "Failed to get song info"}), 404
                
                recommendations = ytmusic.get_watch_playlist(videoId=song_id, limit=5)
                if recommendations and "tracks" in recommendations:
                    recs = []
                    for track in recommendations["tracks"]:
                        if track.get("videoId") == song_id:
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
                logger.error(f"Recommendation error: {e}")
            return util.get_fallback_recommendations()
        
        return jsonify({"error": "Invalid action"}), 400
    
    except Exception as e:
        logger.error(f"Sequence error: {e}")
        return jsonify({"error": str(e)}), 500



@bp.route("/api/download/<song_id>")
@login_required
def api_download2(song_id):
    """Smart download handler that tries YouTube first for video ID-like names."""
    try:
        # Extract potential video ID
        potential_vid = song_id[6:] if song_id.startswith("local-") else song_id
        
        # First try YouTube if it looks like a video ID
        if util.is_potential_video_id(potential_vid):
            try:
                # Check if already downloaded
                existing_path = util.get_download_info(potential_vid)
                if existing_path and os.path.exists(existing_path):
                    # Get title from database
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("SELECT title FROM downloads WHERE video_id = ?", (potential_vid,))
                    row = c.fetchone()
                    conn.close()
                    
                    if row and row[0]:
                        safe_title = util.sanitize_filename(row[0])
                        download_name = f"{safe_title}.flac"
                    else:
                        download_name = f"{potential_vid}.flac"
                        
                    return send_file(
                        existing_path,
                        as_attachment=True,
                        download_name=download_name,
                    )
                
                # Try getting YouTube metadata
                info = ytmusic.get_song(potential_vid)
                title = info.get("videoDetails", {}).get("title", "Unknown")
                safe_title = util.sanitize_filename(title)
                
                # Download the file
                flac_path = util.download_flac(potential_vid , session.get('user_id') )
                if not flac_path:
                    raise Exception("Download failed")
                    
                # Send file with proper name
                download_name = f"{safe_title}.flac" if safe_title else f"{potential_vid}.flac"
                return send_file(
                    flac_path,
                    as_attachment=True,
                    download_name=download_name
                )
                
            except Exception as yt_error:
                logger.info(f"YouTube attempt failed for {potential_vid}: {yt_error}")
                # If YouTube fails, continue to local file handling
        
        # Handle as local file
        if song_id.startswith("local-"):
            meta = local_songs.get(song_id)
            if not meta:
                return jsonify({"error": "File not found"}), 404
                
            # Get original filename
            original_name = os.path.basename(meta["path"])
            
            # Try to get a clean name from metadata
            if meta.get("title") and meta.get("artist"):
                clean_name = util.sanitize_filename(f"{meta['artist']} - {meta['title']}")
                # Keep original extension
                _, ext = os.path.splitext(original_name)
                download_name = f"{clean_name}{ext}"
            else:
                download_name = original_name
                
            return send_file(
                meta["path"],
                as_attachment=True,
                download_name=download_name
            )
            
        # If not local- prefix, treat as direct YouTube ID
        try:
            info = ytmusic.get_song(song_id)
            title = info.get("videoDetails", {}).get("title", "Unknown")
            safe_title = util.sanitize_filename(title)
            
            flac_path = util.download_flac(song_id , session.get('user_id'))
            if not flac_path:
                return jsonify({"error": "Download failed"}), 500
                
            download_name = f"{safe_title}.flac" if safe_title else f"{song_id}.flac"
            return send_file(
                flac_path,
                as_attachment=True,
                download_name=download_name
            )
            
        except Exception as e:
            logger.error(f"Download error for {song_id}: {e}")
            # Try to send file with ID if it exists
            if os.path.exists(f"music/{song_id}.flac"):
                return send_file(
                    f"music/{song_id}.flac",
                    as_attachment=True,
                    download_name=f"{song_id}.flac"
                )
            return jsonify({"error": "Download failed"}), 500
            
    except Exception as e:
        logger.error(f"Download route error: {e}")
        return jsonify({"error": str(e)}), 500

@bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint that returns a success message"""
    return jsonify({"status": "healthy", "message": "Server is running"}), 200

@bp.route('/api/artist-info/<artist_name>')
@login_required
def get_artist_info(artist_name):
    try:
        # Split multiple artists and try to get info for the primary artist
        primary_artist = artist_name.split(',')[0].strip()
        
        # Search for artist
        results = ytmusic.search(primary_artist, filter='artists')
        if not results:
            # Try a more lenient search
            results = ytmusic.search(primary_artist)
            # Filter for artist results
            results = [r for r in results if r.get('category') == 'Artists']
            
            if not results:
                logger.warning(f"No artist found for: {primary_artist}")
                # Return minimal info to prevent UI errors
                return jsonify({
                    'name': primary_artist,
                    'thumbnail': '',
                    'description': 'Artist information not available',
                    'genres': [],
                    'year': None,
                    'stats': {
                        'subscribers': '0',
                        'views': '0',
                        'monthlyListeners': '0'
                    },
                    'topSongs': [],
                    'links': {}
                })
            
        artist = results[0]
        artist_id = artist.get('browseId')
        
        if not artist_id:
            logger.warning(f"No artist ID found for: {primary_artist}")
            return jsonify({
                'name': primary_artist,
                'thumbnail': '',
                'description': 'Artist information not available',
                'genres': [],
                'year': None,
                'stats': {
                    'subscribers': '0',
                    'views': '0',
                    'monthlyListeners': '0'
                },
                'topSongs': [],
                'links': {}
            })

        # Get detailed artist info
        artist_data = ytmusic.get_artist(artist_id)
        if not artist_data:
            raise Exception("Failed to fetch artist details")

        # Rest of the processing remains same
        description = util.process_description(artist_data.get('description', ''))
        thumbnail_url = util.get_best_thumbnail(artist_data.get('thumbnails', []))
        genres = util.process_genres(artist_data)
        stats = util.get_artist_stats(artist_data)
        top_songs = util.process_top_songs(artist_data)
        links = util.process_artist_links(artist_data, artist_id)

        response = {
            'name': artist_data.get('name', primary_artist),
            'thumbnail': thumbnail_url,
            'description': description,
            'genres': genres,
            'year': util.extract_year(artist_data),
            'stats': stats,
            'topSongs': top_songs,
            'links': links
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in get_artist_info: {str(e)}")
        # Return minimal info to prevent UI errors
        return jsonify({
            'name': artist_name.split(',')[0].strip(),
            'thumbnail': '',
            'description': 'Failed to load artist information',
            'genres': [],
            'year': None,
            'stats': {
                'subscribers': '0',
                'views': '0',
                'monthlyListeners': '0'
            },
            'topSongs': [],
            'links': {}
        })
    


@bp.route("/api/search")
@login_required
def api_search():
    """
    Enhanced search endpoint that handles:
    - Regular text search
    - YouTube/YouTube Music URLs (songs, playlists, albums)
    - Direct video IDs
    - Empty queries with local songs from Redis and default songs
    """
    # Load local songs from Redis instead of a file
    local_songs = load_local_songs_from_file()
    q = request.args.get("q", "").strip()
    page = int(request.args.get("page", 0))
    limit = int(request.args.get("limit", 20))

    # Helper function to get thumbnail from Redis or fallback to external URL
    def get_thumbnail(song_id):
        song_data = redis_client.hgetall(song_id)
        if song_data and "thumbnail" in song_data:
            return song_data["thumbnail"]
        return f"https://i.ytimg.com/vi/{song_id}/hqdefault.jpg"

    # === 1. Process if input is a YouTube link ===
    if "youtube.com" in q or "youtu.be" in q:
        try:
            # Handle as a playlist if the URL contains "playlist" or "list="
            if "playlist" in q or "list=" in q:
                ydl_opts = {
                    'quiet': True,
                    'extract_flat': True,
                    'force_generic_extractor': False
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(q, download=False)
                    if info and 'entries' in info:
                        results = []
                        seen_ids = set()
                        for entry in info['entries']:
                            if not entry:
                                continue
                            video_id = entry.get('id')
                            if not video_id or video_id in seen_ids:
                                continue
                            thumbnail = get_thumbnail(video_id)
                            result = {
                                "id": video_id,
                                "title": entry.get('title', 'Unknown'),
                                "artist": entry.get('artist', entry.get('uploader', 'Unknown Artist')),
                                "album": entry.get('album', ''),
                                "duration": int(entry.get('duration', 0)),
                                "thumbnail": thumbnail
                            }
                            results.append(result)
                            seen_ids.add(video_id)
                            if len(results) >= limit:
                                break
                        start = page * limit
                        end = start + limit
                        return jsonify(results[start:end])

            # Handle as a single video/song
            parsed = urlparse(q)
            params = parse_qs(parsed.query)
            video_id = None
            if "youtu.be" in q:
                video_id = q.split("/")[-1].split("?")[0]
            elif "v" in params:
                video_id = params["v"][0]

            if video_id:
                ydl_opts = {
                    'quiet': True,
                    'extract_flat': False,
                    'force_generic_extractor': False
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(f"https://youtube.com/watch?v={video_id}", download=False)
                    if info:
                        thumbnail = get_thumbnail(video_id)
                        result = [{
                            "id": video_id,
                            "title": info.get('title', 'Unknown'),
                            "artist": info.get('artist', info.get('uploader', 'Unknown Artist')),
                            "album": info.get('album', ''),
                            "duration": int(info.get('duration', 0)),
                            "thumbnail": thumbnail
                        }]
                        return jsonify(result)
        except Exception as e:
            logger.error(f"Error processing link '{q}': {e}")
            # Fall through to regular search if link processing fails

    # === 2. Process if input is a direct YouTube video ID ===
    if re.match(r'^[a-zA-Z0-9_-]{11}$', q):
        try:
            ydl_opts = {
                'quiet': True,
                'extract_flat': False,
                'force_generic_extractor': False
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://youtube.com/watch?v={q}", download=False)
                if info:
                    thumbnail = get_thumbnail(q)
                    result = [{
                        "id": q,
                        "title": info.get('title', 'Unknown'),
                        "artist": info.get('artist', info.get('uploader', 'Unknown Artist')),
                        "album": info.get('album', ''),
                        "duration": int(info.get('duration', 0)),
                        "thumbnail": thumbnail
                    }]
                    return jsonify(result)
        except Exception as e:
            logger.error(f"Error processing video ID '{q}': {e}")
            return jsonify([])

    # === 3. Process empty query: Combine local songs from Redis and default songs ===
    if not q:
        combined_res = []
        seen_ids = set()
        # Add all local songs from Redis
        local_songs_list = list(local_songs.values())
        for song in local_songs_list:
            if song["id"] not in seen_ids:
                combined_res.append(song)
                seen_ids.add(song["id"])
        # Add default songs
        default_songs = get_default_songs()
        for song in default_songs:
            if song["id"] not in seen_ids:
                song["thumbnail"] = get_thumbnail(song["id"])
                combined_res.append(song)
                seen_ids.add(song["id"])
        start = page * limit
        end = start + limit
        return jsonify(combined_res[start:end])

    # === 4. Regular text search ===
    seen_ids = set()
    combined_res = []

    # Add local songs matching the query
    local_res = util.filter_local_songs(q)
    for song in local_res:
        if song["id"] not in seen_ids:
            combined_res.append(song)
            seen_ids.add(song["id"])

    # Define helper function to search using yt-dlp
    def search_ytdlp():
        try:
            ydl_opts = {
                'quiet': True,
                'extract_flat': True,
                'force_generic_extractor': False
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"ytsearch{limit}:{q}", download=False)
                results = []
                if info and 'entries' in info:
                    for entry in info['entries']:
                        if not entry:
                            continue
                        video_id = entry.get('id')
                        if not video_id:
                            continue
                        thumbnail = get_thumbnail(video_id)
                        results.append({
                            "id": video_id,
                            "title": entry.get('title', 'Unknown'),
                            "artist": entry.get('artist', entry.get('uploader', 'Unknown Artist')),
                            "album": entry.get('album', ''),
                            "duration": int(entry.get('duration', 0)),
                            "thumbnail": thumbnail
                        })
                return results
        except Exception as e:
            logger.error(f"Error in YouTube search via yt-dlp: {e}")
            return []

    # Run both search methods concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_utmusic = executor.submit(util.search_songs, q)
        future_yt = executor.submit(search_ytdlp)
        utmusic_results = future_utmusic.result()
        yt_results = future_yt.result()

    # Merge results with YTMusic results first, then yt-dlp results
    for song in (utmusic_results + yt_results):
        if song["id"] not in seen_ids:
            # Check Redis for thumbnail if not already set
            if not song.get("thumbnail"):
                song["thumbnail"] = get_thumbnail(song["id"])
            combined_res.append(song)
            seen_ids.add(song["id"])

    start = page * limit
    end = start + limit
    return jsonify(combined_res[start:end])
@bp.route("/api/song-info/<song_id>")
@login_required
def api_song_info(song_id):
    """Fetch metadata for a single song (local or YouTube) with Redis thumbnail priority."""
    local_songs = load_local_songs_from_file()
    
    # Check Redis first for any song (local or downloaded)
    song_data = redis_client.hgetall(song_id)
    if song_data and "thumbnail" in song_data:
        return jsonify({
            "id": song_id,
            "title": song_data.get("title", "Unknown"),
            "artist": song_data.get("artist", "Unknown Artist"),
            "album": song_data.get("album", ""),
            "thumbnail": song_data["thumbnail"],
            "duration": int(song_data.get("duration", 0))
        })

    # Handle local songs not yet in song_data
    if song_id.startswith("local-"):
        meta = local_songs.get(song_id)
        if not meta:
            return jsonify({"error": "Local song not found"}), 404
        return jsonify(meta)

    # Fetch from YouTube API with fallback thumbnail
    try:
        if song_id not in song_cache:
            data = ytmusic.get_song(song_id)
            song_cache[song_id] = data
        else:
            data = song_cache[song_id]

        vd = data.get("videoDetails", {})
        title = vd.get("title", "Unknown")
        length = int(vd.get("lengthSeconds", 0))
        artist = vd.get("author", "Unknown Artist")
        if data.get("artists"):
            artist = data["artists"][0].get("name", "Unknown Artist")
        album = data.get("album", {}).get("name", "")
        thumb = f"https://i.ytimg.com/vi/{song_id}/hqdefault.jpg"  # Fallback

        return jsonify({
            "id": song_id,
            "title": title,
            "artist": artist,
            "album": album,
            "thumbnail": thumb,
            "duration": length
        })
    except Exception as e:
        logger.error(f"api_song_info error: {e}")
        return jsonify({"error": str(e)}), 400



@bp.route('/api/get-recommendations/<song_id>')
@login_required
def api_get_recommendations(song_id):
    """Get recommendations using available YTMusic methods."""
    try:
        if song_id.startswith("local-"):
            return util.get_local_song_recommendations(song_id)

        recommendations = []
        seen_songs = set()

        # 1. Get current song info
        song_info = ytmusic.get_song(song_id)
        if not song_info:
            return util.fallback_recommendations()

        # 2. Get related songs from watch playlist (most reliable method)
        try:
            watch_playlist = ytmusic.get_watch_playlist(videoId=song_id, limit=25)
            if watch_playlist and "tracks" in watch_playlist:
                for track in watch_playlist["tracks"]:
                    if util.add_recommendation(track, recommendations, seen_songs, song_id):
                        if len(recommendations) >= 5:
                            logger.info(f"Found recommendations from watch playlist for {song_id}")
                            return jsonify(recommendations)
        except Exception as e:
            logger.warning(f"Watch playlist error: {e}")

        # 3. If not enough recommendations, try artist's songs
        if len(recommendations) < 5 and "artists" in song_info:
            try:
                artist_id = song_info["artists"][0].get("id")
                if artist_id:
                    artist_data = ytmusic.get_artist(artist_id)
                    if artist_data and "songs" in artist_data:
                        artist_songs = list(artist_data["songs"])
                        random.shuffle(artist_songs)
                        for track in artist_songs[:10]:
                            if util.add_recommendation(track, recommendations, seen_songs, song_id):
                                if len(recommendations) >= 5:
                                    break
            except Exception as e:
                logger.warning(f"Artist recommendations error: {e}")

        # 4. If still not enough, search for similar songs
        if len(recommendations) < 5:
            try:
                # Get current song details for search
                title = song_info.get("videoDetails", {}).get("title", "")
                artist = song_info.get("videoDetails", {}).get("author", "")
                if title and artist:
                    # Search with song title and artist
                    search_results = ytmusic.search(f"{title} {artist}", filter="songs", limit=10)
                    for track in search_results:
                        if util.add_recommendation(track, recommendations, seen_songs, song_id):
                            if len(recommendations) >= 5:
                                break
            except Exception as e:
                logger.warning(f"Search recommendations error: {e}")

        # 5. Last resort: get popular songs
        if len(recommendations) < 5:
            try:
                popular_songs = ytmusic.search("popular songs", filter="songs", limit=10)
                for track in popular_songs:
                    if util.add_recommendation(track, recommendations, seen_songs, song_id):
                        if len(recommendations) >= 5:
                            break
            except Exception as e:
                logger.warning(f"Popular songs error: {e}")

        # Ensure we have at least some recommendations
        if not recommendations:
            return util.fallback_recommendations()

        # Shuffle for variety and return
        random.shuffle(recommendations)
        return jsonify(recommendations[:5])

    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        return util.fallback_recommendations()
    


@bp.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        if 'step' not in session:
            # Initial email submission
            email = request.form.get('email')
            if not email:
                return render_template_string(
                    var_templates.RESET_PASSWORD_HTML,
                    step='email',
                    error='Email is required'
                )
                
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('SELECT id FROM users WHERE email = ?', (email,))
            user = c.fetchone()
            conn.close()
            
            if not user:
                return render_template_string(
                    var_templates.RESET_PASSWORD_HTML,
                    step='email',
                    error='No account found with this email'
                )
                
            # Send verification code
            otp = util.generate_otp()
            util.store_otp(email, otp, 'reset')
            var_templates.send_forgot_password_email(email , otp)
            
            session['reset_email'] = email
            session['step'] = 'verify'
            
            return render_template_string(
                var_templates.RESET_PASSWORD_HTML,
                step='verify',
                email=email
            )
            
        elif session['step'] == 'verify':
            # OTP verification
            email = session.get('reset_email')
            otp = request.form.get('otp')
            
            if not util.verify_otp(email, otp, 'reset'):
                return render_template_string(
                    var_templates.RESET_PASSWORD_HTML,
                    step='verify',
                    email=email,
                    error='Invalid or expired code'
                )
                
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('SELECT id FROM users WHERE email = ?', (email,))
            user_id = c.fetchone()[0]
            conn.close()
            
            session['step'] = 'new_password'
            session['user_id_reset'] = user_id
            
            return render_template_string(
                var_templates.RESET_PASSWORD_HTML,
                step='new_password',
                user_id=user_id
            )
            
        elif session['step'] == 'new_password':
            # Password update
            new_password = request.form.get('new_password')
            user_id = session.get('user_id_reset')
            if not new_password:
                session.pop('step', None)
                session.pop('reset_email', None)
                session.pop('user_id_reset', None)
            
            if len(new_password) < 6:
                return render_template_string(
                    var_templates.RESET_PASSWORD_HTML,
                    step='new_password',
                    user_id=user_id,
                    error='Password must be at least 6 characters'
                )
                
            password_hash = bcrypt.hashpw(
                new_password.encode(),
                bcrypt.gensalt()
            ).decode()
            
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute(
                'UPDATE users SET password_hash = ? WHERE id = ?',
                (password_hash, user_id)
            )
            conn.commit()
            conn.close()
            
            # Clear session
            session.pop('step', None)
            session.pop('reset_email', None)
            session.pop('user_id_reset', None)
            
            return redirect(url_for('playback.login'))
            
    return render_template_string(var_templates.RESET_PASSWORD_HTML, step='email')



@bp.route('/forgot_username', methods=['GET', 'POST'])
def forgot_username():
    if request.method == 'POST':
        if 'step' not in session:
            email = request.form.get('email')
            if not email:
                return render_template_string(
                    var_templates.FORGOT_USERNAME_HTML,
                    step='email',
                    error='Email is required'
                )
                
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('SELECT username FROM users WHERE email = ?', (email,))
            user = c.fetchone()
            conn.close()
            
            if not user:
                return render_template_string(
                    var_templates.FORGOT_USERNAME_HTML,
                    step='email',
                    error='No account found with this email'
                )
                
            # Send username directly to email
            var_templates.send_forgot_username_email(email , user)
            
            return render_template_string(
                var_templates.LOGIN_HTML,
                login_step='initial',
                success='Username has been sent to your email'
            )
            
    return render_template_string(var_templates.FORGOT_USERNAME_HTML, step='email')




@bp.route('/logout')
def logout():
    if 'user_id' in session and 'session_token' in session:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            DELETE FROM active_sessions 
            WHERE user_id = ? AND session_token = ?
        """, (session['user_id'], session['session_token']))
        conn.commit()
        conn.close()
    
    session.clear()
    return redirect(url_for('playback.login'))


@bp.route('/login', methods=['GET', 'POST'])
def login():
    # First check if user is already logged in with valid session
    if 'user_id' in session and 'session_token' in session:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        try:
            c.execute("""
                SELECT 1 FROM active_sessions 
                WHERE user_id = ? AND session_token = ? 
                AND expires_at > CURRENT_TIMESTAMP
            """, (session['user_id'], session['session_token']))
            valid_session = c.fetchone()
            
            if valid_session:
                return redirect(url_for('playback.home'))
            else:
                # Clear invalid session
                session.clear()
        except Exception as e:
            logger.error(f"Session check error: {e}")
            session.clear()
        finally:
            conn.close()
    
    if request.method == 'POST':
        login_id = request.form.get('login_id')
        password = request.form.get('password')
        
        # Input validation
        if not login_id or not password:
            return render_template_string(
                var_templates.LOGIN_HTML,
                login_step='initial',
                error='Both login ID and password are required'
            )
        
        # Clear any existing temporary login data
        if 'temp_login' in session:
            session.pop('temp_login')
            
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        try:
            # Check if login_id is email or username
            c.execute("""
                SELECT id, password_hash, twofa_method, email,
                       (SELECT COUNT(*) FROM active_sessions 
                        WHERE user_id = users.id 
                        AND expires_at > CURRENT_TIMESTAMP) as active_sessions
                FROM users 
                WHERE email = ? OR username = ?
            """, (login_id, login_id))
            
            user = c.fetchone()
            
            if user and bcrypt.checkpw(password.encode(), user[1].encode()):
                user_id, password_hash, twofa_method, email, active_sessions = user
                
                # Check for existing active sessions
                if active_sessions > 0:
                    # Terminate other sessions
                    c.execute("""
                        DELETE FROM active_sessions 
                        WHERE user_id = ? 
                    """, (user_id,))
                    conn.commit()
                    logger.info(f"Terminated existing sessions for user {user_id}")
                
                if twofa_method != 'none':  # 2FA enabled
                    # Generate temporary login token
                    token = secrets.token_urlsafe(32)
                    session['temp_login'] = {
                        'token': token,
                        'user_id': user_id,
                        'twofa_method': twofa_method
                    }
                    
                    if twofa_method == 'email':
                        # Generate and send OTP
                        otp = util.generate_otp()
                        util.store_otp(email, otp, 'login')
                        util.send_email(
                            email, 
                            'Login Verification', 
                            f'Your verification code is: {otp}'
                        )
                    
                    return render_template_string(
                        var_templates.LOGIN_HTML,
                        login_step='2fa',
                        login_token=token,
                        twofa_method=twofa_method
                    )
                
                # No 2FA - direct login
                session_token = secrets.token_urlsafe(32)
                expires_at = datetime.now() + timedelta(days=7)
                
                # Create new session
                c.execute("""
                    INSERT INTO active_sessions (user_id, session_token, expires_at)
                    VALUES (?, ?, ?)
                """, (user_id, session_token, expires_at))
                
                conn.commit()
                
                # Set session cookies
                session.clear()
                session['user_id'] = user_id
                session['session_token'] = session_token
                session['last_session_check'] = int(time.time())
                
                return redirect(url_for('playback.home'))
            else:
                # Invalid credentials - add small delay to prevent brute force
                time.sleep(random.uniform(0.1, 0.3))
                return render_template_string(
                    var_templates.LOGIN_HTML,
                    login_step='initial',
                    error='Invalid credentials'
                )
                
        except Exception as e:
            logger.error(f"Login error: {e}")
            return render_template_string(
                var_templates.LOGIN_HTML,
                login_step='initial',
                error='An error occurred during login'
            )
        finally:
            conn.close()
            
    # GET request - show login form
    return render_template_string(
        var_templates.LOGIN_HTML, 
        login_step='initial'
    )

@bp.route("/favicon.ico")
def set_fake():
    return "not there..."
@bp.route('/login_verify', methods=['POST'])
def login_verify():
    if 'temp_login' not in session:
        return redirect(url_for('playback.login'))
        
    temp = session['temp_login']
    otp = request.form.get('otp')
    token = request.form.get('login_token')
    
    if token != temp['token']:
        return render_template_string(
            var_templates.LOGIN_HTML,
            login_step='2fa',
            error='Invalid session'
        )
    
    if temp['twofa_method'] == 'email':
        # Verify email OTP
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT email FROM users WHERE id = ?', (temp['user_id'],))
        email = c.fetchone()[0]
        conn.close()
        
        if not util.verify_otp(email, otp, 'login'):
            return render_template_string(
                var_templates.LOGIN_HTML,
                login_step='2fa',
                login_token=token,
                twofa_method=temp['twofa_method'],
                error='Invalid or expired code'
            )
    
    # Create new session after successful 2FA
    session_token = secrets.token_urlsafe(32)
    expires_at = datetime.now() + timedelta(days=7)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO active_sessions (user_id, session_token, expires_at)
        VALUES (?, ?, ?)
    """, (temp['user_id'], session_token, expires_at))
    conn.commit()
    conn.close()
    
    # Clear temporary login data and set permanent session
    session.clear()
    session['user_id'] = temp['user_id']
    session['session_token'] = session_token
    
    return redirect(url_for('playback.home'))

# Add this to your Flask routes file

@bp.route('/terms-register', methods=['GET'])
def terms_register():
    """Route to get terms and conditions for the registration page."""
    try:
        # Read terms from a file
        with open(os.path.join(os.getcwd() , "terms" , "terms_register.txt"), 'r') as file:
            terms_content = file.read()
            
        # Format the content with HTML for better display
        formatted_terms = f"""
        <div class="space-y-4">
            <h4 class="text-xl font-semibold mb-3">Sangeet Premium Terms of Service</h4>
            {terms_content}
        </div>
        """
        return formatted_terms
    except FileNotFoundError:
        # Fallback to default terms if file not found
        print("terms file not found...")
        return "Something missing our side cant load terms right now please dont register if you aren't sure.."
        


@bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        full_name = request.form.get('full_name')
        password = request.form.get('password')
        
        # Basic validation
        if not all([email, username, full_name, password]):
            return render_template_string(
                var_templates.REGISTER_HTML,
                register_step='initial',
                error='All fields are required'
            )
            
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Check if email/username exists
        c.execute("""
            SELECT 1 FROM users 
            WHERE email = ? OR username = ?
        """, (email, username))
        
        if c.fetchone():
            conn.close()
            return render_template_string(
                var_templates.REGISTER_HTML,
                register_step='initial',
                error='Email or username already exists'
            )
            
        # Generate verification OTP
        otp = util.generate_otp()
        util.store_otp(email, otp, 'register')
        
        # Store registration data in session
        token = secrets.token_urlsafe(32)
        session['register_data'] = {
            'token': token,
            'email': email,
            'username': username,
            'full_name': full_name,
            'password': password
        }
        
        # Send verification email
        var_templates.send_register_otp_email(email , otp)
        
        return render_template_string(
            var_templates.REGISTER_HTML,
            register_step='verify',
            email=email,
            register_token=token
        )
        
    return render_template_string(var_templates.REGISTER_HTML, register_step='initial')



@bp.route('/register/verify', methods=['POST'])
def register_verify():
    if 'register_data' not in session:
        return redirect(url_for('playback.register'))
        
    data = session['register_data']
    otp = request.form.get('otp')
    token = request.form.get('register_token')
    
    if token != data['token']:
        return render_template_string(
            var_templates.REGISTER_HTML,
            register_step='verify',
            error='Invalid session'
        )
        
    if not util.verify_otp(data['email'], otp, 'register'):
        return render_template_string(
            var_templates.REGISTER_HTML,
            register_step='verify',
            email=data['email'],
            register_token=token,
            error='Invalid or expired code'
        )
        
    # Create user account
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    password_hash = bcrypt.hashpw(
        data['password'].encode(), 
        bcrypt.gensalt()
    ).decode()
    
    c.execute("""
        INSERT INTO users (email, username, full_name, password_hash)
        VALUES (?, ?, ?, ?)
    """, (data['email'], data['username'], data['full_name'], password_hash))
    
    user_id = c.lastrowid
    
    # Generate session token and create active session
    session_token = secrets.token_urlsafe(32)
    expires_at = datetime.now() + timedelta(days=7)
    
    c.execute("""
        INSERT INTO active_sessions (user_id, session_token, expires_at)
        VALUES (?, ?, ?)
    """, (user_id, session_token, expires_at))
    
    conn.commit()
    conn.close()
    
    # Set both required session variables
    session.pop('register_data')
    session['user_id'] = user_id
    session['session_token'] = session_token
    
    return redirect(url_for('playback.home'))
@bp.route("/api/insights")
@login_required
def get_insights():
    """Get comprehensive listening insights for the current user."""
    user_id = session['user_id']
    conn = sqlite3.connect(DB_PATH)  # Still using SQLite for listening_history
    c = conn.cursor()
    
    try:
        insights = {
            "overview": util.get_overview_stats(c, user_id),
            "recent_activity": util.get_recent_activity(c, user_id),
            "top_artists": util.get_top_artists(c, user_id),
            "listening_patterns": util.get_listening_patterns(c, user_id),
            "completion_rates": util.get_completion_rates(c, user_id)
        }
        return jsonify(insights)
    finally:
        conn.close()


@bp.route("/api/listen/start", methods=["POST"])
@login_required
def api_listen_start():
    """Start a new listening session for the current user."""
    try:
        user_id = session['user_id']
        data = request.json
        if not data or not all(k in data for k in ["songId", "title", "artist"]):
            return jsonify({"error": "Missing required fields"}), 400

        session_id = util.generate_session_id()
        listen_id = util.record_listen_start(
            user_id=user_id,
            song_id=data["songId"],
            title=data["title"],
            artist=data["artist"],
            session_id=session_id
        )

        return jsonify({
            "status": "success",
            "listenId": listen_id,
            "sessionId": session_id
        })
    except Exception as e:
        logger.error(f"Listen start error: {e}")
        return jsonify({"error": str(e)}), 500

@bp.route("/api/listen/end", methods=["POST"])
@login_required
def api_listen_end():
    """End a listening session for the current user."""
    try:
        user_id = session['user_id']
        data = request.json
        if not data or "listenId" not in data:
            return jsonify({"error": "Missing listenId"}), 400
            
        listen_id = int(data["listenId"])
        
        # Verify the listen_id belongs to the user
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT user_id FROM listening_history WHERE id = ?", (listen_id,))
        row = c.fetchone()
        if not row or row[0] != user_id:
            return jsonify({"error": "Invalid listen ID"}), 403
        
        duration = data.get("duration", 0)
        listened_duration = data.get("listenedDuration", 0)

        util.record_listen_end(
            listen_id=listen_id,
            duration=duration,
            listened_duration=listened_duration
        )

        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Listen end error: {e}")
        return jsonify({"error": str(e)}), 500


@bp.route("/api/proxy/image")
@login_required
def proxy_image():
    """Proxy endpoint for fetching images with CORS headers."""
    url = request.args.get('url')
    if not url:
        return "No URL provided", 400

    # Validate URL is from trusted domains
    allowed_domains = {'i.ytimg.com', 'img.youtube.com'}
    try:
        domain = urlparse(url).netloc
        if domain not in allowed_domains:
            return "Invalid domain", 403
    except:
        return "Invalid URL", 400

    content, content_type = util.fetch_image(url)
    if content:
        response = make_response(content)
        response.headers['Content-Type'] = content_type
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Cache-Control'] = 'public, max-age=31536000'
        return response

    return "Failed to fetch image", 500




@bp.route("/get-extension")
def get_extension():
    return render_template("extension.html")



@bp.route('/download/extension')
def download_extension():
    """
    Handle the download of the extension zip file.
    Replace 'extension.zip' with your actual file name.
    """
    try:
        return send_file(
            os.path.join(os.getcwd() , "payloads" , "extension" , "sangeet-premium.zip"),
            as_attachment=True,
            download_name='sangeet-premium.zip',
            mimetype='application/zip'
        )
    except Exception as e:
        return f"Error: File not found", 404
@bp.route('/sangeet-download/<video_id>')
def sangeet_download(video_id):
    user_id = session.get('user_id')  # Get user ID from session

    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    # Get metadata from ytmusicapi
    try:
        info = ytmusic.get_song(video_id)
        vd = info.get("videoDetails", {})
        title = vd.get("title", "Unknown")
        artist = vd.get("author", "Unknown Artist")
        album = info.get("album", {}).get("name", "")
        thumbnail = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"

        safe_title = util.sanitize_filename(title) or "Track"
        dl_name = f"{safe_title}.flac"

        return render_template('download.html', 
            title=title, 
            artist=artist, 
            album=album, 
            thumbnail=thumbnail,
            dl_name=dl_name,
            video_id=video_id
        )

    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({"error": "Failed to process download"}), 500
    


@bp.route('/download-file/<video_id>')
def download_file(video_id):
    user_id = session.get('user_id')  # Get user ID from session

    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    flac_path = util.download_flac(video_id, user_id)
    if not flac_path:
        return jsonify({"error": "Failed to process FLAC"}), 500
    
    # Get metadata for file name
    try:
        info = ytmusic.get_song(video_id)
        vd = info.get("videoDetails", {})
        title = vd.get("title", "Unknown")
        safe_title = util.sanitize_filename(title) or "Track"
        dl_name = f"{safe_title}.flac"

        return send_file(flac_path, as_attachment=True, download_name=dl_name)

    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({"error": "Failed to process download"}), 500
    





@bp.route("/data/download/icons/<type>")
def icons(type):
    if type == "download":
        with open(os.path.join(os.getcwd() , "assets" , "favicons" , "download" , "fav.txt") , "r") as fav:
         data = fav.read()
        fav.close()
        return jsonify({"base64": data})
    elif type == "sangeet-home":
        return send_file(os.path.join(os.getcwd() , "assets" , "gifs" , "sangeet" , "index.gif"))
    elif type == "get-extension":
        with open(os.path.join(os.getcwd() , "assets" , "favicons" , "get-extension" , "fav.txt") , "r") as fav:
         data = fav.read()
        fav.close()
        return jsonify({"base64": data})
    elif type == "login-system-login":
        with open(os.path.join(os.getcwd() , "assets" , "favicons" , "login-system" , "login.txt") , "r") as fav:
         data = fav.read()
        fav.close()
        return jsonify({"base64": data})
    elif type == "login-system-register":
        with open(os.path.join(os.getcwd() , "assets" , "favicons" , "login-system" , "register.txt") , "r") as fav:
         data = fav.read()
        fav.close()
        return jsonify({"base64": data})
    elif type == "login-system-forgot":
        with open(os.path.join(os.getcwd() , "assets" , "favicons" , "login-system" , "forgot.txt") , "r") as fav:
         data = fav.read()
        fav.close()
        return jsonify({"base64": data})
    else:
        with open(os.path.join(os.getcwd() , "assets" , "favicons" , "genric" , "fav.txt") , "r") as fav:
         data = fav.read()
        fav.close()
        return jsonify({"base64": data})
    

@bp.route("/embed/<song_id>")
def embed_player(song_id):
    """Serve an embeddable player for a specific song."""
    try:
        # Get customization options
        size = request.args.get("size", "normal")  # small, normal, large
        theme = request.args.get("theme", "default")  # default, purple, blue, dark
        autoplay = request.args.get("autoplay", "false").lower() == "true"
        
        # Get song info
        if song_id.startswith("local-"):
            meta = local_songs.get(song_id)
            if not meta:
                return jsonify({"error": "Song not found"}), 404
            song_info = {
                "id": song_id,
                "title": meta["title"],
                "artist": meta["artist"],
                "thumbnail": meta["thumbnail"] or url_for('playback.static', filename='images/default-cover.jpg'),
                "duration": meta["duration"]
            }
            # For local files, use the local stream endpoint
            stream_url = url_for('playback.api_stream_local', song_id=song_id)
        else:
            try:
                if song_id not in song_cache:
                    data = ytmusic.get_song(song_id)
                    song_cache[song_id] = data
                else:
                    data = song_cache[song_id]

                vd = data.get("videoDetails", {})
                song_info = {
                    "id": song_id,
                    "title": vd.get("title", "Unknown"),
                    "artist": vd.get("author", "Unknown Artist"),
                    "thumbnail": f"https://i.ytimg.com/vi/{song_id}/hqdefault.jpg",
                    "duration": int(vd.get("lengthSeconds", 0))
                }
                
                # Download/get FLAC stream URL
                flac_path = util.download_flac(song_id , session.get('user_id')  )
                if not flac_path:
                    return jsonify({"error": "Failed to process audio"}), 500
                
                stream_url = url_for('playback.stream_file', song_id=song_id)
                
            except Exception as e:
                logger.error(f"Error getting song info: {e}")
                return jsonify({"error": "Failed to get song info"}), 500

        # Generate the embed HTML
        return render_template(
            "embed.html",
            song=song_info,
            size=size,
            theme=theme,
            autoplay=autoplay,
            stream_url=stream_url,
            host_url=SERVER_DOMAIN
        )
        
    except Exception as e:
        logger.error(f"Embed error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@bp.route("/play/<song_id>")
def play_song(song_id):
    """Redirect to main player with the selected song."""
    return redirect(url_for('playback.home', song=song_id))

@bp.before_request
def before_request():
    # Clear expired sessions
    util.cleanup_expired_sessions()
    
    # Check if current session is expired
    if 'user_id' in session and 'session_token' in session:
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
            session.clear()

@bp.route("/api/embed-code/<song_id>")
def get_embed_code(song_id):
    """Get the iframe code for embedding a song."""
    try:
        size = request.args.get("size", "normal")
        theme = request.args.get("theme", "default")
        autoplay = request.args.get("autoplay", "false")
        
        # Set dimensions based on size
        dimensions = {
            "small": (320, 160),
            "normal": (400, 200),
            "large": (500, 240)
        }
        width, height = dimensions.get(size, dimensions["normal"])
        
        # Generate iframe code
        embed_url = f"{request.host_url.rstrip('/')}/embed/{song_id}?size={size}&theme={theme}&autoplay={autoplay}"
        iframe_code = (
            f'<iframe src="{embed_url}" '
            f'width="{width}" height="{height}" '
            'frameborder="0" allowtransparency="true" '
            'allow="encrypted-media; autoplay" loading="lazy">'
            '</iframe>'
        )
        
        return jsonify({
            "code": iframe_code,
            "url": embed_url,
            "width": width,
            "height": height
        })
        
    except Exception as e:
        logger.error(f"Embed code error: {e}")
        return jsonify({"error": "Internal server error"}), 500
    

# @bp.route('/api/queue', methods=['GET'])
# @login_required
# def api_queue():
#     user_id = session['user_id']
#     limit = int(request.args.get('limit', 5))
#     offset = int(request.args.get('offset', 0))
#     history = util.get_play_history(user_id, limit, offset)
#     return jsonify(history)

@bp.route('/api/queue', methods=['GET'])
@login_required
def api_queue():
    """Return user's play history with pagination."""
    user_id = session['user_id']
    limit = int(request.args.get('limit', 5))  # Default to 5 items
    offset = int(request.args.get('offset', 0))
    history_key = f"user_history:{user_id}"

    try:
        # Fetch history entries from Redis with pagination
        history_entries = redis_client.lrange(history_key, offset, offset + limit - 1)
        history = []

        for entry in history_entries:
            data = json.loads(entry)
            song_id = data['song_id']

            # Fetch metadata from Redis or API
            song_data = redis_client.hgetall(song_id)
            if not song_data:
                if song_id.startswith("local-"):
                    meta = local_songs.get(song_id, {})
                else:
                    meta = get_video_info(song_id)
                song_data = {
                    "title": meta.get("title", "Unknown"),
                    "artist": meta.get("artist", "Unknown Artist"),
                    "thumbnail": meta.get("thumbnail", f"https://i.ytimg.com/vi/{song_id}/hqdefault.jpg")
                }
                redis_client.hset(song_id, mapping=song_data)

            history.append({
                "id": song_id,
                "title": song_data.get("title", "Unknown"),
                "artist": song_data.get("artist", "Unknown Artist"),
                "thumbnail": song_data.get("thumbnail", ""),
                "played_at": data["played_at"],
                "session_id": data["session_id"],
                "sequence_number": data["sequence_number"]
            })

        return jsonify(history)
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return jsonify({"error": str(e)}), 500

@bp.route("/api/stats")
@login_required
def api_stats():
    """Return user-specific usage stats using Redis."""
    local_songs = load_local_songs_from_file()
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    user_id = session['user_id']
    stats_key = f"user_stats:{user_id}"
    history_key = f"user_history:{user_id}"
    
    try:
        # Get basic stats from Redis
        total_plays = int(redis_client.hget(stats_key, "total_plays") or 0)
        total_listened_time = int(redis_client.hget(stats_key, "total_listened_time") or 0)
        favorite_song_id = redis_client.hget(stats_key, "favorite_song_id")
        favorite_artist = redis_client.hget(stats_key, "favorite_artist")
        
        # Get unique songs from history
        history_entries = redis_client.lrange(history_key, 0, -1)
        unique_songs = len(set(json.loads(entry)['song_id'] for entry in history_entries))
        
        # Downloads still use SQLite (assuming not migrating downloads to Redis)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM user_downloads WHERE user_id = ?", (user_id,))
        total_downloads = c.fetchone()[0]
        
        download_size = 0
        c.execute("SELECT path FROM user_downloads WHERE user_id = ?", (user_id,))
        for (path,) in c.fetchall():
            try:
                if os.path.exists(path):
                    download_size += os.path.getsize(path)
            except:
                continue
        
        # Top 5 most played
        song_counts = {}
        for entry in history_entries:
            song_id = json.loads(entry)['song_id']
            song_counts[song_id] = song_counts.get(song_id, 0) + 1
        
        top_songs = []
        for sid, count in sorted(song_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            try:
                if sid.startswith("local-"):
                    if sid in local_songs:
                        meta = local_songs[sid]
                        top_songs.append({
                            "id": sid,
                            "title": meta["title"],
                            "artist": meta["artist"],
                            "plays": count
                        })
                else:
                    if sid not in song_cache:
                        data = ytmusic.get_song(sid)
                        song_cache[sid] = data
                    else:
                        data = song_cache[sid]
                    vd = data.get("videoDetails", {})
                    title = vd.get("title", "Unknown")
                    artist = vd.get("author", "Unknown Artist")
                    if data.get("artists"):
                        artist = data["artists"][0].get("name", artist)
                    top_songs.append({
                        "id": sid,
                        "title": title,
                        "artist": artist,
                        "plays": count
                    })
            except:
                continue
        
        conn.close()
        
        return jsonify({
            "total_plays": total_plays,
            "total_listened_time": total_listened_time,
            "unique_songs": unique_songs,
            "total_downloads": total_downloads,
            "download_size": download_size,
            "top_songs": top_songs,
            "favorite_song": favorite_song_id,
            "favorite_artist": favorite_artist,
            "local_songs_count": len(local_songs)
        })
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({"error": str(e)}), 500

@bp.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@bp.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500



# @bp.route("/api/stream/<song_id>")
# # @login_required
# def api_stream(song_id):
#     """Obtain a streaming URL for a given song_id (local or YouTube)."""
#     user_id = session['user_id']
#     if not user_id == None:
#       util.record_song(song_id, user_id)

#     if song_id.startswith("local-"):
#         return jsonify({
#             "local": True,
#             "url": f"/api/stream-local/{song_id}"
#         })

#     flac_path = util.download_flac(song_id, user_id)
#     if not flac_path:
#         return jsonify({"error": "Download/FLAC conversion failed"}), 500

#     return jsonify({
#         "url": f"/api/stream-file/{song_id}",
#         "local": False
#     })

@bp.route("/api/stream/<song_id>")
@login_required
def api_stream(song_id):
    user_id = session['user_id']
    client_timestamp = request.args.get('timestamp')
    util.record_song(song_id, user_id, client_timestamp)

    if song_id.startswith("local-"):
        return jsonify({"local": True, "url": f"/api/stream-local/{song_id}"})

    flac_path = util.download_flac(song_id, user_id)
    if not flac_path:
        return jsonify({"error": "Download/FLAC conversion failed"}), 500

    return jsonify({"url": f"/api/stream-file/{song_id}", "local": False})

@bp.route("/api/download/<song_id>")
@login_required
def api_download(song_id):
    """Provide the FLAC file as a downloadable attachment."""
    local_songs = load_local_songs_from_file()
    user_id = session['user_id']
    
    if song_id.startswith("local-"):
        meta = local_songs.get(song_id)
        if not meta:
            return jsonify({"error": "Local file not found"}), 404
        filename = os.path.basename(meta["path"])
        return send_file(
            meta["path"],
            as_attachment=True,
            download_name=filename
        )

    flac_path = util.download_flac(song_id, user_id)
    if not flac_path:
        return jsonify({"error": "Failed to process FLAC"}), 500

    # Get metadata and record download
    try:
        info = ytmusic.get_song(song_id)
        vd = info.get("videoDetails", {})
        title = vd.get("title", "Unknown")
        artist = vd.get("author", "Unknown Artist")
        album = info.get("album", {}).get("name", "")
        
        # Note: download is already recorded in download_flac function
        
        safe_title = util.sanitize_filename(title) or "Track"
        dl_name = f"{safe_title}.flac"

        return send_file(
            flac_path,
            as_attachment=True,
            download_name=dl_name
        )
    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({"error": "Failed to process download"}), 500



@bp.route("/api/similar/<song_id>")
@login_required
def api_similar(song_id):
    """Get similar songs based on current song, fallback if local or error."""
    try:
        if song_id.startswith("local-"):
            return util.fallback_recommendations()

        song_info = ytmusic.get_song(song_id)
        if not song_info:
            return util.fallback_recommendations()

        similar_songs = []

        # 1) Radio-based suggestions
        try:
            radio = ytmusic.get_watch_playlist(song_id, limit=10)
            if radio and "tracks" in radio:
                for track in radio["tracks"]:
                    vid = track.get("videoId")
                    if vid and vid != song_id and track.get("isAvailable") != False:
                        title = track.get("title", "Unknown")
                        art = "Unknown Artist"
                        if "artists" in track and track["artists"]:
                            art = track["artists"][0].get("name", art)
                        alb = ""
                        if "album" in track and isinstance(track["album"], dict):
                            alb = track["album"].get("name", "")
                        thumb = f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
                        dur = track.get("duration_seconds", 0)
                        similar_songs.append({
                            "id": vid,
                            "title": title,
                            "artist": art,
                            "album": alb,
                            "thumbnail": thumb,
                            "duration": dur
                        })
        except Exception as e:
            logger.warning(f"Radio recommendations failed: {e}")

        # 2) Artist's other songs
        try:
            if "artists" in song_info and song_info["artists"]:
                artist_id = song_info["artists"][0].get("id")
                if artist_id:
                    artist_songs = ytmusic.get_artist(artist_id)
                    if artist_songs and "songs" in artist_songs:
                        random_songs = random.sample(
                            artist_songs["songs"], 
                            min(3, len(artist_songs["songs"]))
                        )
                        for track in random_songs:
                            vid = track.get("videoId")
                            if vid and vid != song_id:
                                title = track.get("title", "Unknown")
                                art = "Unknown Artist"
                                if "artists" in track and track["artists"]:
                                    art = track["artists"][0].get("name", art)
                                alb = ""
                                if "album" in track and isinstance(track["album"], dict):
                                    alb = track["album"].get("name", "")
                                thumb = f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
                                dur = track.get("duration_seconds", 0)
                                similar_songs.append({
                                    "id": vid,
                                    "title": title,
                                    "artist": art,
                                    "album": alb,
                                    "thumbnail": thumb,
                                    "duration": dur
                                })
        except Exception as e:
            logger.warning(f"Artist recommendations failed: {e}")

        # 3) If not enough, do a quick search
        if len(similar_songs) < 5:
            try:
                t = song_info.get("videoDetails", {}).get("title", "")
                a = song_info.get("videoDetails", {}).get("author", "")
                results = ytmusic.search(f"{t} {a}", filter="songs", limit=5)
                for track in results:
                    vid = track.get("videoId")
                    if vid and vid != song_id:
                        ttitle = track.get("title", "Unknown")
                        tartist = "Unknown Artist"
                        if "artists" in track and track["artists"]:
                            tartist = track["artists"][0].get("name", tartist)
                        talbum = ""
                        if "album" in track and isinstance(track["album"], dict):
                            talbum = track["album"].get("name", "")
                        thumb = f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
                        dur = track.get("duration_seconds", 0)
                        similar_songs.append({
                            "id": vid,
                            "title": ttitle,
                            "artist": tartist,
                            "album": talbum,
                            "thumbnail": thumb,
                            "duration": dur
                        })
            except Exception as e:
                logger.warning(f"Search recommendations failed: {e}")

        random.shuffle(similar_songs)
        return jsonify(similar_songs[:5])

    except Exception as e:
        logger.error(f"Similar songs error: {e}")
        return util.fallback_recommendations()


@bp.route("/api/history/clear", methods=["POST"])
@login_required
def api_clear_history():
    """Clear play history."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM history")
        conn.commit()
        conn.close()
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Clear history error: {e}")
        return jsonify({"error": str(e)}), 500


@bp.route("/api/downloads/clear", methods=["POST"])
@login_required
def api_clear_downloads():
    """Clear all downloads in DB and remove files from disk."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT path FROM downloads")
        paths = [row[0] for row in c.fetchall()]

        for path in paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                logger.warning(f"Failed to delete file {path}: {e}")

        c.execute("DELETE FROM downloads")
        conn.commit()
        conn.close()
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Clear downloads error: {e}")
        return jsonify({"error": str(e)}), 500



@bp.route("/api/stream-file/<song_id>")
# @login_required
def stream_file(song_id):
    """Serve the FLAC file with range requests for seeking."""
    flac_path = os.path.join(os.getenv("MUSIC_PATH"), f"{song_id}.flac")
    if not os.path.exists(flac_path):
        return jsonify({"error": "File not found"}), 404

    try:
        return send_file(flac_path)

    except Exception as e:
        logger.error(f"stream_file error: {e}")
        return jsonify({"error": str(e)}), 500


@bp.route("/api/stream-local/<song_id>")
# @login_required
def api_stream_local(song_id):
    """Serve a local file with range requests."""
    local_songs = load_local_songs_from_file()
    meta = local_songs.get(song_id)
    if not meta:
        return jsonify({"error": "Local file not found"}), 404
    path = meta["path"]
    if not os.path.isfile(path):
        return jsonify({"error": "File not on disk"}), 404

    try:
       return send_file(path)
    except Exception as e:
        logger.error(f"stream_local error: {e}")
        return jsonify({"error": str(e)}), 500



@bp.route("/api/lyrics/<song_id>")
def api_lyrics(song_id):
    """Return YTMusic lyrics array or [] for local/no lyrics."""
    
    # Handle local songs
    if song_id.startswith("local-"):
        try:
            # Load the locals.json file
            with open(os.path.join(os.getcwd() , "locals" , "local.json"), 'r') as f:
                locals_data = json.load(f)
            
            # Get the song data for this local ID
            song_data = locals_data.get(song_id)
            if not song_data or "path" not in song_data:
                logger.info(f"No path found for local song: {song_id}")
                return jsonify([])
            
            # Extract video ID from the filename
            path = song_data["path"]
            filename = path.split("/")[-1]
            video_id = filename.split(".")[0]  # Remove .flac extension
            
            # Use the extracted video ID instead of local-id
            song_id = video_id
            logger.info(f"Using video ID {video_id} from local song")
        except Exception as e:
            logger.error(f"Error extracting video ID from local song: {e}")
            return jsonify([])
    
    # Check if lyrics exist in SQLite cache
    cached_lyrics = get_cached_lyrics(song_id)
    if cached_lyrics:
        logger.info(f"Returning cached lyrics for {song_id}")
        return jsonify(cached_lyrics)
    
    # If not in cache, fetch from YTMusic
    try:
        watch_pl = ytmusic.get_watch_playlist(song_id)
        if not watch_pl or "lyrics" not in watch_pl:
            lbid = watch_pl.get("lyrics")
            if not lbid:
                return jsonify([])
        else:
            lbid = watch_pl["lyrics"]
        
        data = ytmusic.get_lyrics(lbid)
        if data and "lyrics" in data:
            lines = data["lyrics"].split("\n")
            lines.append("\n Sangeet Premium")
            
            # Save to SQLite cache
            cache_lyrics(song_id, lines)
            
            return jsonify(lines)
        return jsonify([])
    except Exception as e:
        logger.error(f"api_lyrics error: {e}")
        return jsonify([])
@bp.route("/api/downloads")
@login_required
def api_downloads():
    """Return all downloads that exist on disk."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT video_id, title, artist, album, downloaded_at
        FROM downloads
        ORDER BY downloaded_at DESC
    """)
    rows = c.fetchall()
    conn.close()

    items = []
    for vid, title, artist, album, ts in rows:
        flac_path = os.path.join(os.getenv("MUSIC_PATH"), f"{vid}.flac")
        if os.path.exists(flac_path):
            thumb = f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
            items.append({
                "id": vid,
                "title": title,
                "artist": artist,
                "album": album,
                "downloaded_at": ts,
                "thumbnail": thumb
            })
    return jsonify(items)






@bp.route('/api/resend-otp', methods=['POST'])
def resend_otp():
    try:
        data = request.json
        purpose = None
        email = None
        
        if 'login_token' in data:
            if 'temp_login' not in session:
                return jsonify({'error': 'Invalid session'}), 400
            temp = session['temp_login']
            
            if data['login_token'] != temp['token']:
                return jsonify({'error': 'Invalid token'}), 400
                
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('SELECT email FROM users WHERE id = ?', (temp['user_id'],))
            email = c.fetchone()[0]
            conn.close()
            purpose = 'login'
            
        elif 'register_token' in data:
            if 'register_data' not in session:
                return jsonify({'error': 'Invalid session'}), 400
            reg = session['register_data']
            
            if data['register_token'] != reg['token']:
                return jsonify({'error': 'Invalid token'}), 400
                
            email = reg['email']
            purpose = 'register'
            
        else:
            return jsonify({'error': 'Invalid request'}), 400
            
        # Generate and send new OTP
        otp = util.generate_otp()
        util.store_otp(email, otp, purpose)
        util.send_email(
            email,
            'New Verification Code',
            f'Your new verification code is: {otp}'
        )
        
        return jsonify({'status': 'success'})
        
    except Exception as e:
        logger.error(f"Resend OTP error: {e}")
        return jsonify({'error': str(e)}), 500
@bp.route("/api/session-status")
def check_session_status():
    """Check if current session is still valid."""
    if 'user_id' not in session or 'session_token' not in session:
        return jsonify({
            "valid": False,
            "reason": "no_session"
        }), 401

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Check if this specific session token is still valid
        c.execute("""
            SELECT COUNT(*) FROM active_sessions
            WHERE user_id = ? AND session_token = ?
            AND expires_at > CURRENT_TIMESTAMP
        """, (session['user_id'], session['session_token']))

        valid_session = c.fetchone()[0] > 0

        # Get total active sessions for this user
        c.execute("""
            SELECT COUNT(*) FROM active_sessions
            WHERE user_id = ?
            AND session_token != ?
            AND expires_at > CURRENT_TIMESTAMP
        """, (session['user_id'], session['session_token']))

        other_sessions = c.fetchone()[0] > 0

        conn.close()

        if not valid_session:
            reason = "logged_out_elsewhere" if other_sessions else "expired"
            return jsonify({
                "valid": False,
                "reason": reason
            }), 401

        return jsonify({"valid": True})

    except Exception as e:
        logger.error(f"Session check error: {e}")
        return jsonify({
            "valid": False,
            "reason": "error"
        }), 500
@bp.route("/design/<type>")
def design(type):
    """Serves CSS files from the design directory."""
    css_dir = os.path.join(os.getcwd(), "design", "css")
    allowed_files = {
        "index": "index.css",
        "embed": "embed.css",
        "settings" : "settings.css",
        "share" : "share.css",
        "report-issues" : "report_issues.css",
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




@bp.route('/share/open/<media_id>')
def share(media_id):
    if os.path.exists(os.path.join(os.getcwd() , os.getenv("MUSIC_PATH") , f"{media_id}.flac")):
        info = get_media_info(media_id)
        share_url = request.url_root + f"?song={media_id}"
        return render_template('share.html', share_url=share_url, **info )
    else:
        return render_template('share_not_found.html' )







@bp.route('/stream2/open/<media_id>')
def stream2(media_id):
    local_data = load_local_songs_from_file()
    if media_id.startswith("local-"):
        # Stream local file using send_file
        details = local_data.get(media_id)
        if details and "path" in details:
            file_path = details["path"]
            print(file_path)
            return send_file(file_path)
        else:
            return "Local file not found", 404
    else:
        # For non-local videos, redirect to a streaming endpoint (replace with your logic)
        return redirect(f"/api/stream-file/{media_id}")




















# Get all playlists for the user
@bp.route('/api/playlists', methods=['GET'])
@login_required
def get_playlists():
    user_id = session['user_id']
    conn = sqlite3.connect(PLAYLIST_DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT p.id, p.name, COUNT(ps.song_id) as song_count
                 FROM playlists p
                 LEFT JOIN playlist_songs ps ON p.id = ps.playlist_id
                 WHERE p.user_id = ?
                 GROUP BY p.id, p.name''', (user_id,))
    playlists = [{'id': row[0], 'name': row[1], 'song_count': row[2]} for row in c.fetchall()]
    conn.close()
    return jsonify(playlists)

# Create a new playlist
@bp.route('/api/playlists/create', methods=['POST'])
@login_required
def create_playlist():
    user_id = session['user_id']
    data = request.json
    name = data.get('name')
    if not name:
        return jsonify({'error': 'Playlist name is required'}), 400

    conn = sqlite3.connect(PLAYLIST_DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO playlists (user_id, name, share_id) VALUES (?, ?, ?)',
              (user_id, name, secrets.token_urlsafe(16)))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'}), 201

# Add song to playlist
@bp.route('/api/playlists/add_song', methods=['POST'])
@login_required
def add_song_to_playlist():
    user_id = session['user_id']
    data = request.json
    playlist_id = data.get('playlist_id')
    song_id = data.get('song_id')

    if not playlist_id or not song_id:
        return jsonify({'error': 'Playlist ID and Song ID are required'}), 400

    conn = sqlite3.connect(PLAYLIST_DB_PATH)
    c = conn.cursor()
    c.execute('SELECT user_id FROM playlists WHERE id = ?', (playlist_id,))
    result = c.fetchone()
    if not result or result[0] != user_id:
        return jsonify({'error': 'Unauthorized or playlist not found'}), 403

    c.execute('INSERT OR IGNORE INTO playlist_songs (playlist_id, song_id) VALUES (?, ?)',
              (playlist_id, song_id))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})
@bp.route('/api/playlists/<int:playlist_id>/songs', methods=['GET'])
@login_required
def get_playlist_songs(playlist_id):
    user_id = session['user_id']
    conn = sqlite3.connect(PLAYLIST_DB_PATH)
    c = conn.cursor()
    c.execute('SELECT user_id FROM playlists WHERE id = ?', (playlist_id,))
    result = c.fetchone()
    if not result or result[0] != user_id:
        return jsonify({'error': 'Unauthorized or playlist not found'}), 403

    c.execute('''SELECT ps.song_id
                 FROM playlist_songs ps
                 WHERE ps.playlist_id = ?''', (playlist_id,))
    song_ids = [row[0] for row in c.fetchall()]

    songs = []
    for song_id in song_ids:
        if song_id.startswith('local-'):
            meta = local_songs.get(song_id, {})
            songs.append({
                'id': song_id,
                'title': meta.get('title', 'Unknown'),
                'artist': meta.get('artist', 'Unknown Artist'),
                'thumbnail': meta.get('thumbnail', '/static/images/default-cover.jpg')
            })
        else:
            try:
                info = ytmusic.get_song(song_id)
                vd = info.get('videoDetails', {})
                thumbnails = vd.get('thumbnail', {}).get('thumbnails', [])
                if thumbnails:
                    # Select the highest-resolution thumbnail
                    best_thumbnail = max(thumbnails, key=lambda x: x.get('width', 0) * x.get('height', 0))
                    thumbnail_url = best_thumbnail['url']
                else:
                    # Fallback to standard YouTube thumbnail
                    thumbnail_url = f"https://i.ytimg.com/vi/{song_id}/hqdefault.jpg"
                songs.append({
                    'id': song_id,
                    'title': vd.get('title', 'Unknown'),
                    'artist': vd.get('author', 'Unknown Artist'),
                    'thumbnail': thumbnail_url
                })
            except Exception as e:
                logger.error(f"Error fetching song info for {song_id}: {e}")
                songs.append({
                    'id': song_id,
                    'title': 'Unknown',
                    'artist': 'Unknown Artist',
                    'thumbnail': '/static/images/default-cover.jpg'
                })
    conn.close()
    return jsonify(songs)

# Share playlist (make public)
@bp.route('/api/playlists/<int:playlist_id>/share', methods=['POST'])
@login_required
def share_playlist(playlist_id):
    user_id = session['user_id']
    conn = sqlite3.connect(PLAYLIST_DB_PATH)
    c = conn.cursor()
    c.execute('SELECT share_id FROM playlists WHERE id = ? AND user_id = ?', (playlist_id, user_id))
    result = c.fetchone()
    if not result:
        return jsonify({'error': 'Unauthorized or playlist not found'}), 403

    share_id = result[0]
    c.execute('UPDATE playlists SET is_public = 1 WHERE id = ?', (playlist_id,))
    conn.commit()
    conn.close()
    return jsonify({'share_id': share_id})

# Import shared playlist
@bp.route('/playlists/share/<share_id>', methods=['GET'])
@login_required
def import_shared_playlist(share_id):
    user_id = session['user_id']
    conn = sqlite3.connect(PLAYLIST_DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, name FROM playlists WHERE share_id = ? AND is_public = 1', (share_id,))
    playlist = c.fetchone()
    if not playlist:
        return jsonify({'error': 'Playlist not found or not public'}), 404

    playlist_id, name = playlist
    c.execute('SELECT song_id FROM playlist_songs WHERE playlist_id = ?', (playlist_id,))
    song_ids = [row[0] for row in c.fetchall()]

    # Create a new playlist for the user
    new_share_id = secrets.token_urlsafe(16)
    c.execute('INSERT INTO playlists (user_id, name, share_id) VALUES (?, ?, ?)',
              (user_id, f"{name} (Imported)", new_share_id))
    new_playlist_id = c.lastrowid

    for song_id in song_ids:
        c.execute('INSERT INTO playlist_songs (playlist_id, song_id) VALUES (?, ?)',
                  (new_playlist_id, song_id))
    conn.commit()
    conn.close()

    return redirect('/?playlist_added=true')



@bp.route("/api/random-song")
@login_required
def api_random_song():
    """Return the latest song from the user's history, with a fallback to a default song."""
    try:
        # Load local songs from Redis to populate local_songs dictionary
        load_local_songs_from_file()
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        user_id = session['user_id']
        history_key = f"user_history:{user_id}"

        # Get the latest song from user's history
        latest_entry = redis_client.lindex(history_key, 0)
        if latest_entry:
            song_data = json.loads(latest_entry)
            song_id = song_data["song_id"]

            # Handle local song
            if song_id.startswith("local-"):
                meta = local_songs.get(song_id)
                if meta:
                    return jsonify(meta)
                else:
                    logger.warning(f"Local song {song_id} not found in local_songs")

            # Handle YouTube song
            else:
                try:
                    info = ytmusic.get_song(song_id)
                    vd = info.get("videoDetails", {})
                    return jsonify({
                        "id": song_id,
                        "title": vd.get("title", "Unknown"),
                        "artist": vd.get("author", "Unknown Artist"),
                        "album": "",
                        "thumbnail": f"https://i.ytimg.com/vi/{song_id}/hqdefault.jpg"
                    })
                except Exception as e:
                    logger.error(f"Error getting YouTube song info for {song_id}: {e}")

        # Fallback to default song if no history or metadata retrieval fails
        default_songs = ["dQw4w9WgXcQ", "kJQP7kiw5Fk", "9bZkp7q19f0"]
        random_id = random.choice(default_songs)
        info = ytmusic.get_song(random_id)
        vd = info.get("videoDetails", {})
        return jsonify({
            "id": random_id,
            "title": vd.get("title", "Unknown"),
            "artist": vd.get("author", "Unknown Artist"),
            "album": "",
            "thumbnail": f"https://i.ytimg.com/vi/{random_id}/hqdefault.jpg"
        })

    except Exception as e:
        logger.error(f"Error in api_random_song: {e}")
        return jsonify({"error": str(e)}), 500



@bp.route('/api/user-issues', methods=['GET'])
@login_required
def get_user_issues():
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "Authentication required"}), 401

        conn = get_issues_db()
        cursor = conn.cursor()

        # Ensure tables exist (redundant if get_issues_db does it, but safe)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_issues'")
        if not cursor.fetchone():
            conn.close()
            return jsonify([]) # No issues table yet

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='issue_comments'")
        if not cursor.fetchone():
            conn.close()
             # If comments table is missing, just return issues without comments
            cursor = conn.cursor() # Re-open cursor needed after close
            cursor.execute("""
                SELECT id, topic, details, status, created_at, updated_at
                FROM user_issues
                WHERE user_id = ?
                ORDER BY created_at DESC
            """, (user_id,))
            issues_raw = cursor.fetchall()
            conn.close()
            issues_list = [dict(row) for row in issues_raw]
             # Add an empty comments list to each issue for consistent structure
            for issue in issues_list:
                 issue['comments'] = []
            return jsonify(issues_list)


        # --- Fetch issues AND their comments for the logged-in user ---
        cursor.execute("""
            SELECT
                i.id, i.topic, i.details, i.status, i.created_at, i.updated_at,
                c.id as comment_id, c.comment, c.is_admin, c.created_at as comment_created_at
            FROM user_issues i
            LEFT JOIN issue_comments c ON i.id = c.issue_id
            WHERE i.user_id = ?
            ORDER BY i.created_at DESC, c.created_at ASC
        """, (user_id,))

        rows = cursor.fetchall()
        conn.close()

        issues_dict = {}
        for row_dict in map(dict, rows): # Convert each row to a dict
            issue_id = row_dict['id']

            # If it's the first time seeing this issue, initialize it
            if issue_id not in issues_dict:
                issues_dict[issue_id] = {
                    'id': issue_id,
                    'topic': row_dict['topic'],
                    'details': row_dict['details'],
                    'status': row_dict['status'],
                    'created_at': row_dict['created_at'],
                    'updated_at': row_dict['updated_at'],
                    'comments': []
                }

            # If there's a comment in this row, add it
            if row_dict['comment_id'] is not None:
                 comment_data = {
                     'id': row_dict['comment_id'],
                     'content': row_dict['comment'], # Use 'content' to match JS later
                     'is_admin': bool(row_dict['is_admin']), # Ensure boolean
                     'created_at': row_dict['comment_created_at']
                 }
                 # Avoid adding duplicate comments if LEFT JOIN produces multiple rows for the same comment (shouldn't happen with ORDER BY but safer)
                 if not any(c['id'] == comment_data['id'] for c in issues_dict[issue_id]['comments']):
                    issues_dict[issue_id]['comments'].append(comment_data)


        # Convert the dictionary back to a list, maintaining original sort order (by issue creation)
        issues_list = list(issues_dict.values())

        return jsonify(issues_list)

    except sqlite3.Error as e:
         logger.error(f"Database error retrieving user issues: {e}")
         # Be cautious about exposing detailed DB errors
         return jsonify({"error": "Database error fetching issues."}), 500
    except Exception as e:
        logger.error(f"Error retrieving user issues: {e}")
        # Return empty array for frontend robustness
        return jsonify([]) # Return empty list on other errors too


@bp.route('/api/issues/<int:issue_id>/reply', methods=['POST'])
@login_required
def reply_to_issue(issue_id):
    """Allows a logged-in user to add a comment to their own issue."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    data = request.get_json()
    comment_text = data.get('comment')

    if not comment_text:
        return jsonify({"error": "Comment text cannot be empty"}), 400

    try:
        conn = get_issues_db()
        cursor = conn.cursor()

        # --- Security Check: Verify the user owns this issue ---
        cursor.execute("SELECT user_id FROM user_issues WHERE id = ?", (issue_id,))
        issue_owner = cursor.fetchone()

        if not issue_owner:
            conn.close()
            return jsonify({"error": "Issue not found"}), 404

        if issue_owner['user_id'] != user_id:
            conn.close()
            # User is trying to comment on someone else's issue
            return jsonify({"error": "Unauthorized to comment on this issue"}), 403
        # --- End Security Check ---

        # Insert the user's comment
        cursor.execute("""
            INSERT INTO issue_comments (issue_id, user_id, is_admin, comment, created_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (issue_id, user_id, 0, comment_text)) # is_admin = 0 for user replies

        # Update the issue's updated_at timestamp
        cursor.execute("""
            UPDATE user_issues SET updated_at = CURRENT_TIMESTAMP WHERE id = ?
        """, (issue_id,))

        conn.commit()
        new_comment_id = cursor.lastrowid # Get the ID of the new comment
        conn.close()

        # Return success and potentially the new comment ID
        return jsonify({
            "success": True,
            "message": "Reply submitted successfully",
            "comment_id": new_comment_id
            })

    except sqlite3.Error as e:
        logger.error(f"Database error adding user reply to issue {issue_id}: {e}")
        if conn: conn.close()
        return jsonify({"error": f"Database error: {e}"}), 500
    except Exception as e:
        logger.error(f"Error adding user reply to issue {issue_id}: {e}")
        if conn: conn.close()
        return jsonify({"error": "An unexpected error occurred"}), 500
@bp.route('/api/report-issue', methods=['POST'])
@login_required
def report_issue():
    try:
        # Get user ID from session
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "Authentication required"}), 401
        
        # Parse request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        topic = data.get('topic')
        details = data.get('details')
        
        # Validate input
        if not topic or not details:
            return jsonify({"error": "Topic and details are required"}), 400
        
        # Insert into database
        conn = sqlite3.connect(DB_PATH_ISSUES)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO user_issues (user_id, topic, details, status, created_at, updated_at)
            VALUES (?, ?, ?, 'Open', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """, (user_id, topic, details))
        
        # Get the ID of the inserted issue
        issue_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "success": True,
            "message": "Issue reported successfully",
            "issue_id": issue_id
        })
    
    except Exception as e:
        logger.error(f"Error reporting issue: {e}")
        return jsonify({"error": str(e)}), 500














# routes.py
# ... other imports ...
from flask import session, render_template, request, redirect, url_for, jsonify, abort, make_response
import sqlite3
import os
from functools import wraps
import secrets
from datetime import datetime, timedelta



# Helper to get DB connections
def get_issues_db():
    conn = sqlite3.connect(DB_PATH_ISSUES)
    conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
     # Create tables if they don't exist
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_issues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            topic TEXT NOT NULL,
            details TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Open',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS issue_comments (
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             issue_id INTEGER NOT NULL,
             user_id INTEGER, -- Nullable for admin comments
             is_admin BOOLEAN DEFAULT 0,
             comment TEXT NOT NULL,
             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
             FOREIGN KEY (issue_id) REFERENCES user_issues(id) ON DELETE CASCADE -- Add cascade delete
        )
    """)
    conn.commit()
    return conn

def get_main_db():
    conn = sqlite3.connect(DB_PATH_MAIN)
    conn.row_factory = sqlite3.Row
    return conn

# Decorator for admin routes/APIs
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_authed'):
            # For API calls, return 403 Forbidden or 401 Unauthorized
            if request.path.startswith('/api/admin'):
                 return jsonify({"error": "Admin authentication required"}), 401
            # For regular page loads, redirect to the password entry
            return redirect(url_for('playback.view_admin_issues'))
        return f(*args, **kwargs)
    return decorated_function




@bp.route('/view/issues', methods=['GET', 'POST'])
def view_admin_issues():
    error = None
    session_expired = request.args.get('session_expired')

    if request.method == 'POST':
        password = request.form.get('password')
        if password == ADMIN_PASSWORD:
            session['admin_authed'] = True
             # Redirect to the GET version of the same route after successful login
            return redirect(url_for('playback.view_admin_issues'))
        else:
            error = "Invalid Password"
            session.pop('admin_authed', None) # Ensure flag is removed on failure
    elif request.method == 'GET':
         # Check if session is expired (coming from JS redirect)
        if session_expired:
             error = "Session expired or logged out. Please re-enter password."
             session.pop('admin_authed', None) # Clear potentially stale flag

        # If admin is authenticated in the session, show the dashboard
        if session.get('admin_authed'):
            return render_template('admin_issues.html')
        # Otherwise, always show the password prompt (covers initial visit, failed attempts, logout)

    # Render the password prompt page/template string if not authenticated or POST failed
    # Using a simple template string here for brevity
    password_prompt_html = """
    <!DOCTYPE html><html><head><title>Admin Login</title>
    <style>
        body { font-family: sans-serif; background-color: #1f2937; color: #e5e7eb; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
        .login-container { background-color: #374151; padding: 30px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.3); text-align: center; width: 300px; }
        h2 { margin-top: 0; color: #f9fafb; }
        label { display: block; margin-bottom: 5px; font-weight: 500; color: #d1d5db; }
        input[type='password'] { width: 90%; padding: 10px; margin-bottom: 15px; border: 1px solid #4b5563; border-radius: 4px; background-color: #4b5563; color: #e5e7eb; }
        input[type='submit'] { background-color: #3b82f6; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        input[type='submit']:hover { background-color: #2563eb; }
        .error { color: #f87171; margin-top: 15px; font-weight: bold; }
    </style>
    </head><body>
    <div class="login-container">
        <h2>Admin Issue Dashboard</h2>
        <form method="post">
            <label for="password">Enter Admin Password:</label>
            <input type="password" id="password" name="password" required>
            <br>
            <input type="submit" value="Login">
            {% if error %}
                <p class="error">{{ error }}</p>
            {% endif %}
        </form>
    </div>
    </body></html>
    """
    return render_template_string(password_prompt_html, error=error)

# --- API Endpoints for Admin Dashboard ---

@bp.route('/api/admin/issues', methods=['GET'])
@admin_required
def api_get_admin_issues():
    """Fetches all issues with associated user information."""
    issues = []
    users_cache = {} # Simple cache for usernames during this request

    try:
        conn_issues = get_issues_db()
        cursor_issues = conn_issues.cursor()
        cursor_issues.execute("""
            SELECT i.id, i.user_id, i.topic, i.details, i.status, i.created_at, i.updated_at
            FROM user_issues i
            ORDER BY i.created_at DESC
        """)
        all_issues_raw = cursor_issues.fetchall()
        conn_issues.close()

        # Get unique user IDs
        user_ids = list(set(issue['user_id'] for issue in all_issues_raw))

        # Fetch usernames from main DB if there are issues
        if user_ids:
             conn_main = get_main_db()
             cursor_main = conn_main.cursor()
             # Use parameter substitution safely
             placeholders = ','.join('?' for _ in user_ids)
             query = f"SELECT id, username FROM users WHERE id IN ({placeholders})"
             cursor_main.execute(query, user_ids)
             user_rows = cursor_main.fetchall()
             users_cache = {user['id']: user['username'] for user in user_rows}
             conn_main.close()

        # Combine issue data with username
        for issue_raw in all_issues_raw:
            issue_dict = dict(issue_raw) # Convert SqLite3.Row to dict
            issue_dict['username'] = users_cache.get(issue_raw['user_id'], 'Unknown User')
            issues.append(issue_dict)

        return jsonify(issues)

    except sqlite3.Error as e:
         logger.error(f"Database error fetching admin issues: {e}")
         return jsonify({"error": f"Database error: {e}"}), 500
    except Exception as e:
        logger.error(f"Error fetching admin issues: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500


@bp.route('/api/admin/issues/<int:issue_id>', methods=['GET'])
@admin_required
def api_get_admin_issue_details(issue_id):
    """Fetches details for a specific issue, including comments and usernames."""
    try:
        conn_issues = get_issues_db()
        cursor = conn_issues.cursor()

        # Fetch issue details
        cursor.execute("SELECT * FROM user_issues WHERE id = ?", (issue_id,))
        issue = cursor.fetchone()
        if not issue:
            abort(404, description="Issue not found")

        issue_dict = dict(issue)

        # Fetch comments for the issue
        cursor.execute("""
            SELECT c.id, c.user_id, c.is_admin, c.comment, c.created_at
            FROM issue_comments c
            WHERE c.issue_id = ?
            ORDER BY c.created_at ASC
        """, (issue_id,))
        comments_raw = cursor.fetchall()
        conn_issues.close() # Close issues DB connection

        # --- Fetch usernames for issue reporter and commenters ---
        user_ids_to_fetch = set()
        if issue_dict['user_id']:
            user_ids_to_fetch.add(issue_dict['user_id'])
        for comment in comments_raw:
            if comment['user_id'] and not comment['is_admin']:
                 user_ids_to_fetch.add(comment['user_id'])

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
                 conn_main.close()
            except sqlite3.Error as e:
                 logger.error(f"Database error fetching usernames for issue {issue_id}: {e}")
                 # Proceed without usernames if main DB fails
                 pass


        # Add username to issue details
        issue_dict['username'] = users_cache.get(issue_dict['user_id'], 'Unknown User')

        # Add usernames to comments
        comments_list = []
        for comment_raw in comments_raw:
             comment_dict = dict(comment_raw)
             if comment_dict['is_admin']:
                 comment_dict['username'] = None # Or set a specific Admin name if needed
             else:
                 comment_dict['username'] = users_cache.get(comment_dict['user_id'], 'Unknown User')
             comments_list.append(comment_dict)

        issue_dict['comments'] = comments_list

        return jsonify(issue_dict)

    except sqlite3.Error as e:
        logger.error(f"Database error fetching issue details for {issue_id}: {e}")
        if conn_issues: conn_issues.close()
        return jsonify({"error": f"Database error: {e}"}), 500
    except Exception as e:
        logger.error(f"Error fetching issue details for {issue_id}: {e}")
        if conn_issues: conn_issues.close()
        return jsonify({"error": "An unexpected error occurred"}), 500

@bp.route('/api/admin/issues/<int:issue_id>/comment', methods=['POST'])
@admin_required
def api_add_admin_comment(issue_id):
    """Adds an admin comment to an issue."""
    data = request.get_json()
    comment_text = data.get('comment')

    if not comment_text:
        return jsonify({"error": "Comment text is required"}), 400

    try:
        conn = get_issues_db()
        cursor = conn.cursor()

        # Check if issue exists
        cursor.execute("SELECT id FROM user_issues WHERE id = ?", (issue_id,))
        if not cursor.fetchone():
            conn.close()
            abort(404, description="Issue not found")

        cursor.execute("""
            INSERT INTO issue_comments (issue_id, user_id, is_admin, comment, created_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (issue_id, None, 1, comment_text)) # user_id is NULL for admin

        # Optionally, update the issue's updated_at timestamp
        cursor.execute("""
            UPDATE user_issues SET updated_at = CURRENT_TIMESTAMP WHERE id = ?
        """, (issue_id,))

        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": "Comment added"})

    except sqlite3.Error as e:
        logger.error(f"Database error adding admin comment to {issue_id}: {e}")
        return jsonify({"error": f"Database error: {e}"}), 500
    except Exception as e:
        logger.error(f"Error adding admin comment to {issue_id}: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

@bp.route('/api/admin/issues/<int:issue_id>/update_status', methods=['POST'])
@admin_required
def api_update_issue_status(issue_id):
    """Updates the status of an issue."""
    data = request.get_json()
    new_status = data.get('status')

    allowed_statuses = ["Open", "In Progress", "Resolved", "Closed"]
    if not new_status or new_status not in allowed_statuses:
        return jsonify({"error": f"Invalid status. Must be one of: {', '.join(allowed_statuses)}"}), 400

    try:
        conn = get_issues_db()
        cursor = conn.cursor()

        # Check if issue exists and update
        cursor.execute("""
            UPDATE user_issues
            SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (new_status, issue_id))

        if cursor.rowcount == 0:
            conn.close()
            abort(404, description="Issue not found")

        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": f"Status updated to {new_status}"})

    except sqlite3.Error as e:
        logger.error(f"Database error updating status for {issue_id}: {e}")
        return jsonify({"error": f"Database error: {e}"}), 500
    except Exception as e:
        logger.error(f"Error updating status for {issue_id}: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500


@bp.route('/api/admin/stats', methods=['GET'])
@admin_required
def api_get_admin_stats():
    """Fetches statistics about the issues."""
    try:
        conn = get_issues_db()
        cursor = conn.cursor()

        # Example stats: counts by status
        cursor.execute("SELECT status, COUNT(*) as count FROM user_issues GROUP BY status")
        rows = cursor.fetchall()
        conn.close()

        stats = {
            "total": 0,
            "open": 0,
            "in_progress": 0,
            "resolved": 0,
            "closed": 0
        }
        for row in rows:
            status_key = row['status'].lower().replace(' ', '_') # e.g., 'in_progress'
            if status_key in stats:
                 stats[status_key] = row['count']
                 stats['total'] += row['count']

        # You can add more complex stats like resolution time, changes from last week etc. here
        # For simplicity, only returning counts now.

        return jsonify(stats)

    except sqlite3.Error as e:
        logger.error(f"Database error fetching admin stats: {e}")
        return jsonify({"error": f"Database error: {e}"}), 500
    except Exception as e:
        logger.error(f"Error fetching admin stats: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

        

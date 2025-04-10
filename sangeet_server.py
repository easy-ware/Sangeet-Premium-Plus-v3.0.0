# ==============================================================================
# main.py - Sangeet Premium Server Entry Point
# ==============================================================================
# Description:
#   This script serves as the main entry point for launching the Sangeet Premium
#   Flask application. It handles:
#   - Initialization of required libraries (Flask, Colorama, logging, dotenv).
#   - Configuration loading (environment variables, YAML config).
#   - Setting up logging with request/response tracking.
#   - Creating necessary directories based on a JSON structure.
#   - Initializing application components (database, utilities, background tasks).
#   - Starting the Flask development server or a production server (Gunicorn).
#   - Optional setup for Windows startup and Cloudflare tunnels.
# ==============================================================================

# --- Standard Library Imports ---
import json
import logging
import multiprocessing
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from threading import Thread
# ------ KEYGEN ------ 
from server_side import keygen
# --- External Library Imports ---
import pyfiglet
import yaml
from colorama import init, Fore, Style  # For terminal colors
from dotenv import load_dotenv
from flask import Flask, request, has_request_context, g
from termcolor import colored # Alternative coloring library used in one place

# --- Internal Application Imports ---
# Assuming sangeet_premium is a package available in the Python path
from sangeet_premium.sangeet import playback
from sangeet_premium.utils import getffmpeg, cloudflarerun, util, download_cloudflare
from sangeet_premium.database import database
from server_side import config as co # Assuming server_side package exists

# Conditional import for Windows specific functionality
if sys.platform.startswith('win'):
    try:
        from sangeet_premium.utils import starter
    except ImportError:
        starter = None # Handle if starter module is missing
        print(f"{Fore.YELLOW}Warning: Could not import 'starter' module for Windows startup functionality.{Style.RESET_ALL}")
else:
    starter = None


# ==============================================================================
# Initializations and Global Settings
# ==============================================================================

# --- Initialize Libraries ---
init(autoreset=True)  # Initialize colorama for cross-platform terminal colors
load_dotenv(dotenv_path=os.path.join(os.getcwd(), "config", ".env")) # Load .env file

# --- Logging Setup ---
# Basic logging config initially, will be enhanced by setup_logging
logger = logging.getLogger(__name__) # Get logger for this module
logging.basicConfig(level=logging.INFO)

# --- Flask App Initialization ---
app = Flask(__name__)
if os.getenv("FLASK_SECRET_KEY").lower() == "auto":
     app.secret_key = keygen.generate_secure_hex_key(64)
else:
    app.secret_key = os.getenv("FLASK_SECRET_KEY")
app.register_blueprint(playback.bp)
app.register_blueprint(co.bp) # Registering the config blueprint from server_side

# --- Flask Session Configuration ---
app.config['SESSION_COOKIE_SECURE'] = os.getenv('SESSION_COOKIE_SECURE', 'False').lower() == 'true' # Use HTTPS only?
app.config['SESSION_COOKIE_HTTPONLY'] = True # Prevent client-side JS access
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax' # Mitigate CSRF
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7) # Session duration


# ==============================================================================
# Utility Functions
# ==============================================================================

# --- Function: print_banner ---
def print_banner():
    """Displays the application banner using pyfiglet and colorama."""
    try:
        sangeet_text = pyfiglet.figlet_format("SANGEET PREMIUM", font='big')
        plus_text = pyfiglet.figlet_format("PLUS", font='small') # Adjusted font for "PLUS"
        print(f"{Fore.MAGENTA}{Style.BRIGHT}{sangeet_text}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{plus_text}{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW} ♪ Free And Open Source ♪ {Style.RESET_ALL}")
        # Read version from a file or env var if possible
        app_version = os.getenv("APP_VERSION", "3.0.0") # Example fallback
        print(f"{Fore.YELLOW}   Version: {app_version} | Made with {Fore.RED}♥{Style.RESET_ALL}{Fore.YELLOW} by Easy Ware {Style.RESET_ALL}")
        project_url = os.getenv("PROJECT_URL", "https://github.com/easy-ware/Sangeet-Premium-Plus-v3.0.0") # Use PROJECT_URL from .env
        if project_url:
            print(f"{Fore.GREEN}   GitHub: {project_url}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}   Starting server at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}{'='*80}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error displaying banner: {e}{Style.RESET_ALL}")

# --- Function: create_directories_from_json ---
def create_directories_from_json(json_file):
    """Creates directories specified in a JSON structure file."""
    try:
        if not os.path.exists(json_file):
            logger.error(f"Directory structure file not found: {json_file}")
            return
        with open(json_file, 'r', encoding='utf-8') as f:
            dir_structure = json.load(f)

        base_dir = os.getcwd()
        for dir_info in dir_structure.get('directories', []):
            dir_name = dir_info.get('name')
            if not dir_name: continue

            dir_path = os.path.join(base_dir, dir_name)
            try:
                os.makedirs(dir_path, exist_ok=True)
                logger.debug(f"Ensured directory exists: '{dir_path}'")
                # Create subdirectories if specified
                for subdir_name in dir_info.get('subdirs', []):
                    subdir_path = os.path.join(dir_path, subdir_name)
                    os.makedirs(subdir_path, exist_ok=True)
                    logger.debug(f"Ensured subdirectory exists: '{subdir_path}'")
            except OSError as e:
                logger.error(f"Error creating directory '{dir_path}' or its subdirs: {e}")
            except Exception as e:
                 logger.error(f"Unexpected error processing directory entry {dir_info}: {e}")

    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in directory structure file: {json_file}")
    except Exception as e:
        logger.exception(f"Error reading or processing directory structure file {json_file}: {e}")

# --- Function: start_local_songs_refresh ---
def start_local_songs_refresh(flask_app):
    """Starts a background thread that periodically calls util.load_local_songs."""
    def refresh_loop():
        logger.info("Local songs refresh thread started.")
        while True:
            try:
                # Use Flask app context to ensure utilities work correctly if they need it
                with flask_app.app_context():
                    logger.debug("Refreshing local songs...")
                    util.load_local_songs() # Reloads from disk/Redis into util.local_songs
                    playback.load_local_songs_from_file() # Loads from Redis into playback.local_songs
                    logger.debug("Local songs refresh complete.")
            except Exception as e:
                # Log error but keep the thread running
                logger.error(f"Error during periodic local songs refresh: {e}", exc_info=False) # Set exc_info=True for traceback
            # Wait before next refresh
            time.sleep(int(os.getenv("LOCAL_SONG_REFRESH_INTERVAL", "30"))) # Configurable interval (default 30s)

    refresh_thread = Thread(target=refresh_loop, name="LocalSongRefresher", daemon=True)
    refresh_thread.start()

# --- Function: init_app ---
def init_app(flask_app):
    """Initialize application components like background tasks."""
    # Start background tasks
    start_local_songs_refresh(flask_app)
    logger.info("Application background tasks initialized.")
    # Add other initializations if needed


# ==============================================================================
# Logging Setup Function
# ==============================================================================

# --- Function: setup_logging ---
def setup_logging(flask_app, log_level_str='INFO'):
    """Configures Flask logging with console and rotating file handlers."""
    try:
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    except AttributeError:
        log_level = logging.INFO
        print(f"{Fore.YELLOW}Warning: Invalid log level '{log_level_str}' in config. Defaulting to INFO.{Style.RESET_ALL}")

    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"sangeet_server_{datetime.now():%Y-%m-%d}.log")

    # Custom Formatter Class
    class ServerLogFormatter(logging.Formatter):
        COLORS = {
            'DEBUG': Fore.CYAN,
            'INFO': Fore.GREEN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.RED + Style.BRIGHT
        }
        RESET = Style.RESET_ALL

        def __init__(self, use_colors=False):
            # Define format strings
            self.default_format = f"%(asctime)s | %(levelname)-8s | %(message)s"
            self.request_format = f"%(asctime)s | %(levelname_colored)s | [%(request_id)s] %(method)s %(path)s → %(status_code)s%(duration)s"
            # Set date format
            self.datefmt = '%Y-%m-%d %H:%M:%S'
            super().__init__(fmt=self.default_format, datefmt=self.datefmt)
            self.use_colors = use_colors

        def format(self, record):
            # Add color to level name if requested
            record.levelname_colored = record.levelname
            if self.use_colors:
                color = self.COLORS.get(record.levelname, '')
                record.levelname_colored = f"{color}{record.levelname:<8}{self.RESET}"
            else:
                record.levelname_colored = f"{record.levelname:<8}"

            # Format log record based on context
            log_fmt = self.default_format
            if has_request_context():
                # Ensure request_id exists in g
                if not hasattr(g, 'request_id'):
                    g.request_id = str(uuid.uuid4())[:6] # Generate short request ID

                # Populate record attributes needed for request format
                record.request_id = g.request_id
                record.method = request.method
                record.path = request.path
                record.status_code = getattr(record, 'status_code', '-') # Status code added later
                record.duration = ''
                if hasattr(g, 'start_time'):
                    duration_ms = int((datetime.now() - g.start_time).total_seconds() * 1000)
                    record.duration = f" ({duration_ms}ms)"

                log_fmt = self.request_format

                # Add request/response data if present (limit length)
                extra_info = ""
                req_data = getattr(record, 'request_data', getattr(g, 'request_data', None))
                res_data = getattr(record, 'response_data', None)
                if req_data and request.method != 'GET': # Don't log body for GET
                     extra_info += f"\n      Request:  {req_data[:200]}" # Limit length
                if res_data:
                     extra_info += f"\n      Response: {res_data[:200]}" # Limit length

                # Use the base class default formatter with the chosen format string
                self._style._fmt = log_fmt # Temporarily set the format string
                formatted_message = super().format(record)
                self._style._fmt = self.default_format # Reset to default
                return formatted_message + extra_info # Append extra info separately

            else:
                # Non-request context logging
                self._style._fmt = log_fmt
                formatted_message = super().format(record)
                self._style._fmt = self.default_format # Reset
                return formatted_message

    # Create handlers
    # Console Handler (with colors)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ServerLogFormatter(use_colors=True))

    # Rotating File Handler (without colors)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024, # 10 MB
        backupCount=5,        # Keep 5 backup files
        encoding='utf-8'
    )
    file_handler.setFormatter(ServerLogFormatter(use_colors=False))

    # Configure Flask's logger
    flask_app.logger.handlers.clear() # Remove default handlers
    flask_app.logger.setLevel(log_level)
    flask_app.logger.addHandler(console_handler)
    flask_app.logger.addHandler(file_handler)

    # --- Flask Hooks for Logging ---
    @flask_app.before_request
    def before_request_logging():
        """Set start time and potentially log request data before handling."""
        g.start_time = datetime.now()
        # Store request data in g for non-GET requests if needed later
        if request.method != 'GET':
            # Limit size of stored request data
             g.request_data = request.get_data(as_text=True)[:500] # Limit to 500 chars

    @flask_app.after_request
    def after_request_logging(response):
        """Log summary information after the request is handled."""
        # Avoid logging for static files or health checks to reduce noise
        if not request.path.startswith(('/static/', '/favicon.ico', '/health')):
            try:
                # Create a log record manually to pass necessary info
                log_level = logging.WARNING if response.status_code >= 400 else logging.INFO
                record_dict = {
                    'name': flask_app.logger.name, 'level': log_level,
                    'pathname': '', 'lineno': 0, 'msg': '', 'args': (), 'exc_info': None,
                    'status_code': response.status_code,
                    # Add response data if JSON and not too large
                    'response_data': response.get_data(as_text=True)[:200] if response.is_json else None,
                     # Add request data captured in before_request
                    'request_data': getattr(g, 'request_data', None) if request.method != 'GET' else None
                }
                flask_app.logger.handle(logging.makeLogRecord(record_dict))
            except Exception as e:
                # Log internal logging errors separately
                flask_app.logger.error(f"Error during after_request logging: {str(e)}", exc_info=True)
        return response

    @flask_app.errorhandler(Exception)
    def handle_exception_logging(error):
        """Log unhandled exceptions."""
        # Log the exception with traceback
        flask_app.logger.error(f"Unhandled Exception: {str(error)}", exc_info=True)
        # Return a generic 500 response (Flask handles this by default, but explicit is fine)
        return "Internal Server Error", 500

    flask_app.logger.info(f"Flask server logging initialized. Level: {logging.getLevelName(log_level)}. Log file: {log_file}")
    return flask_app.logger


# ==============================================================================
# Server Configuration and Execution
# ==============================================================================

# --- Function: load_server_config ---
def load_server_config(config_file="config/config.yaml"):
    """Loads server configuration settings from a YAML file."""
    default_config = {
        'server_type': 'gunicorn' if not sys.platform.startswith('win') else 'flask',
        'host': '0.0.0.0',
        'port': int(os.getenv('PORT', 5000)), # Use PORT env var if set, else 5000
        'flask': {'debug': False, 'threaded': True, 'processes': 1, 'use_reloader': False},
        'gunicorn': {
            'workers': 'auto', 'worker_class': 'sync', 'timeout': 60,
            'keepalive': 5, 'loglevel': 'info', 'accesslog': '-', 'errorlog': '-',
            'worker_connections': 1000, 'threads': 1, 'graceful_timeout': 30
        }
    }
    config_path = os.path.join(os.getcwd(), config_file)
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded_config = yaml.safe_load(f)
            if loaded_config and 'server_config' in loaded_config:
                 # Merge loaded config over defaults (simple merge, not deep)
                 server_config = default_config.copy()
                 user_conf = loaded_config['server_config']
                 server_config.update({k: v for k, v in user_conf.items() if k in server_config})
                 # Merge nested dicts like flask/gunicorn specifically
                 if 'flask' in user_conf: server_config['flask'].update(user_conf['flask'])
                 if 'gunicorn' in user_conf: server_config['gunicorn'].update(user_conf['gunicorn'])
                 # Ensure port from env var overrides file config
                 server_config['port'] = int(os.getenv('PORT', server_config.get('port', 5000)))
                 logger.info(f"Loaded server configuration from: {config_path}")
                 return server_config
            else:
                 logger.warning(f"'{config_file}' is empty or missing 'server_config' section. Using default server settings.")
                 return default_config
    except FileNotFoundError:
        logger.warning(f"Server configuration file not found: {config_path}. Using default settings.")
        return default_config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {config_path}: {e}. Using default settings.")
        return default_config
    except Exception as e:
        logger.exception(f"Unexpected error loading server config {config_path}: {e}. Using default settings.")
        return default_config


# --- Function: run_production_server ---
def run_production_server(flask_app, config):
    """Runs the Flask application using either Gunicorn (preferred) or Flask's built-in server."""
    server_type = config.get('server_type', 'gunicorn')
    host = config.get('host', '0.0.0.0')
    port = config.get('port', 5000) # Port already incorporates ENV override in load_server_config

    # Ensure logging is set up before starting server
    app_logger = setup_logging(flask_app, config.get('gunicorn', {}).get('loglevel', 'INFO'))

    # --- Determine Server Type ---
    use_gunicorn = False
    if server_type.lower() == 'gunicorn' and not sys.platform.startswith('win'):
        try:
            import gunicorn.app.base
            use_gunicorn = True
        except ImportError:
            app_logger.warning(f"{Fore.YELLOW}Gunicorn not found. Falling back to Flask's development server.{Style.RESET_ALL}")
            use_gunicorn = False
    else:
         if server_type.lower() == 'gunicorn':
              app_logger.warning(f"{Fore.YELLOW}Gunicorn is not recommended on Windows. Using Flask's development server.{Style.RESET_ALL}")
         server_type = 'flask' # Force flask if on windows or gunicorn import fails


    # --- Start Flask Server ---
    if not use_gunicorn:
        flask_config = config.get('flask', {})
        app_logger.info(f"{Fore.CYAN}Starting Flask development server on http://{host}:{port}...{Style.RESET_ALL}")
        flask_app.run(
            host=host,
            port=port,
            debug=flask_config.get('debug', False),
            threaded=flask_config.get('threaded', True),
            # processes=flask_config.get('processes', 1), # processes > 1 not standard for dev server
            use_reloader=flask_config.get('use_reloader', False), # Reloader useful for dev
            # extra_files=flask_config.get('extra_files', []) # Watch extra files for reloading
        )

    # --- Start Gunicorn Server ---
    else:
        gunicorn_config = config.get('gunicorn', {})
        # Determine workers intelligently
        workers_config = gunicorn_config.get('workers', 'auto')
        if workers_config == 'auto':
            workers = (multiprocessing.cpu_count() * 2) + 1 # Common Gunicorn recommendation
        else:
            try: workers = int(workers_config)
            except ValueError: workers = 2 # Default if invalid config
            workers = max(1, workers) # Ensure at least 1 worker

        bind_address = gunicorn_config.get('bind', f"{host}:{port}")
        app_logger.info(f"{Fore.GREEN}Starting Gunicorn server on {bind_address} with {workers} workers...{Style.RESET_ALL}")

        # Optional: Set permissions (Use with caution!)
        # try:
        #     subprocess.run(['chmod', '-R', '777', os.getcwd()], check=False) # Check security implications
        #     logger.info("Attempted to set permissions (chmod -R 777). Verify necessity and security.")
        # except Exception as e:
        #     logger.warning(f"Could not set permissions: {e}. Ensure directory permissions are appropriate.")

        # Gunicorn options dictionary
        options = {
            'bind': bind_address,
            'workers': workers,
            'worker_class': gunicorn_config.get('worker_class', 'sync'), # Consider 'gevent' or 'uvicorn.workers.UvicornWorker' for async
            'timeout': int(gunicorn_config.get('timeout', 60)),
            'keepalive': int(gunicorn_config.get('keepalive', 5)),
            'loglevel': gunicorn_config.get('loglevel', 'info').lower(),
            'accesslog': gunicorn_config.get('accesslog', os.path.join('logs', 'gunicorn_access.log')), # Log to file default
            'errorlog': gunicorn_config.get('errorlog', os.path.join('logs', 'gunicorn_error.log')),   # Log to file default
            'daemon': gunicorn_config.get('daemon', False), # Not recommended usually
            'pidfile': gunicorn_config.get('pidfile'), # Useful for process management
            'worker_connections': int(gunicorn_config.get('worker_connections', 1000)),
            'threads': int(gunicorn_config.get('threads', 1)), # Increase for sync workers if I/O bound
            'graceful_timeout': int(gunicorn_config.get('graceful_timeout', 30)),
            # Add other Gunicorn options as needed from config.yaml
        }
        # Filter out None values which Gunicorn might not like
        options = {k: v for k, v in options.items() if v is not None}

        # Gunicorn Application Class Wrapper
        class StandaloneGunicornApplication(gunicorn.app.base.BaseApplication):
            def __init__(self, app, options=None):
                self.options = options or {}
                self.application = app
                super().__init__()

            def load_config(self):
                config = {key: value for key, value in self.options.items()
                          if key in self.cfg.settings and value is not None}
                for key, value in config.items():
                    self.cfg.set(key.lower(), value)

            def load(self):
                return self.application

        # Run Gunicorn
        StandaloneGunicornApplication(flask_app, options).run()


# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == "__main__":

    # 1. Display Banner
    print_banner()

    # 2. Create Directories defined in config/dir_struc.json
    create_directories_from_json(os.path.join(os.getcwd(), "config", "dir_struc.json"))

    # 3. Initial Setup Tasks
    logger.info("Performing initial setup tasks...")
    try:
        getffmpeg.main() # Download/check ffmpeg
        database.init_db() # Initialize main database schema
        database.init_lyrics_db() # Initialize lyrics cache database
        database.init_playlist_db() # Initialize playlist database
        database.init_backup_his_db() # Initialize backup history database
        database.init_issues_db() # Initialize issues database
        util.load_local_songs() # Perform initial scan and load local songs
        # Ensure MUSIC_DIR exists (redundant if create_directories does it, but safe)
        os.makedirs(os.getenv("MUSIC_PATH", os.path.join(os.getcwd(), "music")), exist_ok=True)
        playback.load_local_songs_from_file() # Load into playback blueprint's cache
        #util.download_default_songs() # Optional: download default songs
        logger.info("Initial setup tasks complete.")
    except Exception as e:
         logger.exception(f"Error during initial setup: {e}. Application might not function correctly.")
         # Decide whether to exit or continue with potential issues
         # sys.exit(1) # Uncomment to exit on setup error

    # 4. Initialize App Components (Background Tasks)
    init_app(app)

    # 5. Platform-Specific Initializations
    # Windows: Add to startup
    if starter and sys.platform.startswith('win'):
        try:
            starter.main(
                os.path.join(os.getcwd(), "sangeet.bat"), # Path to your startup script
                os.path.join(os.getcwd(), "assets", "sangeet_logo", "logo.ico") # Path to icon
            )
            logger.info("Windows startup configuration attempted.")
        except Exception as e:
            logger.error(f"Failed to run Windows starter utility: {e}")

    # Cloudflare Tunnel (if configured)
    if os.getenv("CLOUDFLARE_TUNNEL", "False").lower() == 'true':
        logger.info("Cloudflare tunnel enabled in environment variables.")
        try:
            cloudflare_path = download_cloudflare.get_cloudflared(os.path.join(os.getcwd(), "cloudflare_driver")) # Ensure driver exists
            if cloudflare_path:
                 # Run cloudflared in a separate thread or process if needed, non-blocking
                 # cloudflarerun.run_cloudflare expects port and path
                 cloudflare_port = os.getenv('PORT', 5000) # Use the same port Flask/Gunicorn runs on
                 Thread(target=cloudflarerun.run_cloudflare, args=(cloudflare_port, cloudflare_path), daemon=True).start()
                 logger.info(f"Cloudflare tunnel thread started for port {cloudflare_port}.")
            else:
                 logger.error("Failed to get Cloudflared executable path. Tunnel not started.")
        except Exception as e:
            logger.exception(f"Error starting Cloudflare tunnel: {e}")

    # 6. Load Server Configuration
    server_config = load_server_config() # Loads from config/config.yaml

    # 7. Run the Server
    try:
        run_production_server(app, server_config)
    except Exception as e:
        # Log fatal error during server run attempt
        logger.exception(f"Failed to start or run the server: {e}")
        sys.exit("Server could not be started. Check logs for details.")
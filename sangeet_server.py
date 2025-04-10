# --- Imports ---
import json
import os
import sys
import multiprocessing
import time
import subprocess
import yaml
from flask import Flask, request, has_request_context, g
from sangeet_premium.sangeet import playback
from sangeet_premium.utils import getffmpeg,  cloudflarerun, util, download_cloudflare
if sys.platform.startswith('win'):
    from sangeet_premium.utils import starter
from threading import Thread
import logging
from logging.handlers import RotatingFileHandler
from termcolor import colored
from colorama import init, Fore, Style
import pyfiglet
from datetime import datetime, timedelta
from dotenv import load_dotenv
import uuid
from sangeet_premium.database import database
from server_side import config as co

# --- Initializations ---
init(autoreset=True)  # Initialize colorama
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
load_dotenv(dotenv_path=os.path.join(os.getcwd(), "config", ".env"))

app = Flask(__name__)
if os.getenv("FLASK_SECRET_KEY").lower() == "auto":
     app.secret_key = keygen.generate_secure_hex_key(64)
else:
    app.secret_key = os.getenv("FLASK_SECRET_KEY")
app.register_blueprint(playback.bp)
app.register_blueprint(co.bp)
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# --- Utility Functions ---
def print_banner():
    """Display the application banner with version and startup details."""
    sangeet_text = pyfiglet.figlet_format("SANGEET PREMIUM", font='big')
    premium_text = pyfiglet.figlet_format("PLUS", font='big')
    print(f"{Fore.MAGENTA}{sangeet_text}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{premium_text}{Style.RESET_ALL}")
    print(f"\n{Fore.YELLOW}♪ Premium Music Streaming Service ♪{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Version: 3.0.0 | Made with ♥ by Easy Ware{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Starting server at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Git Hub :  {os.getenv("PROJECT_URL")}{Style.RESET_ALL}")
    print("="*80)

def create_directories_from_json(json_file):
    """Create directories and subdirectories based on dir_struc.json."""
    with open(json_file, 'r') as f:
        dir_structure = json.load(f)
    for dir_info in dir_structure['directories']:
        dir_path = os.path.join(os.getcwd(), dir_info['name'])
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory '{dir_path}' created successfully or already exists.")
        if 'subdirs' in dir_info:
            for subdir in dir_info['subdirs']:
                subdir_path = os.path.join(dir_path, subdir)
                os.makedirs(subdir_path, exist_ok=True)
                print(f"Subdirectory '{subdir_path}' created successfully or already exists.")

def start_local_songs_refresh(app):
    """Start a background thread to refresh local songs every 20 seconds."""
    def refresh_loop():
        while True:
            try:
                with app.app_context():
                    util.load_local_songs()
                    playback.load_local_songs_from_file()
            except Exception as e:
                logger.error(f"Error refreshing local songs: {e}")
            time.sleep(20)
    refresh_thread = Thread(target=refresh_loop, daemon=True)
    refresh_thread.start()
    logger.info("Started local songs refresh thread")

def init_app(app):
    """Initialize the app with background tasks."""
    start_local_songs_refresh(app)

# --- Logging Setup ---
def setup_logging(app, log_level=logging.INFO):
    """Setup enhanced Flask server logging with request/response tracking."""
    init()
    os.makedirs('logs', exist_ok=True)
    log_file = f"logs/flask_server_{datetime.now():%Y%m%d_%H%M}.log"
    
    class ServerLogFormatter(logging.Formatter):
        COLORS = {
            'DEBUG': Fore.CYAN,
            'INFO': Fore.GREEN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.RED + Style.BRIGHT
        }
        
        def __init__(self, use_colors=False):
            super().__init__()
            self.use_colors = use_colors
        
        def format_level(self, level_name):
            if self.use_colors:
                color = self.COLORS.get(level_name, '')
                return f"{color}{level_name:8}{Style.RESET_ALL}"
            return f"{level_name:8}"
        
        def format(self, record):
            try:
                if has_request_context():
                    if not hasattr(g, 'request_id'):
                        g.request_id = str(uuid.uuid4())[:6]
                    duration = ''
                    if hasattr(g, 'start_time'):
                        duration = f" ({int((datetime.now() - g.start_time).total_seconds() * 1000)}ms)"
                    msg = (f"{self.format_level(record.levelname)} "
                           f"[{g.request_id}] {request.method} {request.path} "
                           f"→ {getattr(record, 'status_code', '')}{duration}")
                    if request.method != 'GET' and hasattr(record, 'request_data'):
                        msg += f"\n    Request: {record.request_data}"
                    if hasattr(record, 'response_data'):
                        msg += f"\n    Response: {record.response_data}"
                    return msg
                else:
                    return f"{self.format_level(record.levelname)} {record.getMessage()}"
            except Exception as e:
                return f"Logging Error: {str(e)} | Original: {record.getMessage()}"
    
    handlers = []
    console = logging.StreamHandler()
    console.setFormatter(ServerLogFormatter(use_colors=True))
    handlers.append(console)
    file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding='utf-8')
    file_handler.setFormatter(ServerLogFormatter(use_colors=False))
    handlers.append(file_handler)
    
    app.logger.handlers.clear()
    app.logger.setLevel(log_level)
    for handler in handlers:
        app.logger.addHandler(handler)
    
    @app.before_request
    def track_request():
        g.start_time = datetime.now()
        if request.method != 'GET':
            g.request_data = request.get_data(as_text=True)
    
    @app.after_request
    def log_request(response):
        try:
            if not request.path.startswith(('/static/', '/favicon.ico', '/health')):
                record = logging.LogRecord(
                    name=app.logger.name,
                    level=logging.INFO,
                    pathname='',
                    lineno=0,
                    msg='',
                    args=(),
                    exc_info=None
                )
                record.status_code = response.status_code
                if response.is_json:
                    record.response_data = response.get_data(as_text=True)[:200]
                if request.method != 'GET':
                    record.request_data = getattr(g, 'request_data', '')[:200]
                app.logger.handle(record)
        except Exception as e:
            app.logger.error(f"Logging error: {str(e)}")
        return response
    
    @app.errorhandler(Exception)
    def log_error(error):
        app.logger.error(f"Server Error: {str(error)}", exc_info=True)
        return "Internal Server Error", 500
    
    app.logger.info("Flask server logging initialized")
    return app.logger

# --- Server Configuration ---
def load_server_config(config_file):
    """Load server configuration from config.yaml."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            return config.get('server_config', {})
    except FileNotFoundError:
        logger.error(f"{config_file} not found")
        sys.exit(1)
    except yaml.YAMLError:
        logger.error(f"Invalid YAML in {config_file}")
        sys.exit(1)
def run_production_server(app, config):
    """Run server with proper logging and configuration from config.yaml."""
    server_type = config.get('server_type', 'gunicorn')
    host = config.get('host', '0.0.0.0')
    port = os.getenv('PORT')
    setup_logging(app)
    
    if server_type == 'flask' or sys.platform.startswith('win'):
        if sys.platform.startswith('win'):
            app.logger.warning(colored("Using Flask's server (not for production on Windows)", 'yellow'))
        flask_config = config.get('flask', {})
        app.run(
            host=host,
            port=port,
            debug=flask_config.get('debug', False),
            threaded=flask_config.get('threaded', True),
            processes=flask_config.get('processes', 1),
            use_reloader=flask_config.get('use_reloader', False),
            extra_files=flask_config.get('extra_files', [])
        )
    else:
        try:
            import gunicorn.app.base
            gunicorn_config = config.get('gunicorn', {})
            workers = gunicorn_config.get('workers', 'auto')
            if workers == 'auto':
                workers = max(2, multiprocessing.cpu_count())
            else:
                workers = int(workers)
            
            bind = gunicorn_config.get('bind')
            if not bind:
                bind = f"{host}:{port}"
            
            app.logger.info(colored(f"Starting Gunicorn on {bind} with {workers} workers", 'green'))
            
            try:
                subprocess.Popen(f"chmod -R 777 '{os.getcwd()}'", shell=True)
            except:
                print("give permissions by adding chmod -R 777 /path/to/directory")
            
            options = {
                'bind': bind,
                'workers': workers,
                'worker_class': gunicorn_config.get('worker_class', 'sync'),
                'timeout': gunicorn_config.get('timeout', 30),
                'keepalive': gunicorn_config.get('keepalive', 5),
                'loglevel': gunicorn_config.get('loglevel', 'info'),
                'accesslog': gunicorn_config.get('accesslog', f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_access.log"),
                'errorlog': gunicorn_config.get('errorlog', f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_error.log"),
                'daemon': gunicorn_config.get('daemon', False),
                'pidfile': gunicorn_config.get('pidfile'),
                'worker_connections': gunicorn_config.get('worker_connections', 1000),
                'max_requests': gunicorn_config.get('max_requests', 0),
                'max_requests_jitter': gunicorn_config.get('max_requests_jitter', 0),
                'threads': gunicorn_config.get('threads', 1),
                'worker_tmp_dir': gunicorn_config.get('worker_tmp_dir'),
                'graceful_timeout': gunicorn_config.get('graceful_timeout', 30),
                'max_memory_restart': gunicorn_config.get('max_memory_restart')
            }
            options = {k: v for k, v in options.items() if v is not None}
            
            class GunicornServer(gunicorn.app.base.BaseApplication):
                def __init__(self, app, options=None):
                    self.options = options or {}
                    self.application = app
                    super().__init__()
                
                def load_config(self):
                    for key, value in self.options.items():
                        self.cfg.set(key, value)
                
                def load(self):
                    return self.application
            
            GunicornServer(app, options).run()
        
        except ImportError:
            app.logger.warning(colored("Gunicorn not found, using Flask server", 'yellow'))
            flask_config = config.get('flask', {})
            app.run(
                host=host,
                port=port,
                debug=flask_config.get('debug', False),
                threaded=flask_config.get('threaded', True),
                processes=flask_config.get('processes', 1),
                use_reloader=flask_config.get('use_reloader', False),
                extra_files=flask_config.get('extra_files', [])
            )

# --- Main Execution ---
if __name__ == "__main__":
    # Display banner
    try:
        print_banner()
    except Exception as e:
        print(f"Banner display error: {e}")
        print("Continuing with server startup...")
    
    # Create directories from dir_struc.json
    create_directories_from_json(os.path.join(os.getcwd() , "config" ,"dir_struc.json"))
    
    # Initial setup
    getffmpeg.main()
    database.init_db()
    database.init_lyrics_db()
    util.load_local_songs()
    os.makedirs(os.getenv("MUSIC_PATH"), exist_ok=True)
    playback.load_local_songs_from_file()
    util.download_default_songs()
    database.init_playlist_db()
    database.init_backup_his_db()
    database.init_issues_db()
    init_app(app)
    
    if sys.platform.startswith('win'):
        starter.main(os.path.join(os.getcwd(), "sangeet.bat"), os.path.join(os.getcwd(), "assets", "sangeet_logo", "logo.ico"))
    
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
    
    # Load server configuration and run
    server_config = load_server_config(os.path.join(os.getcwd() , "config" , "config.yaml"))
    run_production_server(app, server_config)
    
 

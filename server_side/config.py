from flask import Flask, render_template, request, redirect, url_for, session, flash , Blueprint
import yaml
import os
from dotenv import load_dotenv


load_dotenv(dotenv_path = os.path.join(os.getcwd() , "config" , ".env"))

bp = Blueprint('config', __name__)  # Create a blueprint

MASTER_PASSWORD =  os.getenv("ADMIN_PASS") # Replace with a secure password in production
CONFIG_FILE = os.path.join(os.getcwd() , "config" , "config.yaml")
AVAILABLE_CATEGORIES = ['server_config']  # Add more categories in the future (e.g., 'settings_2')

def load_config():
    """Load the configuration from config.yaml."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise Exception("config.yaml not found. Please create it with the required structure.")
    except yaml.YAMLError:
        raise Exception("Error parsing config.yaml. Please check its format.")

def save_config(config):
    """Save the configuration to config.yaml."""
    with open(CONFIG_FILE, 'w') as f:
        yaml.safe_dump(config, f)

def admin_required(f):
    """Decorator to ensure the user is logged in."""
    def wrap(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('config.config_login'))
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

@bp.route('/config', methods=['GET', 'POST'])
def config_login():
    """Handle login page and master password verification."""
    if request.method == 'POST':
        password = request.form.get('password')
        if password == MASTER_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('config.settings'))
        else:
            flash('Incorrect password. Please try again.', 'danger')
    return render_template('admin_pass_auth.html')

@bp.route('/settings')
@admin_required
def settings():
    """Display the settings page with current config values for the selected category."""
    category = request.args.get('category', 'server_config')
    if category not in AVAILABLE_CATEGORIES:
        category = 'server_config'
    config = load_config()
    selected_config = config.get(category, {})
    return render_template('settings.html', category=category, config=selected_config, available_categories=AVAILABLE_CATEGORIES)

@bp.route('/save_settings', methods=['POST'])
@admin_required
def save_settings():
    """Update only the submitted settings in config.yaml while preserving existing structure."""
    category = request.form.get('category')
    if category not in AVAILABLE_CATEGORIES:
        flash('Invalid category.', 'danger')
        return redirect(url_for('config.settings'))
    
    config = load_config()
    
    if category == 'server_config':
        # Update only the fields that were submitted in the form
        if 'server_type' in request.form:
            config['server_config']['server_type'] = request.form.get('server_type')
        if 'host' in request.form:
            config['server_config']['host'] = request.form.get('host')
        if 'port' in request.form:
            config['server_config']['port'] = int(request.form.get('port')) if request.form.get('port') else None
        if 'sangeet_backend' in request.form:
            config['server_config']['sangeet_backned'] = request.form.get('sangeet_backned')
        
        # Flask settings
        if 'flask_debug' in request.form or request.form.get('flask_debug') == 'off':
            config['server_config']['flask']['debug'] = 'flask_debug' in request.form
        if 'flask_threaded' in request.form or request.form.get('flask_threaded') == 'off':
            config['server_config']['flask']['threaded'] = 'flask_threaded' in request.form
        if 'flask_processes' in request.form:
            config['server_config']['flask']['processes'] = int(request.form.get('flask_processes')) if request.form.get('flask_processes') else config['server_config']['flask']['processes']
        if 'flask_use_reloader' in request.form or request.form.get('flask_use_reloader') == 'off':
            config['server_config']['flask']['use_reloader'] = 'flask_use_reloader' in request.form
        if 'flask_extra_files' in request.form:
            config['server_config']['flask']['extra_files'] = request.form.get('flask_extra_files', '').split(', ') if request.form.get('flask_extra_files') else []

        # Gunicorn settings
        if 'gunicorn_workers' in request.form:
            config['server_config']['gunicorn']['workers'] = request.form.get('gunicorn_workers')
        if 'gunicorn_worker_class' in request.form:
            config['server_config']['gunicorn']['worker_class'] = request.form.get('gunicorn_worker_class')
        if 'gunicorn_timeout' in request.form:
            config['server_config']['gunicorn']['timeout'] = int(request.form.get('gunicorn_timeout')) if request.form.get('gunicorn_timeout') else config['server_config']['gunicorn']['timeout']
        if 'gunicorn_keepalive' in request.form:
            config['server_config']['gunicorn']['keepalive'] = int(request.form.get('gunicorn_keepalive')) if request.form.get('gunicorn_keepalive') else config['server_config']['gunicorn']['keepalive']
        if 'gunicorn_loglevel' in request.form:
            config['server_config']['gunicorn']['loglevel'] = request.form.get('gunicorn_loglevel')
        if 'gunicorn_accesslog' in request.form:
            config['server_config']['gunicorn']['accesslog'] = request.form.get('gunicorn_accesslog') or None
        if 'gunicorn_errorlog' in request.form:
            config['server_config']['gunicorn']['errorlog'] = request.form.get('gunicorn_errorlog') or None
        if 'gunicorn_bind' in request.form:
            config['server_config']['gunicorn']['bind'] = request.form.get('gunicorn_bind') or None
        if 'gunicorn_preload' in request.form or request.form.get('gunicorn_preload') == 'off':
            config['server_config']['gunicorn']['preload'] = 'gunicorn_preload' in request.form
        if 'gunicorn_daemon' in request.form or request.form.get('gunicorn_daemon') == 'off':
            config['server_config']['gunicorn']['daemon'] = 'gunicorn_daemon' in request.form
        if 'gunicorn_pidfile' in request.form:
            config['server_config']['gunicorn']['pidfile'] = request.form.get('gunicorn_pidfile') or None
        if 'gunicorn_worker_connections' in request.form:
            config['server_config']['gunicorn']['worker_connections'] = int(request.form.get('gunicorn_worker_connections')) if request.form.get('gunicorn_worker_connections') else config['server_config']['gunicorn']['worker_connections']
        if 'gunicorn_max_requests' in request.form:
            config['server_config']['gunicorn']['max_requests'] = int(request.form.get('gunicorn_max_requests')) if request.form.get('gunicorn_max_requests') else config['server_config']['gunicorn']['max_requests']
        if 'gunicorn_max_requests_jitter' in request.form:
            config['server_config']['gunicorn']['max_requests_jitter'] = int(request.form.get('gunicorn_max_requests_jitter')) if request.form.get('gunicorn_max_requests_jitter') else config['server_config']['gunicorn']['max_requests_jitter']
        if 'gunicorn_threads' in request.form:
            config['server_config']['gunicorn']['threads'] = int(request.form.get('gunicorn_threads')) if request.form.get('gunicorn_threads') else config['server_config']['gunicorn']['threads']
        if 'gunicorn_worker_tmp_dir' in request.form:
            config['server_config']['gunicorn']['worker_tmp_dir'] = request.form.get('gunicorn_worker_tmp_dir') or None
        if 'gunicorn_graceful_timeout' in request.form:
            config['server_config']['gunicorn']['graceful_timeout'] = int(request.form.get('gunicorn_graceful_timeout')) if request.form.get('gunicorn_graceful_timeout') else config['server_config']['gunicorn']['graceful_timeout']
        if 'gunicorn_max_memory_restart' in request.form:
            config['server_config']['gunicorn']['max_memory_restart'] = request.form.get('gunicorn_max_memory_restart') or None

    save_config(config)
    flash('Settings saved successfully. Restart the server to apply changes.', 'success')
    return redirect(url_for('config.settings', category=category))

@bp.route('/config/logout')
@admin_required
def config_logout():
    """Log out the user and redirect to login page."""
    session.pop('logged_in', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('config.config_login'))


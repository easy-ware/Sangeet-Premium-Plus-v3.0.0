import os
import sys
import platform
import subprocess
import venv
from pathlib import Path
import hashlib
import logging
import getpass
from colorama import init, Fore, Style, AnsiToWin32
from datetime import datetime
import argparse
import signal

class SmartVENVCreator:
    def __init__(self, venv_name=None, requirements_path='requirements.txt', 
                 log_dir=None, log_level=logging.INFO):
        """Initialize with platform-specific detection and setup"""
        # Detect platform details
        self.os_name = platform.system().lower()
        self.os_version = platform.platform()
        self.is_windows = self.os_name == 'windows'
        self.is_cygwin = 'cygwin' in platform.system().lower()
        self.is_msys = 'msys' in platform.system().lower()
        self.is_mingw = 'mingw' in platform.system().lower()
        self.is_wsl = 'microsoft' in platform.uname().release.lower()
        
        # Initialize color support based on platform
        self._setup_color_support()
        
        # Rest of initialization
        self.python_executable = self._get_python_executable()
        self.username = getpass.getuser()
        self.log_dir = self._setup_log_directory(log_dir)
        self.log_file = self._create_session_log_file()
        
        # Setup logging with color support
        logging.basicConfig(
            level=log_level,
            format=(
                f'{Fore.CYAN}%(asctime)s{Style.RESET_ALL} - '
                f'{Fore.GREEN}%(levelname)s{Style.RESET_ALL} - %(message)s'
            ),
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.log_file)
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Set venv name and paths
        self.venv_name = venv_name or self._generate_default_venv_name()
        self.requirements_path = Path(requirements_path).resolve()
        self.venv_path = Path(f".{self.venv_name}")
        
        # Set activation scripts based on platform
        self._set_activation_scripts()
        
        self.current_process = None

    def _setup_color_support(self):
        """Setup color support for different platforms"""
        # Initialize colorama with appropriate settings
        init(wrap=True, strip=False, convert=None)
        
        if self.is_windows:
            # Enable virtual terminal processing on Windows
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                # Enable ENABLE_VIRTUAL_TERMINAL_PROCESSING
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except:
                pass
        
        # Set environment variables for color support
        os.environ['FORCE_COLOR'] = '1'
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        # Set term if not set
        if 'TERM' not in os.environ:
            os.environ['TERM'] = 'xterm-256color'

    def _set_activation_scripts(self):
        """Set activation scripts based on platform"""
        if self.is_windows:
            scripts_dir = 'Scripts'
            activate_script = 'activate.bat'
        else:
            scripts_dir = 'bin'
            activate_script = 'activate'
            
        if self.is_cygwin or self.is_msys or self.is_mingw:
            scripts_dir = 'bin'
            activate_script = 'activate'
        
        self.activation_scripts = {
            'windows': str(self.venv_path / 'Scripts' / 'activate.bat'),
            'cygwin': str(self.venv_path / 'bin' / 'activate'),
            'msys': str(self.venv_path / 'bin' / 'activate'),
            'mingw': str(self.venv_path / 'bin' / 'activate'),
            'linux': str(self.venv_path / 'bin' / 'activate'),
            'darwin': str(self.venv_path / 'bin' / 'activate')
        }

    def _get_python_executable(self):
        """Get the appropriate Python executable for the platform"""
        executable_map = {
            'windows': sys.executable,
            'linux': '/usr/bin/python3',
            'darwin': '/usr/local/bin/python3',
            'cygwin': '/usr/bin/python3',
            'msys': '/usr/bin/python3',
            'mingw': '/usr/bin/python3'
        }
        return executable_map.get(self.os_name, sys.executable)

    def _setup_log_directory(self, log_dir=None):
        """Setup log directory with enhanced creation capabilities"""
        try:
            if log_dir:
                log_path = Path(log_dir).expanduser().resolve()
            else:
                # Create venv-specific logs directory
                log_path = Path.cwd() / 'logs' / 'venve-logs'
            
            log_path.mkdir(parents=True, exist_ok=True)
            return log_path
        except (OSError, PermissionError) as e:
            print(f"{Fore.RED}Error creating log directory: {e}{Style.RESET_ALL}")
            return Path.cwd()

    def _create_session_log_file(self):
        """Create a session-specific log file"""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
        log_filename = f"venv_creation_{timestamp}.log"
        return str(self.log_dir / log_filename)

    def _generate_default_venv_name(self):
        """Generate a unique venv name based on project and system details"""
        project_name = Path.cwd().name
        system_details = f"{self.os_name}_{platform.machine()}"
        
        venv_name = f"{self.username}-{project_name}-{system_details}"
        venv_name = ''.join(
            c if c.isalnum() or c in ['-', '_'] else '_' 
            for c in venv_name
        )
        
        return venv_name.lower()

    def _hash_requirements(self):
        """Generate hash of requirements file"""
        if not self.requirements_path.exists():
            return None
        
        with open(self.requirements_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signals"""
        if self.current_process:
            self.logger.info("\nReceived interrupt signal. Terminating...")
            try:
                if self.is_windows:
                    self.current_process.send_signal(signal.CTRL_C_EVENT)
                else:
                    self.current_process.send_signal(signal.SIGINT)
            except Exception as e:
                self.logger.error(f"Error terminating process: {e}")
                self.current_process.terminate()

    def create_venv(self):
        """Create virtual environment if not exists"""
        if self.venv_path.exists():
            self.logger.info(f"{Fore.YELLOW}Virtual environment already exists at {self.venv_path}{Style.RESET_ALL}")
            return False
        
        try:
            venv.create(self.venv_path, with_pip=True)
            self.logger.info(f"{Fore.GREEN}Virtual environment created successfully at {self.venv_path}{Style.RESET_ALL}")
            return True
        except Exception as e:
            self.logger.error(f"{Fore.RED}Failed to create virtual environment: {e}{Style.RESET_ALL}")
            return False

    def install_requirements(self):
        """Install requirements with change detection"""
        if not self.requirements_path.exists():
            self.logger.warning(f"{Fore.YELLOW}No requirements.txt found at {self.requirements_path}{Style.RESET_ALL}")
            return False

        hash_file = self.venv_path / '.requirements_hash'
        current_hash = self._hash_requirements()

        if hash_file.exists():
            with open(hash_file, 'r') as f:
                old_hash = f.read().strip()
            
            if old_hash == current_hash:
                self.logger.info(f"{Fore.GREEN}Requirements unchanged. Skipping reinstallation.{Style.RESET_ALL}")
                return True

        pip_path = self.venv_path / ('Scripts' if self.is_windows else 'bin') / ('pip.exe' if self.is_windows else 'pip')
        try:
            subprocess.check_call([
                str(pip_path), 'install', '-r', str(self.requirements_path),
                '--upgrade', '--no-cache-dir'
            ])
            
            with open(hash_file, 'w') as f:
                f.write(current_hash)
            
            self.logger.info(f"{Fore.GREEN}Requirements installed successfully.{Style.RESET_ALL}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"{Fore.RED}Failed to install requirements: {e}{Style.RESET_ALL}")
            return False

    def get_activation_command(self):
        """Get platform-specific activation command"""
        if self.is_cygwin or self.is_msys or self.is_mingw:
            platform_key = 'cygwin' if self.is_cygwin else ('msys' if self.is_msys else 'mingw')
        else:
            platform_key = self.os_name

        activation_command = self.activation_scripts.get(platform_key)
        if not activation_command:
            self.logger.warning(f"{Fore.YELLOW}Unsupported platform: {platform_key}{Style.RESET_ALL}")
            return None
        
        return f"source {activation_command}" if not self.is_windows else activation_command

    def run_script(self, script_path, script_args=None):
        """Run a Python script with universal color support"""
        if self.is_windows and not (self.is_cygwin or self.is_msys or self.is_mingw):
            python_exe = self.venv_path / 'Scripts' / 'python.exe'
        else:
            python_exe = self.venv_path / 'bin' / 'python'

        if not python_exe.exists():
            self.logger.error(f"Virtual environment Python not found at {python_exe}")
            return False

        try:
            cmd = [str(python_exe), str(script_path)]
            if script_args:
                cmd.extend(script_args)

            env = os.environ.copy()
            env.update({
                'PYTHONIOENCODING': 'utf-8',
                'PYTHONUNBUFFERED': '1',
                'FORCE_COLOR': '1',
                'COLORTERM': 'truecolor',
                'TERM': os.environ.get('TERM', 'xterm-256color')
            })

            if self.is_windows and not (self.is_cygwin or self.is_msys or self.is_mingw):
                env['VIRTUAL_ENV'] = str(self.venv_path)
                env['PATH'] = f"{self.venv_path / 'Scripts'}{os.pathsep}{env['PATH']}"
            else:
                env['VIRTUAL_ENV'] = str(self.venv_path)
                env['PATH'] = f"{self.venv_path / 'bin'}{os.pathsep}{env['PATH']}"

            signal.signal(signal.SIGINT, self._handle_interrupt)
            print(f"\nRunning script: {script_path}\n", flush=True)

            if self.is_windows and not (self.is_cygwin or self.is_msys or self.is_mingw):
                stream = AnsiToWin32(sys.stdout).stream
                self.current_process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=stream,
                    stderr=stream,
                    bufsize=0,
                    universal_newlines=True
                )
            else:
                self.current_process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=None,
                    stderr=None,
                    bufsize=0
                )

            return_code = self.current_process.wait()
            print()

            if return_code == 0:
                print(f"Script completed successfully", flush=True)
                return True
            else:
                print(f"Script failed with return code {return_code}", flush=True)
                return False

        except Exception as e:
            print(f"Unexpected error running script: {e}", flush=True)
            return False
        finally:
            signal.signal(signal.SIGINT, signal.default_int_handler)
            self.current_process = None

    def run(self):
        """Execute the full venv creation and setup process"""
        self.logger.info(f"{Fore.CYAN}Starting Smart VENV Creator{Style.RESET_ALL}")
        self.logger.info(f"Platform: {Fore.GREEN}{self.os_name}{Style.RESET_ALL}")
        self.logger.info(f"Platform Details: {Fore.GREEN}{self.os_version}{Style.RESET_ALL}")
        self.logger.info(f"Username: {Fore.GREEN}{self.username}{Style.RESET_ALL}")
        self.logger.info(f"Virtual Environment Name: {Fore.MAGENTA}{self.venv_name}{Style.RESET_ALL}")
        self.logger.info(f"Log File: {Fore.BLUE}{self.log_file}{Style.RESET_ALL}")
        
        try:
            if self.create_venv():
                self.install_requirements()
                
                activation_cmd = self.get_activation_command()
                if activation_cmd:
                    self.logger.info(f"{Fore.MAGENTA}Activation Command: {activation_cmd}{Style.RESET_ALL}")
        except Exception as e:
            self.logger.error(f"{Fore.RED}An error occurred during VENV creation: {e}{Style.RESET_ALL}")
        
        self.logger.info(f"{Fore.CYAN}Process Completed{Style.RESET_ALL}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Smart Virtual Environment Creator')
    parser.add_argument('-n', '--name', 
                       help='Custom virtual environment name')
    parser.add_argument('-r', '--requirements', 
                       default='requirements.txt',
                       help='Path to requirements.txt file')
    parser.add_argument('-l', '--log-dir', 
                       help='Directory for log files')
    parser.add_argument('-d', '--debug', 
                       action='store_true',
                       help='Enable debug logging')
    parser.add_argument('-s', '--script',
                       help='Python script to run after environment creation')
    parser.add_argument('script_args',
                       nargs=argparse.REMAINDER,
                       help='Arguments to pass to the script')
    return parser.parse_args()

def create_env(venv_name=None, requirements_path='requirements.txt', log_dir=None, script_path=None, script_args=None):
    """Programmatic interface for creating virtual environment"""
    creator = SmartVENVCreator(
        venv_name=venv_name, 
        requirements_path=requirements_path, 
        log_dir=log_dir,
        log_level=logging.DEBUG
    )
    creator.run()
    
    if script_path:
        creator.run_script(script_path, script_args)

    return creator

def main():
    """Main entry point for command line usage"""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Determine log level based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    
    # Create virtual environment
    creator = SmartVENVCreator(
        venv_name=args.name, 
        requirements_path=args.requirements, 
        log_dir=args.log_dir,
        log_level=log_level
    )
    creator.run()

    # Run script if specified
    if args.script:
        creator.run_script(args.script, args.script_args)

if __name__ == "__main__":
    main()
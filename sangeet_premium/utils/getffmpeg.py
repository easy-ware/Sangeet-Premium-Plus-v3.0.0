import os
import sys
import platform
import requests
import logging
import subprocess
from pathlib import Path
from colorama import init, Fore, Style
from tqdm import tqdm
import time

# Initialize colorama
init(autoreset=True)

# FFmpeg Windows components URLs
FFMPEG_URLS = {
    'ffmpeg.exe': 'https://github.com/easy-ware/Sangeet-Premium/releases/download/components/ffmpeg.exe',
    'ffprobe.exe': 'https://github.com/easy-ware/Sangeet-Premium/releases/download/components/ffprobe.exe',
    'ffplay.exe': 'https://github.com/easy-ware/Sangeet-Premium/releases/download/components/ffplay.exe'
}

# Package manager commands for Unix-like systems
PACKAGE_MANAGERS = {
    'apt': {
        'check': 'dpkg -l ffmpeg',
        'install': 'sudo apt update && sudo apt install -y ffmpeg'
    },
    'dnf': {
        'check': 'dnf list installed ffmpeg',
        'install': 'sudo dnf install -y ffmpeg'
    },
    'yum': {
        'check': 'yum list installed ffmpeg',
        'install': 'sudo yum install -y ffmpeg'
    },
    'pacman': {
        'check': 'pacman -Qi ffmpeg',
        'install': 'sudo pacman -S --noconfirm ffmpeg'
    },
    'brew': {
        'check': 'brew list ffmpeg',
        'install': 'brew install ffmpeg'
    }
}

# Directory setup for Windows
FFMPEG_DIR = os.path.join(os.getcwd(), 'ffmpeg', 'bin')

def setup_logger():
    logger = logging.getLogger('FFmpegDownloader')
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    class ColoredFormatter(logging.Formatter):
        format_str = '%(asctime)s - %(levelname)s - %(message)s'
        
        FORMATS = {
            logging.INFO: Fore.GREEN + format_str + Style.RESET_ALL,
            logging.WARNING: Fore.YELLOW + format_str + Style.RESET_ALL,
            logging.ERROR: Fore.RED + format_str + Style.RESET_ALL
        }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno, self.format_str)
            formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
            return formatter.format(record)

    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)
    return logger

def format_size(size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0

def detect_package_manager():
    for pm, commands in PACKAGE_MANAGERS.items():
        try:
            result = subprocess.run(['which', pm], capture_output=True, text=True)
            if result.returncode == 0:
                return pm
        except:
            continue
    return None

def install_unix_ffmpeg(package_manager, logger):
    try:
        # Check if FFmpeg is already installed
        check_cmd = PACKAGE_MANAGERS[package_manager]['check']
        check_result = subprocess.run(check_cmd, shell=True, capture_output=True)
        
        if check_result.returncode == 0:
            logger.info(f"{Fore.GREEN}FFmpeg is already installed via {package_manager}")
            return True
        
        # Install FFmpeg
        logger.info(f"Installing FFmpeg using {package_manager}...")
        install_cmd = PACKAGE_MANAGERS[package_manager]['install']
        process = subprocess.run(install_cmd, shell=True, check=True)
        
        if process.returncode == 0:
            logger.info(f"{Fore.GREEN}Successfully installed FFmpeg using {package_manager}")
            return True
        return False
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing FFmpeg: {str(e)}")
        return False

def download_windows_component(filename, url, logger):
    save_path = os.path.join(FFMPEG_DIR, filename)
    
    # Check if file already exists
    if os.path.exists(save_path):
        logger.info(f"{Fore.CYAN}{filename}{Style.RESET_ALL} already exists")
        return True
    
    try:
        logger.info(f"Downloading {Fore.CYAN}{filename}{Style.RESET_ALL}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        if total_size:
            logger.info(f"Size: {Fore.YELLOW}{format_size(total_size)}{Style.RESET_ALL}")
        
        with open(save_path, 'wb') as file, \
             tqdm(
                desc=f"{Fore.GREEN}Downloading {filename}{Style.RESET_ALL}",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
             ) as pbar:
            
            downloaded = 0
            start_time = time.time()
            
            for chunk in response.iter_content(chunk_size=8192):
                size = file.write(chunk)
                pbar.update(size)
                downloaded += size
            
            duration = time.time() - start_time
            speed = downloaded / (1024 * 1024 * duration)  # MB/s
            
        logger.info(f"Successfully downloaded {filename} - Speed: {Fore.GREEN}{speed:.2f} MB/s{Style.RESET_ALL}")
        return True
    
    except Exception as e:
        logger.error(f"Error downloading {filename}: {str(e)}")
        return False

def main():
    logger = setup_logger()
    system = platform.system()
    
    logger.info(f"Detected system: {Fore.CYAN}{system}{Style.RESET_ALL}")
    
    if system == "Windows":
        # Windows installation process
        Path(FFMPEG_DIR).mkdir(parents=True, exist_ok=True)
        logger.info(f"Installing FFmpeg components in: {Fore.CYAN}{FFMPEG_DIR}{Style.RESET_ALL}")
        
        success_count = 0
        total_components = len(FFMPEG_URLS)
        
        for filename, url in FFMPEG_URLS.items():
            if download_windows_component(filename, url, logger):
                success_count += 1
        
        if success_count == total_components:
            logger.info(f"{Fore.GREEN}All FFmpeg components are ready!{Style.RESET_ALL}")
            logger.info(f"Add {Fore.CYAN}{FFMPEG_DIR}{Style.RESET_ALL} to your system PATH")
        else:
            logger.warning(f"Downloaded {success_count}/{total_components} components. Some components may be missing.")
    
    else:
        # Unix-like systems (Linux/macOS) installation process
        package_manager = detect_package_manager()
        
        if not package_manager:
            logger.error("No supported package manager found. Please install FFmpeg manually.")
            logger.info("Supported package managers: apt, dnf, yum, pacman, brew")
            return
        
        if install_unix_ffmpeg(package_manager, logger):
            logger.info(f"{Fore.GREEN}FFmpeg installation complete!{Style.RESET_ALL}")
            # Verify installation
            try:
                result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"FFmpeg version: {result.stdout.split('version')[1].split()[0]}")
            except:
                pass
        else:
            logger.error("FFmpeg installation failed!")

if __name__ == "__main__":
    main()

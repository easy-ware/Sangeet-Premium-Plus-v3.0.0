import os
import platform
import requests
import stat
from pathlib import Path

def setup_ytdlp():
    """
    Set up yt-dlp executable and return its path.
    Creates system-specific subdirectories in res folder.
    Returns:
        str: Path to the yt-dlp executable
    """
    try:
        # Detect system
        system = platform.system().lower()
        machine = platform.machine().lower()

        print(f"Detected system: {system}")
        print(f"Detected machine architecture: {machine}")

        # More precise architecture mapping
        download_patterns = {
            # ARM architectures
            'aarch64': 'yt-dlp_linux_aarch64',  # For RPi 5 and other ARM64
            'armv7l': 'yt-dlp_linux_armv7l',    # For older RPi and ARMv7
            'armv6l': 'yt-dlp_linux_armv7l',    # For very old RPi
            # X86 architectures
            'x86_64': 'yt-dlp_linux',           # Standard 64-bit Linux
            'amd64': 'yt-dlp_linux',            # Alternative 64-bit name
            'i386': 'yt-dlp_linux_x86',         # 32-bit x86
            'i686': 'yt-dlp_linux_x86'          # Alternative 32-bit name
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

        # Set executable name
        executable = "yt-dlp.exe" if system == "windows" else "yt-dlp"
        executable_path = res_dir / executable

        # Get latest release info
        api_url = "https://api.github.com/repos/yt-dlp/yt-dlp/releases/latest"
        response = requests.get(api_url)
        release_data = response.json()

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

        # Set executable permissions for non-Windows systems
        if system != "windows":
            executable_path.chmod(executable_path.stat().st_mode | stat.S_IEXEC)

        print(f"Successfully installed to: {executable_path}")

        return str(executable_path)

    except Exception as e:
        print(f"Error setting up yt-dlp: {e}")
        return None

# Example usage:
if name == "main":
    YTDLP_PATH = setup_ytdlp()
    if YTDLP_PATH:
        print(f"\nYT-DLP ready to use at: {YTDLP_PATH}")

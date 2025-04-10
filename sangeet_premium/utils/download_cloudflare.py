import os
import platform
import requests
import tempfile

def get_cloudflared(base_dir=None):
    """
    Attempts to download the latest Cloudflared binary. If the existing binary's version matches
    the latest version on GitHub, skips the download. If any issue occurs during the download,
    falls back to an existing binary if present, otherwise returns None.

    Args:
        base_dir (str, optional): Directory to store cloudflared. Defaults to ./drivers if None.

    Returns:
        str or None: Full path to the cloudflared binary if successful or existing, else None.
    """
    try:
        # Set default directory if none provided
        if base_dir is None:
            base_dir = os.path.join(os.getcwd(), "drivers")
        
        # Detect system info
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        # Map architectures
        arch_map = {
            'x86_64': 'amd64',
            'aarch64': 'arm64',
            'armv7l': 'arm',
            'i386': '386'
        }
        arch = arch_map.get(machine, machine)
        
        # Set system name
        if system == 'darwin':
            system = 'darwin'
        elif system == 'windows':
            system = 'windows'
        else:
            system = 'linux'
            
        # Setup paths
        platform_dir = os.path.join(base_dir, f"{system}-{arch}")
        binary_name = 'cloudflared.exe' if system == 'windows' else 'cloudflared'
        binary_path = os.path.join(platform_dir, binary_name)
        version_file = os.path.join(platform_dir, 'version.txt')
        
        # Create directory if it doesn’t exist
        os.makedirs(platform_dir, exist_ok=True)
        
        # Get latest release info from GitHub
        headers = {'Accept': 'application/vnd.github.v3+json'}
        response = requests.get(
            "https://api.github.com/repos/cloudflare/cloudflared/releases/latest",
            headers=headers
        )
        response.raise_for_status()
        release_data = response.json()
        latest_version = release_data['tag_name'].replace('v', '')
        
        # Check if existing binary is up to date
        if os.path.isfile(version_file) and os.path.isfile(binary_path):
            try:
                with open(version_file, 'r') as f:
                    current_version = f.read().strip()
                if current_version == latest_version:
                    print(f"Cloudflared is up to date: {latest_version}")
                    return binary_path
            except Exception as e:
                print(f"Error reading version file: {e}")
                # Proceed to download
        
        # Find the correct asset for the system and architecture
        asset_name = f"cloudflared-{system}-{arch}"
        if system == 'windows':
            asset_name += '.exe'
        
        download_url = next(
            (asset['browser_download_url'] for asset in release_data['assets'] 
             if asset['name'].lower() == asset_name.lower()),
            None
        )
        if not download_url:
            raise Exception(f"No binary found for {system}-{arch}")
        
        # Download the binary to a temporary file
        response = requests.get(download_url)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, dir=platform_dir, mode='wb') as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
        
        # Set executable permissions if not on Windows
        if system != 'windows':
            os.chmod(tmp_path, 0o755)
        
        # Atomically move the temporary file to the final binary path
        os.replace(tmp_path, binary_path)
        
        # Save the version to the version file
        with open(version_file, 'w') as f:
            f.write(latest_version)
        
        print(f"Successfully downloaded cloudflared {latest_version}")
        return binary_path
    
    except Exception as e:
        print(f"Error: {str(e)}")
        if os.path.isfile(binary_path):
            print(f"Using existing binary: {binary_path}")
            return binary_path
        else:
            print("No existing binary found")
            return None

# Example usage
if __name__ == "__main__":
    # Get with default directory
    path = get_cloudflared()
    print(f"Cloudflared path: {path}")
    
    # Get with custom directory
    custom_path = get_cloudflared("C:/my_drivers")
    print(f"Cloudflared custom path: {custom_path}")
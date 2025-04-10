" To start the sangeet server with venv and isolated    "
import sys
import os
import socket
import subprocess
from sangeet_premium import venv_check
from sangeet_premium.utils import venv_create


def is_connected():
    try:
        # Try to connect to Google's public DNS server
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except (socket.timeout, OSError):
        return False


def install_colorama():
    """
    Attempt to install colorama library with error handling
    and provide feedback about the installation process.
    """
    try:
        # Check if colorama is already installed
        import colorama
        print("Colorama is already installed.")
        return True
    except ImportError:
        try:
            # Attempt to install colorama using pip
            print("Colorama not found. Attempting to install...")
            if is_connected():
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'colorama'])
                print("Colorama successfully installed.")
                return True
            else:
                print("connect to web....")
        except subprocess.CalledProcessError:
            print("Error: Failed to install colorama.")
            return False
        except Exception as e:
            print(f"Unexpected error during installation: {e}")
            return False


install_colorama()

dependencies = [
    "Flask",
    "colorama",
    "python-dotenv",
    "ytmusicapi",
    "ntplib",
    "pytz",
    "yt-dlp",
    "bcrypt",
    "mutagen",
    "pyyaml",
    "requests",
    "pyfiglet",
    "redis==5.2.1",
    "gunicorn",
    "tqdm",
    "termcolor",
    "bs4==0.0.2"
]

if sys.platform.startswith("win"):
    dependencies.extend([
        "pywin32==308",
        "winshell==0.6"
    ])

with open(os.path.join(os.getcwd(), "requirements", "req.txt"), "w") as f:
    f.write("\n".join(dependencies))



if not os.path.exists(os.path.join(os.getcwd(), ".sangeet-premium-venv")):
    if is_connected():
        venv_create.create_env(
            "sangeet-premium-venv",
            os.path.join(os.getcwd(), "requirements", "req.txt"),
            os.path.join(os.getcwd(), "logs", "venve-logs"),
            os.path.join(os.getcwd(), "sangeet_server.py")
        )
    else:
        print("please connect device to internet as setup required..")
else:
    if venv_check.check_venv(os.path.join(os.getcwd(), ".sangeet-premium-venv")):
        if is_connected():
            print("Starting sangeet in online mode")
            venv_create.create_env(
                "sangeet-premium-venv",
                os.path.join(os.getcwd(), "requirements", "req.txt"),
                os.path.join(os.getcwd(), "logs", "venve-logs"),
                os.path.join(os.getcwd(), "sangeet_server.py")
            )
        else:
            if os.path.exists(os.path.join(os.getcwd(), ".sangeet-premium-venv")):
                print("starting sangeet in offline only mode... some features will be disabled")
                venv_create.create_env(
                    "sangeet-premium-venv",
                    os.path.join(os.getcwd(), "requirements", "req.txt"),
                    os.path.join(os.getcwd(), "logs", "venve-logs"),
                    os.path.join(os.getcwd(), "sangeet_server.py")
                )
            else:
                print("sorry to start project venv is needed but you arent connected to web..")
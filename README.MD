# Sangeet Premium Plus 🎵 v3.0.0 LTS

<p align="center">
  <img src="promo/logo.png" alt="Sangeet Premium Plus Logo" width="600" height="400"/>
</p>

A modern, open-source music player with a stunning user interface, smart recommendations, enhanced features, and high-quality audio streaming - completely free forever. This is the **Long Term Support (LTS)** edition.

## 📋 Table of Contents
- [Why Sangeet Premium Plus?](#-why-sangeet-premium-plus)
- [Features](#-features)
- [LTS Support (Long Term Support)](#-lts-support-long-term-support)
- [Requirements](#-requirements)
- [Installation](#-installation)
  - [Standard Installation](#standard-installation)
  - [Docker Installation](#docker-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

## 🌟 Why Sangeet Premium Plus?
In today's digital music landscape, listeners face several challenges:
- Major streaming platforms charge premium fees for high-quality audio
- Expensive subscription models with restrictive features
- Limited control over music organization and playback
- Closed ecosystems that lock users into specific platforms
- Algorithmic recommendations that often prioritize promoted content

Sangeet Premium Plus solves these issues by providing:
- Completely free, high-quality audio streaming
- Beautiful, responsive user interface
- Advanced music recommendation system
- Full control over your music library
- No ads, no subscriptions, no limitations
- Ability to share songs and embed them in other platforms

## ✨ Features
- **Stunning UI/UX**: Modern, intuitive interface with smooth animations.
- **Smart Recommendations**: AI-powered music suggestions based on your listening habits.
- **High-Quality Audio**: Crystal clear audio streaming via **lossless FLAC format**. *Note: FLAC files are large and require good network bandwidth for smooth playback.*
- **Library Management**: Organize your music collection effortlessly (including local files).
- **Cross-Platform**: Run the server on Windows, macOS, and Linux. Access via web browser.
- **Offline Mode**: Download your favorite tracks in FLAC format for offline listening.
- **Last.fm Integration**: Scrobble your listening history (Requires configuration).
- **Customizable Themes**: Personalize your player's appearance (basic themes included).
- **Lyrics Support**: **Enhanced** real-time synchronized lyrics fetching and display.
- **Smart Login System**: Secure authentication with OTP email verification.
- **Playlist System**: Create, manage, and share playlists with friends or import shared ones.
- **Songs Embed**: Embed your songs to other pages via iframe with customization options.
- **Share**: Share your favorite songs with friends via unique links.
- **Download**: Download songs directly in high-quality FLAC format.
- **Redis Integration**: Utilizes Redis for improved performance, caching, and session management.
- **Results Caching**: Faster search results and data retrieval through smart caching mechanisms.
- **User Issue Reporting & Admin Chat**: A dedicated system for users to report problems or communicate directly with server administrators via a web interface.
- **LTS Support**: Long Term Support focusing on stability and critical updates (see below).

## 🛡️ LTS Support (Long Term Support)
This v3.0.0 release is designated as an **LTS (Long Term Support)** version. Our approach to LTS focuses on providing stability and security over rapid feature additions. This means:

- Updates will prioritize critical bug fixes, security vulnerabilities, and essential compatibility adjustments.
- While designated LTS, it follows a *Conditional Term Support (CTS)* philosophy: updates are provided when necessary rather than on a fixed schedule.
- Focus is on maintaining core functionality and security for an extended period.
- Major new features might be reserved for future non-LTS versions.

This approach allows us to ensure the reliability of the LTS version while directing development resources effectively.

## 📋 Requirements
- **Python:** Version 3.12 or higher.
- **Redis:** Required for caching and session management. Running via Docker is recommended.
- **Docker & Docker Compose:** Required if using the Docker installation method (recommended) or for running Redis easily. Ensure Docker Desktop (Windows/Mac) or Docker Engine/Compose (Linux) is installed and running.
- **SMTP Server Access:** Credentials for an SMTP server (like Gmail with an App Password) are needed for email features (OTP verification, etc.).
- **Network Bandwidth:** **Good internet connectivity is required for smooth streaming**, especially given the use of high-quality FLAC audio. Slower connections may result in buffering or playback interruptions.
- **Dependencies:** Python packages listed in `requirements/req.txt`.
- **For Windows Environment** : Install  pywin32==308 and winshell==0.6 `pip install pywin32==308 winshell==0.6`

## 🚀 Installation

Choose one of the following methods:

### Standard Installation (Running Python App Directly)

This method requires you to manage Python dependencies and run Redis separately.

1.  **Install Prerequisites:**
    *   Ensure Python 3.12+ is installed and added to your system's PATH. Verify with `python --version` or `python3 --version`.
    *   Install Docker and Docker Compose (needed for easily running Redis). Download from the [official Docker website](https://www.docker.com/products/docker-desktop/).

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/easy-ware/Sangeet-Premium-Plus-v3.0.0.git
    cd Sangeet-Premium-Plus-v3.0.0
    ```

3.  **Install Python dependencies:**
    ```bash
    # It's recommended to use a virtual environment
    # python3 -m venv venv
    # source venv/bin/activate  # On Windows use `venv\Scripts\activate`

    pip install -r requirements/req.txt
    # or on some systems:
    python3 -m pip install -r requirements/req.txt
    ```

4.  **Configure environment variables:**
    *   Navigate to the `/config/` directory.
    *   Create a file named `.env`.
    *   Copy the structure from the example below (or `.env.example` if provided) into your `/config/.env` file.
    *   **IMPORTANT:** Fill in your actual credentials, paths, and generate a strong `FLASK_SECRET_KEY`.
    ```dotenv
    # --- /config/.env Configuration ---
    # (Use ALL CAPS for variable names)

    # SMTP Email Configuration (For OTP, etc.)
    SMTP_USER="your_email@gmail.com"       # Sending email address
    SMTP_PASSWORD="your_gmail_app_password" # Use Gmail App Password if using Gmail
    SMTP_HOST="smtp.gmail.com"             # Your SMTP host
    SMTP_PORT=587                          # Your SMTP port (587 for TLS, 465 for SSL)

    # Music File Paths
    LOCAL_SONGS_PATHS="music" # Semicolon-separated paths to scan existing music. Example: "/path/to/music1;/path/to/folder2"
    MUSIC_PATH="music"        # Where downloaded FLACs are stored by Sangeet. Example: "/data/sangeet_downloads"

    # Server Network Configuration
    SANGEET_BACKEND="http://localhost:5000" # Publicly accessible URL of your server. Example: https://music.yourdomain.com
    PORT=5000                               # Port the server runs on. Example: 80 or 5000

    # Database Configuration
    DB_PATH="database_files"               # Directory for SQLite files.

    # Admin Authentication
    ADMIN_PASSWORD="your_very_secure_admin_password" # Secure password for /view/issues. CHANGE THIS!

    # Application Internals & Metadata
    PROJECT_URL="https://github.com/easy-ware/Sangeet-Premium-Plus-v3.0.0"
    ADMIN_MAIL="your_admin_contact_email@example.com"
    CLOUDFLARE_TUNNEL="False"              # Set to "True" to enable Cloudflare Tunnel

    # Flask Specific Configuration
    # CRITICAL: Generate using: python -c 'import secrets; print(secrets.token_hex(32))'
    FLASK_SECRET_KEY="generate_a_very_long_random_secret_key_here"
    APP_VERSION="3.0.0"

    # Optional overrides (defaults usually fine):
    # SESSION_COOKIE_SECURE="False"        # Set to "True" if using HTTPS
    # LOG_LEVEL="INFO"                     # DEBUG, INFO, WARNING, ERROR
    # LOCAL_SONG_REFRESH_INTERVAL="30"     # Scan interval in seconds
    ```

5.  **Run Redis using Docker:**
    This command uses the provided `redis.yaml` to start a Redis container.
    ```bash
    # Make sure Docker Desktop or Docker Engine is running
    docker compose -f redis.yaml up -d
    ```
    (The `-d` runs Redis detached in the background). Verify it's running with `docker ps`.

6.  **Run the Sangeet Python Application:**
    Navigate back to the root directory of the project if you were in `/config/`.
    ```bash
    python sangeet_server.py
    # or on some systems:
    python3 sangeet_server.py
    ```
    The server should now start, and you'll see log output in your terminal.

### Docker Installation (Recommended - Runs Both Sangeet & Redis)

This method bundles the application and its dependencies, including Redis, into Docker containers.

1.  **Install Prerequisites:**
    *   Install Docker and Docker Compose. Download from the [official Docker website](https://www.docker.com/products/docker-desktop/). Ensure the Docker service/daemon is running.

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/easy-ware/Sangeet-Premium-Plus-v3.0.0.git
    cd Sangeet-Premium-Plus-v3.0.0
    ```

3.  **Configure environment variables:**
    *   Navigate to the `/config/` directory.
    *   Create a file named `.env`.
    *   Copy the structure from the Standard Installation (Step 4) into `/config/.env`.
    *   **Fill in your actual credentials.** Use relative paths like `music` and `database_files` for `MUSIC_PATH` and `DB_PATH` as these will map to Docker volumes.
    *   **Generate a strong `FLASK_SECRET_KEY`.**

4.  **Build and Run with Docker Compose:**
    Navigate back to the root directory of the project. This command uses the main `docker-compose.yml` file.
    ```bash
    # This builds the Sangeet image and starts Sangeet + Redis containers
    docker compose up --build
    ```
    To run in the background (detached mode):
    ```bash
    docker compose up --build -d
    ```
    You can view logs using `docker compose logs -f` (add `-f` to follow).

## ⚙️ Configuration

### Environment Variables
All primary configuration is managed via the `/config/.env` file. Refer to the detailed template in the Installation section. **Remember to restart the server (Python script or Docker container) after modifying the `.env` file for changes to take effect.**

### Directory Structure
The application expects and potentially creates the following directories relative to the root:
-   `/config` - Contains configuration files (`.env`, `config.yaml`, `dir_struc.json`).
-   `/database_files` - Stores various SQLite database files. (Mapped to volume in Docker)
-   `/music` - Default directory for storing downloaded music files (configurable via `MUSIC_PATH`). (Mapped to volume in Docker)
-   `/logs` - Stores server log files (`gunicorn_access.log`, `gunicorn_error.log`, `sangeet_server_*.log`).
-   `/res` - Stores downloaded `yt-dlp` and `ffmpeg` binaries.
-   `/locals` (if used) - May contain local metadata files.
-   `/payloads` (if used) - May contain assets like the browser extension.

## 🎮 Usage
Once the server is running, access the Sangeet Premium Plus interface by navigating to the URL defined in your `SANGEET_BACKEND` environment variable (e.g., `http://localhost:5000` if running locally with port 5000).

-   **Admin Panel:** Visit `/view/issues` to access the issue management dashboard (requires the `ADMIN_PASSWORD` from `.env`).
-   **Network Access:** For access from other devices, configure port forwarding or use the Cloudflare Tunnel feature (`CLOUDFLARE_TUNNEL="True"`). The Cloudflare URL (e.g., `your-tunnel-name.trycloudflare.com`) is often easiest. Direct local IP access might work but isn't guaranteed for all features.
-   **Streaming Quality:** Remember that FLAC provides high quality but requires significant bandwidth. If you experience buffering, check your network connection speed.

## 🔧 Troubleshooting

### Log Files
Check the following log files located in the `/logs` directory for detailed error information:
-   `sangeet_server_YYYY-MM-DD.log`: Main application logs (Flask/custom logging).
-   `gunicorn_access.log`: Access logs when running with Gunicorn (Docker default).
-   `gunicorn_error.log`: Error logs when running with Gunicorn (Docker default).

### Common Issues
1.  **Server Won't Start / Module Not Found:**
    *   Verify Python 3.12+.
    *   Ensure dependencies installed (`pip install -r requirements/req.txt`).
    *   Check terminal/log files for specific errors.
2.  **Redis Connection Error:**
    *   Verify the Redis container/service is running (`docker ps` or check system services).
    *   Ensure Redis is accessible on `localhost:6379` or the configured host/port. Check firewalls.
3.  **Can't Access From Other Devices:**
    *   Use Cloudflare Tunnel URL if enabled.
    *   Check server binding address (should be `0.0.0.0` for network access).
    *   Check firewall settings on the server and network.
4.  **Downloads/Streaming Failing:**
    *   Ensure `yt-dlp` and `ffmpeg` setup completed without errors (check initial startup logs).
    *   Verify `MUSIC_PATH` directory exists and has write permissions for the user/process running Sangeet (or for the Docker container).
    *   Check network connectivity to YouTube/YTMusic. High bandwidth is needed for FLAC streaming.

### Reporting Issues
If issues persist, please report them on the **official GitHub Issues page**:
[https://github.com/easy-ware/Sangeet-Premium-Plus-v3.0.0/issues](https://github.com/easy-ware/Sangeet-Premium-Plus-v3.0.0/issues)

Include: Description, Steps to Reproduce, Expected vs. Actual, Version (v3.0.0 LTS), System Info (OS, Python, Docker), Relevant Logs (from `/logs`), Screenshots.

## 🤝 Contributing
Contributions are welcome! Please follow these steps:

1.  **Fork the repository:** `https://github.com/easy-ware/Sangeet-Premium-Plus-v3.0.0.git`
2.  **Create a feature branch:** `git checkout -b feature/YourAmazingFeature`
3.  **Make your changes.** Adhere to existing code style.
4.  **Test your changes thoroughly.**
5.  **Commit your changes:** `git commit -m 'Add: Your amazing feature'`
6.  **Push to your branch:** `git push origin feature/YourAmazingFeature`
7.  **Open a Pull Request** against the `main` branch of the official repository.

## 📜 License
This project is licensed under the Sangeet Premium Plus License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
-   The developers of `yt-dlp`, `ytmusicapi`, Flask, Redis, Mutagen, and all other open-source libraries used.
-   The vibrant open-source community.
-   Contributors and users providing feedback.

## 📧 Contact
-   **GitHub:** [easy-ware](https://github.com/easy-ware)
-   **Issue Tracker:** [Sangeet Premium Plus Issues](https://github.com/easy-ware/Sangeet-Premium-Plus-v3.0.0/issues)

---
<p align="center">
  Made with ❤️ by the Sangeet Premium Plus Team
</p>
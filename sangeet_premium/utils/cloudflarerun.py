import subprocess
import threading
import re
from colorama import init, Fore, Style
import time

def monitor_output(pipe):
    """Monitor the process output for the Cloudflare URL"""
    while True:
        line = pipe.readline()
        if not line:
            break
        # Look for trycloudflare.com URL in the output
        url_match = re.search(r'https?://[a-zA-Z0-9-]+\.trycloudflare\.com', line.decode('utf-8', errors='ignore'))
        if url_match:
            print(f"{Fore.MAGENTA}""="*80)
            print(f"{Fore.GREEN}\nAccess on other devices URL : {url_match.group(0)}\n{Fore.GREEN}")
            print(f"{Fore.MAGENTA}""="*45)

def run_cloudflare(port , driver):
    # Command to run cloudflared tunnel
    try:
        result = subprocess.run(
            [driver, '--version'],
            capture_output=True,
            text=True,
            timeout=5  # 5 second timeout
        )
        
        # If driver command succeeded
        if result.returncode == 0:
            driver = driver
        else:
            driver="cloudflared"
    except:
        driver = "cloudflared"
    command = [
        driver,
        'tunnel',
        '--url',
        f'localhost:{port}',
        '--protocol',
        'http2'
    ]
    
    # Start the process in background
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Start monitoring threads for both stdout and stderr
    stdout_thread = threading.Thread(target=monitor_output, args=(process.stdout,))
    stderr_thread = threading.Thread(target=monitor_output, args=(process.stderr,))
    
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    
    stdout_thread.start()
    stderr_thread.start()
    
    return process

def main():
    # Replace with your port number
    port = 8000
    
    # Start the cloudflared process
    process = run_cloudflare(port)
    
    try:
        # Keep the script running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        process.terminate()

if __name__ == "__main__":
    main()

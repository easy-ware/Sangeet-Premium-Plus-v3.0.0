# Use Python 3.12 slim as the base image for a lightweight container
FROM python:3.12-slim

# Install system dependencies and Cloudflared in a single layer to reduce image size
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3-pip  ffmpeg wget curl gnupg && \
    pip install colorama && \
    # Set up Cloudflare's GPG key and repository
    curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null && \
    gpg --dearmor /usr/share/keyrings/cloudflare-main.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared jammy main" | tee /etc/apt/sources.list.d/cloudflared.list && \
    apt-get update && \
    apt-get install -y cloudflared && \
    # Clean up to minimize image size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /sangeet

# Expose port  for the application
EXPOSE 80

# Copy application files from the current directory to /sangeet in the container
COPY . /sangeet/
RUN python3 -m pip install -r requirements/req.txt
# Define the command to run the Python application
CMD ["python3", "/sangeet/sangeet_server.py"]
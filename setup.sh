#!/bin/bash

echo "🚀 Setting up Jetson Camera System..."

# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pip network-manager

# Install Python packages
pip3 install -r requirements.txt

# Create directories
mkdir -p templates

# Setup permissions for nmcli (để không cần sudo)
echo "$USER ALL=(ALL) NOPASSWD: /usr/bin/nmcli" | sudo tee /etc/sudoers.d/nmcli

# Copy systemd service
sudo cp jetson-camera.service /etc/systemd/system/
sudo sed -i "s/YOUR_USERNAME/$USER/g" /etc/systemd/system/jetson-camera.service
sudo sed -i "s|/home/YOUR_USERNAME|$HOME|g" /etc/systemd/system/jetson-camera.service

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable jetson-camera.service
sudo systemctl start jetson-camera.service

echo "Setup complete!"
echo "Access web interface at: http://$(hostname -I | awk '{print $1}'):8000"
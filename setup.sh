#!/bin/bash

# Setup script for Streamlit Cloud deployment
# This helps with package installation issues

mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml

# Pre-install build dependencies
apt-get update
apt-get install -y build-essential

# Upgrade pip to latest version
pip install --upgrade pip setuptools wheel

# Install packages in order to avoid conflicts
pip install numpy>=1.24.0
pip install pandas>=2.1.0
pip install tensorflow-cpu>=2.13.0 
#!/bin/bash
# ==============================================================================
# PRISM-MEC Azure Cloud Worker Deployment Script
# Deploys and starts the FastAPI task execution service on the remote Azure VM
# ==============================================================================

# Load configuration from .env if present
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

AZURE_USER=${AZURE_USER:-azureuser}
AZURE_HOST=${AZURE_HOST:-20.219.130.205}
AZURE_API_PORT=${AZURE_API_PORT:-8000}
AZURE_API_KEY=${AZURE_API_KEY:-prism_sec_healthcare_mec_2026}

echo "================================================================="
echo "Deploying PRISM-MEC Cloud Worker to Azure: ${AZURE_USER}@${AZURE_HOST}"
echo "================================================================="

# Create remote directory and copy server file
echo "1. Copying server files to Azure VM..."
ssh ${AZURE_USER}@${AZURE_HOST} "mkdir -p ~/prism_mec"
scp azure/server.py ${AZURE_USER}@${AZURE_HOST}:~/prism_mec/

# Setup Python environment on remote VM and start service
echo "2. Installing requirements and launching worker in background..."
ssh ${AZURE_USER}@${AZURE_HOST} bash -c "'
    cd ~/prism_mec
    sudo apt-get update -y && sudo apt-get install -y python3-pip
    pip3 install fastapi uvicorn numpy pydantic
    # Stop existing instance if running
    pkill -f \"uvicorn server:app\" || true
    # Launch with API Key in background
    export AZURE_API_KEY=\"${AZURE_API_KEY}\"
    export AZURE_API_PORT=\"${AZURE_API_PORT}\"
    nohup python3 -m uvicorn server:app --host 0.0.0.0 --port ${AZURE_API_PORT} > worker.log 2>&1 &
    sleep 2
    echo \"Checking server status:\"
    curl -s http://localhost:${AZURE_API_PORT}/health
    echo \"\"
'"

echo "================================================================="
echo "Deployment completed!"
echo "You can test the endpoint: curl http://${AZURE_HOST}:${AZURE_API_PORT}/health"
echo "================================================================="

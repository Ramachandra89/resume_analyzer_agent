#!/bin/bash

# Update system and install dependencies
sudo apt-get update
sudo apt-get install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    git \
    nvidia-cuda-toolkit

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download LLAMA-2-7b model
python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf'); tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf'); model.save_pretrained('llama-2-7b'); tokenizer.save_pretrained('llama-2-7b')"

# Create systemd service file
sudo tee /etc/systemd/system/resume-coach.service << EOF
[Unit]
Description=Resume Coach AI Service
After=network.target

[Service]
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/venv/bin"
ExecStart=$(pwd)/venv/bin/streamlit run frontend/app.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and start service
sudo systemctl daemon-reload
sudo systemctl enable resume-coach
sudo systemctl start resume-coach

# Print status
echo "Service status:"
sudo systemctl status resume-coach 
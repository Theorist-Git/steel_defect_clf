#!/bin/bash

set -xe

echo "Updating package lists..."
apt-get update --allow-releaseinfo-change

echo "Installing python3 and pip3..."
apt-get install -y python3 python3-pip

echo "Installing tflite-runtime, numpy, pillow, and scikit-learn..."
pip3 install ai-edge-litert numpy pillow scikit-learn

python3 rpi1_inference.py

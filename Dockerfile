# Use NVIDIA's base CUDA image with Python 3.12 support
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set the architecture to x86_64 explicitly
ARG TARGETARCH=x86_64

ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install prerequisites
RUN apt-get update && apt-get install -y software-properties-common

# Add deadsnakes PPA for Python 3.12
RUN add-apt-repository -y ppa:deadsnakes/ppa && apt-get update

# Ensure the system is updated and install Python 3.12
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3.12-venv

RUN python3.12 -m ensurepip --upgrade

# Ensure pip, setuptools, and wheel are up to date
RUN python3.12 -m pip install --no-cache-dir --upgrade setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

# COPY requirements_dev.txt .
# RUN python3.12 -m pip install --no-cache-dir -r requirements_dev.txt
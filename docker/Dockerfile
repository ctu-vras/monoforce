FROM ros:noetic-perception

# Install dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    bridge-utils \
    build-essential \
    htop \
    net-tools \
    screen \
    tmux \
    vim \
    wget \
    curl \
    git \
    sshfs \
    python3-pip \
    python3-catkin-tools \
    ros-noetic-ros-numpy \
    ros-noetic-grid-map

# Install python dependencies
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# Install python requirements
ENV CATKIN_WS /root/catkin_ws
COPY requirements.txt .
RUN pip install -r requirements.txt

# NVIDIA Warp
RUN pip install https://github.com/NVIDIA/warp/releases/download/v1.0.2/warp_lang-1.0.2-py3-none-manylinux2014_x86_64.whl

# Set python3 as default
RUN ln -s /usr/bin/python3 /usr/bin/python

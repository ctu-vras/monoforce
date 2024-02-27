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
    mayavi2 \
    python3-pip \
    python3-catkin-tools \
    ros-noetic-ros-numpy \
    ros-noetic-jsk-rviz-plugins \
    ros-noetic-rviz \
    ros-noetic-grid-map

# Install python dependencies
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /monoforce

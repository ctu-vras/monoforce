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
    ros-noetic-grid-map \
    ros-noetic-husky-description \
    ros-noetic-xacro \
    ros-noetic-robot-state-publisher \
    ros-noetic-joint-state-publisher

# Install python dependencies
RUN pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install python requirements
ENV CATKIN_WS /root/catkin_ws
COPY requirements.txt .
RUN pip install -r requirements.txt

# NVIDIA Warp
RUN pip install https://github.com/NVIDIA/warp/releases/download/v1.0.2/warp_lang-1.0.2-py3-none-manylinux2014_x86_64.whl

# Set python3 as default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy ROS package
COPY ./ $CATKIN_WS/src/monoforce/

# Build ROS package
WORKDIR $CATKIN_WS
RUN catkin config \
      --extend /opt/ros/$ROS_DISTRO \
      --cmake-args \
        -DCMAKE_BUILD_TYPE=Release && \
    catkin build && \
    sed -i '/exec "$@"/i \
            source "/root/catkin_ws/devel/setup.bash"' /ros_entrypoint.sh

# Clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*
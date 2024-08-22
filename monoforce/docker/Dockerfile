# Use the official ROS Noetic perception image on Ubuntu 20.04
FROM ros:noetic-perception

# Set environment variables
ENV ROS_DISTRO=noetic

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-vcstool \
    python3-rosdep \
    python3-catkin-tools \
    ros-noetic-husky* \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create and setup the catkin workspace
RUN mkdir -p /root/traversability_ws/src/
WORKDIR /root/traversability_ws/src/

# Clone the monoforce repository
RUN git clone https://github.com/ctu-vras/monoforce.git

# Install Python requirements
WORKDIR /root/traversability_ws/src/monoforce/monoforce/docker/
# Pytorch and Pytorch3D
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu113
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
# NVIDIA Warp
RUN pip install https://github.com/NVIDIA/warp/releases/download/v1.0.2/warp_lang-1.0.2-py3-none-manylinux2014_x86_64.whl
# Other requirements
RUN pip install -r requirements.txt

# Set Python3 as the default Python version
RUN ln -s /usr/bin/python3 /usr/bin/python

# Initialize wstool and merge dependencies
WORKDIR /root/traversability_ws/src/
RUN wstool init
RUN wstool merge monoforce/monoforce_gazebo/dependencies.rosinstall
RUN wstool merge monoforce/monoforce_navigation/dependencies.rosinstall
RUN wstool up -j 4

# Configure and build the workspace
WORKDIR /root/traversability_ws/
RUN catkin init
RUN catkin config --extend /opt/ros/$ROS_DISTRO/
RUN catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
RUN pip install rospkg==1.5.1
RUN apt update && rosdep install --from-paths src --ignore-src -r -y

# Build the workspace
RUN catkin build

# Source the workspace setup script
RUN echo "source /root/traversability_ws/devel/setup.bash" >> ~/.bashrc

# Set the default command to bash
CMD ["bash"]

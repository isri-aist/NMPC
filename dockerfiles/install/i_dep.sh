#!/bin/sh
set -e

apt-get update
apt-mark hold ros-*
apt-get upgrade -y

basic_dep="git \
           curl \
           nano \
           vim \
           python3-pip \
           python3-colcon-common-extensions \
           wget \
           ninja-build \
           x11-apps"

apt update
apt install -y python3-colcon-clean doxygen texlive-latex-base texlive-fonts-recommended texlive-latex-extra dvipng
apt-get install -y $basic_dep
apt-get update
apt-get upgrade -y

mkdir /root/catkin_ws
cd /root/catkin_ws

# Clean up
apt-get autoremove -y
apt-get clean -y
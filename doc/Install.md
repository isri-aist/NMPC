## Install

### Requirements
- Compiler supporting C++17
- Tested on `Ubuntu 20.04 / ROS Noetic` and `Ubuntu 18.04 / ROS Melodic`

### Installation procedure
It is assumed that ROS is installed.

1. Setup catkin workspace.
```bash
$ mkdir -p ~/ros/ws_nmpc/src
$ cd ~/ros/ws_nmpc
$ wstool init src
$ wstool set -t src isri-aist/NMPC git@github.com:isri-aist/NMPC.git --git -y
$ wstool update -t src
```

2. Install dependent packages.
```bash
$ source /opt/ros/${ROS_DISTRO}/setup.bash
$ rosdep install -y -r --from-paths src --ignore-src
```

3. Build a package.
```bash
$ catkin build -DCMAKE_BUILD_TYPE=RelWithDebInfo --catkin-make-args all tests
# For best performance
$ catkin build -DOPTIMIZE_FOR_NATIVE=ON -DCMAKE_BUILD_TYPE=Release --catkin-make-args all tests
```

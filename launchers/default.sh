#!/bin/bash

source /environment.sh

source /opt/ros/noetic/setup.bash
source /code/devel/setup.bash --extend

exec roslaunch object_detection object_detection_node.launch veh:=$VEHICLE_NAME
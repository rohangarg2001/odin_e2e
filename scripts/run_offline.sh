#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source setup_env.sh
# Keep offline output on Curiosity, separate from the robot's ROS domain.
export ROS_DOMAIN_ID=228 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_STATIC_PEERS FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE
exec .venv/bin/python -u scripts/process_bag.py "$@"

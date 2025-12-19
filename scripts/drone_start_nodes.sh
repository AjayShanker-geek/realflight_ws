#!/usr/bin/env bash
set -euo pipefail

# Per-drone launcher: starts the offboard state machine and the trajectory follower for this DRONE_ID.
# No coordinator is launched here; run the coordinator once on ground control with the companion script.

DRONE_ID="${DRONE_ID:-0}"
TOTAL_DRONES="${TOTAL_DRONES:-3}"
CSV_BASE_PATH="${CSV_BASE_PATH:-data/3drone_trajectories_new}"
YAW_SETPOINT="${YAW_SETPOINT:-3.1415926}"
TAKEOFF_TIME="${TAKEOFF_TIME:-5.0}"
GOTO_TIME="${GOTO_TIME:-10.0}"
USE_RAW_TRAJ="${USE_RAW_TRAJ:-false}"

SESSION_NAME="drone_nodes_${DRONE_ID}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source ROS environments
set +u
source /opt/ros/humble/setup.bash
source "$WS_DIR/install/setup.bash"
set -u

# Kill any old session for this drone
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# Create a two-pane session: LEFT follow_traj_node, RIGHT offboard_sync_goto_node
tmux new-session -d -s "$SESSION_NAME" -c "$WS_DIR"
tmux split-window -h -t "$SESSION_NAME" -c "$WS_DIR"

readarray -t PANES < <(tmux list-panes -t "$SESSION_NAME" -F '#{pane_id}')
LEFT_PANE="${PANES[0]}"
RIGHT_PANE="${PANES[1]}"

CSV_PATH="${CSV_BASE_PATH}/drone_${DRONE_ID}_traj_smoothed_100hz.csv"

# LEFT: trajectory follower (publishes TRAJ setpoints when state machine enters TRAJ)
tmux send-keys -t "$LEFT_PANE" "cd $WS_DIR" C-m
tmux send-keys -t "$LEFT_PANE" "source /opt/ros/humble/setup.bash && source $WS_DIR/install/setup.bash" C-m
tmux send-keys -t "$LEFT_PANE" "echo \"Starting follow_traj_node for drone ${DRONE_ID} (total=${TOTAL_DRONES})\"" C-m
tmux send-keys -t "$LEFT_PANE" \
  "ros2 run traj_test follow_traj_node -- ${DRONE_ID} ${TOTAL_DRONES} --ros-args -p csv_path:=${CSV_PATH} -p yaw_setpoint:=${YAW_SETPOINT}" C-m

# RIGHT: offboard sync GOTO state machine (handles takeoff + goto + state publication)
tmux send-keys -t "$RIGHT_PANE" "cd $WS_DIR" C-m
tmux send-keys -t "$RIGHT_PANE" "source /opt/ros/humble/setup.bash && source $WS_DIR/install/setup.bash" C-m
tmux send-keys -t "$RIGHT_PANE" "echo \"Starting sync_goto.launch.py for drone ${DRONE_ID} (raw_traj=${USE_RAW_TRAJ})\"" C-m
tmux send-keys -t "$RIGHT_PANE" \
  "ros2 launch offboard_state_machine sync_goto.launch.py drone_id:=${DRONE_ID} traj_base_dir:=${CSV_BASE_PATH} use_raw_traj:=${USE_RAW_TRAJ} takeoff_time:=${TAKEOFF_TIME} goto_time:=${GOTO_TIME}" C-m

tmux attach-session -t "$SESSION_NAME"

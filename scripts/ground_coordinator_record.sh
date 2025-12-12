#!/usr/bin/env bash
set -euo pipefail

# Ground-control launcher: runs the single swarm GOTO/TRAJ coordinator and records all topics to a rosbag.

TOTAL_DRONES="${TOTAL_DRONES:-3}"
DRONE_IDS_CSV="${DRONE_IDS_CSV:-0,1,2}"
TAKEOFF_ALT="${TAKEOFF_ALT:-0.4}"
ALT_TOL="${ALT_TOL:-0.05}"
HOVER_WAIT_TIME="${HOVER_WAIT_TIME:-5.0}"

SESSION_NAME="ground_coordinator"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

set +u
source /opt/ros/humble/setup.bash
source "$WS_DIR/install/setup.bash"
set -u

tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# Two panes: LEFT coordinator, RIGHT rosbag -a
tmux new-session -d -s "$SESSION_NAME" -c "$WS_DIR"
tmux split-window -h -t "$SESSION_NAME" -c "$WS_DIR"

readarray -t PANES < <(tmux list-panes -t "$SESSION_NAME" -F '#{pane_id}')
LEFT_PANE="${PANES[0]}"
RIGHT_PANE="${PANES[1]}"

DRONE_IDS_YAML="[${DRONE_IDS_CSV}]"

# LEFT: coordinator (singleton)
tmux send-keys -t "$LEFT_PANE" "cd $WS_DIR" C-m
tmux send-keys -t "$LEFT_PANE" "source /opt/ros/humble/setup.bash && source $WS_DIR/install/setup.bash" C-m
tmux send-keys -t "$LEFT_PANE" "echo \"Starting swarm_goto_coordinator_node (drones=${DRONE_IDS_CSV}, hover_wait=${HOVER_WAIT_TIME}s)\"" C-m
tmux send-keys -t "$LEFT_PANE" \
  "ros2 run traj_test swarm_goto_coordinator_node --ros-args -p total_drones:=${TOTAL_DRONES} -p drone_ids:=${DRONE_IDS_YAML} -p takeoff_alt:=${TAKEOFF_ALT} -p alt_tol:=${ALT_TOL} -p hover_wait_time:=${HOVER_WAIT_TIME}" C-m

# RIGHT: rosbag all topics
BAG_DIR="$WS_DIR/rosbags"
tmux send-keys -t "$RIGHT_PANE" "cd $WS_DIR" C-m
tmux send-keys -t "$RIGHT_PANE" "source /opt/ros/humble/setup.bash && source $WS_DIR/install/setup.bash" C-m
tmux send-keys -t "$RIGHT_PANE" "mkdir -p $BAG_DIR" C-m
tmux send-keys -t "$RIGHT_PANE" "BAG_PATH=${BAG_DIR}/gc_$(date +%Y%m%d_%H%M%S)" C-m
tmux send-keys -t "$RIGHT_PANE" "echo \"Recording all topics to \$BAG_PATH\"" C-m
tmux send-keys -t "$RIGHT_PANE" "ros2 bag record -a -o \$BAG_PATH" C-m

tmux attach-session -t "$SESSION_NAME"

#!/usr/bin/env bash
set -euo pipefail

# Launch smooth-feedback PVA (realflight) with optional trajectory override.
# Usage: ./scripts/pva_feedback.sh <trajectory_directory>
# If no arg is given, uses data_root in the template YAML.

DRONE_ID="${DRONE_ID:-0}"
SESSION_NAME="pva_feedback_${DRONE_ID}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

resolve_traj_dir() {
  local input="$1"
  if [[ -z "$input" ]]; then
    echo ""
    return
  fi
  if [[ -d "$input" ]]; then
    realpath "$input"
    return
  fi
  if [[ -d "$WS_DIR/$input" ]]; then
    realpath "$WS_DIR/$input"
    return
  fi
  echo "Trajectory directory not found: $input" >&2
  exit 1
}

TRAJ_DIR="$(resolve_traj_dir "${1:-}")"
PARAMS_TEMPLATE="$WS_DIR/src/pva_control/config/pva_smooth_feedback_realflight_params.yaml"
if [[ ! -f "$PARAMS_TEMPLATE" ]]; then
  echo "Params template not found: $PARAMS_TEMPLATE" >&2
  exit 1
fi

PARAMS_DIR="$WS_DIR/tmp"
mkdir -p "$PARAMS_DIR"
PARAMS_FILE="${PARAMS_DIR}/pva_smooth_realflight_params_${DRONE_ID}_$(date +%Y%m%d_%H%M%S).yaml"
if [[ -n "$TRAJ_DIR" ]]; then
  sed "s#^[[:space:]]*data_root:.*#    data_root: \"${TRAJ_DIR}\"#" "$PARAMS_TEMPLATE" > "$PARAMS_FILE"
else
  cp "$PARAMS_TEMPLATE" "$PARAMS_FILE"
fi

echo "Using params file: $PARAMS_FILE"
if [[ -n "$TRAJ_DIR" ]]; then
  echo "Data root overridden to: $TRAJ_DIR"
fi

# Source ROS without nounset to avoid upstream issues
set +u
source /opt/ros/humble/setup.bash
source "$WS_DIR/install/setup.bash"
set -u

# tmux session just for the PVA smooth feedback node
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
tmux new-session -d -s "$SESSION_NAME" -c "$WS_DIR"
tmux split-window -h -t "$SESSION_NAME" -c "$WS_DIR"
readarray -t PANES < <(tmux list-panes -t "$SESSION_NAME" -F '#{pane_id}')
LEFT_PANE="${PANES[0]}"
RIGHT_PANE="${PANES[1]}"
RECORD_PANE=$(tmux split-window -v -t "$RIGHT_PANE" -c "$WS_DIR" -P -F '#{pane_id}')

# LEFT: PVA smooth realflight launch
tmux send-keys -t "$LEFT_PANE" "cd $WS_DIR" C-m
tmux send-keys -t "$LEFT_PANE" "source /opt/ros/humble/setup.bash && source $WS_DIR/install/setup.bash" C-m
tmux send-keys -t "$LEFT_PANE" "ros2 launch pva_control pva_smooth_realflight_swarm.launch.py params_file:=${PARAMS_FILE}" C-m

# RIGHT: placeholder for sync_goto/manual commands
tmux send-keys -t "$RIGHT_PANE" "cd $WS_DIR" C-m
tmux send-keys -t "$RIGHT_PANE" "source /opt/ros/humble/setup.bash && source $WS_DIR/install/setup.bash" C-m
tmux send-keys -t "$RIGHT_PANE" "echo \"Run sync_goto or other commands here\"" C-m

# RECORD: rosbag
PX4_NAMESPACE=$([[ \"$DRONE_ID\" -eq 0 ]] && echo \"/fmu/\" || echo \"/px4_${DRONE_ID}/fmu/\")
STATE_CMD_TOPIC=\"/state/command_drone_${DRONE_ID}\"
STATE_STATE_TOPIC=\"/state/state_drone_${DRONE_ID}\"
TRAJ_TOPIC=\"${PX4_NAMESPACE}in/trajectory_setpoint\"
LOCAL_POS_TOPIC=\"${PX4_NAMESPACE}out/vehicle_local_position\"
BAG_DIR=\"$WS_DIR/rosbags\"
mkdir -p \"$BAG_DIR\"
BAG_PATH=\"${BAG_DIR}/drone_${DRONE_ID}_$(date +%Y%m%d_%H%M%S)\"

tmux send-keys -t "$RECORD_PANE" "cd $WS_DIR" C-m
tmux send-keys -t "$RECORD_PANE" "source /opt/ros/humble/setup.bash && source $WS_DIR/install/setup.bash" C-m
tmux send-keys -t "$RECORD_PANE" "echo \"Recording ROS 2 bag for drone ${DRONE_ID}\"" C-m
tmux send-keys -t "$RECORD_PANE" "echo \"Topics:\"; echo \"  - ${STATE_CMD_TOPIC}\"; echo \"  - ${STATE_STATE_TOPIC}\"; echo \"  - ${TRAJ_TOPIC}\"; echo \"  - ${LOCAL_POS_TOPIC}\"" C-m
tmux send-keys -t "$RECORD_PANE" "echo \"Output: ${BAG_PATH}\"" C-m
tmux send-keys -t "$RECORD_PANE" "ros2 bag record -o ${BAG_PATH} ${STATE_CMD_TOPIC} ${STATE_STATE_TOPIC} ${TRAJ_TOPIC} ${LOCAL_POS_TOPIC}" C-m

tmux attach-session -t "$SESSION_NAME"

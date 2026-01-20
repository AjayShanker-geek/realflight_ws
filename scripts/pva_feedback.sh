#!/usr/bin/env bash
set -euo pipefail

# Launch smooth-feedback PVA (realflight) + sync goto with bag recording.
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

extract_data_root() {
  local template="$1"
  local line
  line=$(grep -E "^[[:space:]]*data_root:" "$template" | head -n 1 || true)
  if [[ -z "$line" ]]; then
    echo ""
    return
  fi
  echo "$line" | sed -E 's/^[[:space:]]*data_root:[[:space:]]*//; s/[[:space:]]*$//; s/^\"//; s/\"$//'
}

TRAJ_DIR_OVERRIDE="${1:-}"
TRAJ_DIR="$(resolve_traj_dir "$TRAJ_DIR_OVERRIDE")"
PARAMS_TEMPLATE="$WS_DIR/src/pva_control/config/pva_smooth_feedback_realflight_params.yaml"
if [[ ! -f "$PARAMS_TEMPLATE" ]]; then
  echo "Params template not found: $PARAMS_TEMPLATE" >&2
  exit 1
fi

if [[ -n "$TRAJ_DIR" ]]; then
  EFFECTIVE_TRAJ_DIR="$TRAJ_DIR"
else
  DEFAULT_TRAJ_DIR="$(extract_data_root "$PARAMS_TEMPLATE")"
  if [[ -z "$DEFAULT_TRAJ_DIR" ]]; then
    echo "data_root not found in template: $PARAMS_TEMPLATE" >&2
    exit 1
  fi
  EFFECTIVE_TRAJ_DIR="$(resolve_traj_dir "$DEFAULT_TRAJ_DIR")"
fi
TRAJ_DIR_ESCAPED="$(printf '%q' "$EFFECTIVE_TRAJ_DIR")"

PARAMS_DIR="$WS_DIR/tmp"
mkdir -p "$PARAMS_DIR"
PARAMS_FILE="${PARAMS_DIR}/pva_smooth_realflight_params_${DRONE_ID}_$(date +%Y%m%d_%H%M%S).yaml"
sed "s#^[[:space:]]*data_root:.*#    data_root: \"${EFFECTIVE_TRAJ_DIR}\"#" "$PARAMS_TEMPLATE" > "$PARAMS_FILE"
PARAMS_FILE_ESCAPED="$(printf '%q' "$PARAMS_FILE")"

echo "Using params file: $PARAMS_FILE"
echo "Trajectory directory for PVA + state machine: $EFFECTIVE_TRAJ_DIR"
if [[ -n "$TRAJ_DIR_OVERRIDE" ]]; then
  echo "Data root overridden via argument."
else
  echo "Data root sourced from template."
fi

# Source ROS without nounset to avoid upstream issues
set +u
source /opt/ros/humble/setup.bash
source "$WS_DIR/install/setup.bash"
set -u

if [[ "$DRONE_ID" -eq 0 ]]; then
  PX4_NAMESPACE="/fmu/"
else
  PX4_NAMESPACE="/px4_${DRONE_ID}/fmu/"
fi

STATE_CMD_TOPIC="/state/command_drone_${DRONE_ID}"
STATE_STATE_TOPIC="/state/state_drone_${DRONE_ID}"
TRAJ_TOPIC="${PX4_NAMESPACE}in/trajectory_setpoint"
LOCAL_POS_TOPIC="${PX4_NAMESPACE}out/vehicle_local_position"

# Build rosbag topic list; add payload mocap feeds when DRONE_ID=2
BAG_TOPICS=(
  "${STATE_CMD_TOPIC}"
  "${STATE_STATE_TOPIC}"
  "${TRAJ_TOPIC}"
  "${LOCAL_POS_TOPIC}"
)
if [[ "$DRONE_ID" -eq 2 ]]; then
  BAG_TOPICS+=(
    "/vrpn_mocap/multilift_payload/pose"
    "/vrpn_mocap/multilift_payload/twist"
    "/vrpn_mocap/multilift_payload/accel"
  )
fi
BAG_TOPICS_STR="${BAG_TOPICS[*]}"

# tmux session for smooth PVA, state machine, and rosbag
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
tmux send-keys -t "$LEFT_PANE" "ros2 launch pva_control pva_smooth_realflight_swarm.launch.py params_file:=${PARAMS_FILE_ESCAPED}" C-m

# RIGHT: sync_goto state machine with same trajectory directory
tmux send-keys -t "$RIGHT_PANE" "cd $WS_DIR" C-m
tmux send-keys -t "$RIGHT_PANE" "sleep 5 && source /opt/ros/humble/setup.bash && source $WS_DIR/install/setup.bash" C-m
tmux send-keys -t "$RIGHT_PANE" "ros2 launch offboard_state_machine sync_goto.launch.py traj_base_dir:=${TRAJ_DIR_ESCAPED}" C-m

# RECORD: rosbag
BAG_DIR="$WS_DIR/rosbags"
mkdir -p "$BAG_DIR"
BAG_PATH="${BAG_DIR}/drone_${DRONE_ID}_$(date +%Y%m%d_%H%M%S)"

tmux send-keys -t "$RECORD_PANE" "cd $WS_DIR" C-m
tmux send-keys -t "$RECORD_PANE" "source /opt/ros/humble/setup.bash && source $WS_DIR/install/setup.bash" C-m
tmux send-keys -t "$RECORD_PANE" "echo \"Recording ROS 2 bag for drone ${DRONE_ID}\"" C-m
tmux send-keys -t "$RECORD_PANE" "echo \"Topics:\"; for t in ${BAG_TOPICS_STR}; do echo \"  - \$t\"; done" C-m
tmux send-keys -t "$RECORD_PANE" "echo \"Output: ${BAG_PATH}\"" C-m
tmux send-keys -t "$RECORD_PANE" "ros2 bag record -o ${BAG_PATH} ${BAG_TOPICS_STR}" C-m

tmux attach-session -t "$SESSION_NAME"

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

tmux send-keys -t "$SESSION_NAME" "cd $WS_DIR" C-m
tmux send-keys -t "$SESSION_NAME" "source /opt/ros/humble/setup.bash && source $WS_DIR/install/setup.bash" C-m
tmux send-keys -t "$SESSION_NAME" "ros2 launch pva_control pva_smooth_realflight_swarm.launch.py params_file:=${PARAMS_FILE}" C-m

tmux attach-session -t "$SESSION_NAME"

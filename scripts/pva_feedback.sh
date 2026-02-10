#!/usr/bin/env bash
set -euo pipefail

# Launch smooth-feedback PVA (realflight) + sync goto with MQ bridge.
# Usage: ./scripts/pva_feedback.sh [OPTIONS] [trajectory_directory]
# If no trajectory arg is given, uses data_root in the template YAML.

DRONE_ID="${DRONE_ID:-0}"
TOTAL_DRONES="${TOTAL_DRONES:-6}"
DRONE_IDS_INPUT="${DRONE_IDS:-}"
TOTAL_DRONES_SET=false
SESSION_NAME="pva_feedback_${DRONE_ID}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<EOF
Usage: pva_feedback.sh [OPTIONS] [trajectory_directory]

Options:
  -n, --num-drones N    Total number of drones in the swarm (default: ${TOTAL_DRONES})
  -i, --drone-ids IDS   Drone IDs (comma-separated like 0,1,2 or compact like 012)
  -h, --help            Show this help and exit

Environment:
  DRONE_ID selects which vehicle this invocation controls (default: 0)
EOF
}

TRAJ_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--num-drones)
      TOTAL_DRONES="$2"; TOTAL_DRONES_SET=true; shift 2 ;;
    -i|--drone-ids)
      DRONE_IDS_INPUT="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift; break ;;
    -*)
      echo "Unknown option: $1" >&2; usage; exit 1 ;;
    *)
      TRAJ_ARG="$1"; shift; break ;;
  esac
done

if [[ $# -gt 0 ]]; then
  echo "Unexpected extra arguments: $*" >&2
  usage
  exit 1
fi

if ! [[ "$TOTAL_DRONES" =~ ^[0-9]+$ ]] || (( TOTAL_DRONES < 1 )); then
  echo "Invalid --num-drones value: $TOTAL_DRONES (must be positive integer)" >&2
  exit 1
fi

normalize_drone_ids() {
  local raw="$1"
  raw="${raw// /}"
  if [[ -z "$raw" ]]; then
    echo ""
    return
  fi
  if [[ "$raw" == *","* ]]; then
    echo "$raw"
    return
  fi
  if [[ "$raw" =~ ^[0-9]+$ ]]; then
    # Single digit like "6" means total drones -> IDs 0..5 (unless total_drones==1).
    if [[ ${#raw} -eq 1 ]] && { [[ "$TOTAL_DRONES_SET" == "false" ]] || [[ "$TOTAL_DRONES" -eq "$raw" ]]; }; then
      TOTAL_DRONES="$raw"
      echo "$(seq -s, 0 $((TOTAL_DRONES - 1)))"
      return
    fi
    # Compact list like "012345" -> 0,1,2,3,4,5
    if [[ "$raw" =~ ^0[0-9]+$ ]]; then
      local ids=""
      local i
      for ((i=0; i<${#raw}; i++)); do
        ids+="${raw:i:1},"
      done
      echo "${ids%,}"
      return
    fi
  fi
  echo "$raw"
}

DRONE_IDS_INPUT="$(normalize_drone_ids "$DRONE_IDS_INPUT")"

if ! [[ "$TOTAL_DRONES" =~ ^[0-9]+$ ]] || (( TOTAL_DRONES < 1 )); then
  echo "Invalid --num-drones value after normalization: $TOTAL_DRONES (must be positive integer)" >&2
  exit 1
fi

if [[ -z "$DRONE_IDS_INPUT" ]]; then
  DRONE_IDS_INPUT="$(seq -s, 0 $((TOTAL_DRONES - 1)))"
fi

DRONE_IDS_INPUT="${DRONE_IDS_INPUT// /}"
IFS=',' read -r -a DRONE_IDS_ARR <<< "$DRONE_IDS_INPUT"

if [[ ${#DRONE_IDS_ARR[@]} -ne TOTAL_DRONES ]]; then
  echo "Mismatch: --drone-ids has ${#DRONE_IDS_ARR[@]} IDs but --num-drones is $TOTAL_DRONES" >&2
  exit 1
fi

declare -A SEEN_IDS=()
for id in "${DRONE_IDS_ARR[@]}"; do
  if ! [[ "$id" =~ ^[0-9]+$ ]]; then
    echo "Invalid drone id '$id' (must be non-negative integer)" >&2
    exit 1
  fi
  if [[ -n "${SEEN_IDS[$id]:-}" ]]; then
    echo "Duplicate drone id '$id'" >&2
    exit 1
  fi
  SEEN_IDS[$id]=1
done

if [[ -z "${SEEN_IDS[$DRONE_ID]:-}" ]]; then
  echo "DRONE_ID=$DRONE_ID is not listed in --drone-ids=$DRONE_IDS_INPUT" >&2
  exit 1
fi

DRONE_IDS_COMPACT="${DRONE_IDS_INPUT//,/}"
echo "Drone IDs: ${DRONE_IDS_INPUT} (compact: ${DRONE_IDS_COMPACT})"

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

TRAJ_DIR_OVERRIDE="${TRAJ_ARG:-}"
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

# tmux session for smooth PVA and state machine
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
tmux new-session -d -s "$SESSION_NAME" -c "$WS_DIR"
tmux split-window -h -t "$SESSION_NAME" -c "$WS_DIR"
readarray -t PANES < <(tmux list-panes -t "$SESSION_NAME" -F '#{pane_id}')
LEFT_PANE="${PANES[0]}"
RIGHT_PANE="${PANES[1]}"

# LEFT: PVA smooth realflight launch
tmux send-keys -t "$LEFT_PANE" "cd $WS_DIR" C-m
tmux send-keys -t "$LEFT_PANE" "export ROS_LOCALHOST_ONLY=1" C-m
tmux send-keys -t "$LEFT_PANE" "source /opt/ros/humble/setup.bash && source $WS_DIR/install/setup.bash" C-m
tmux send-keys -t "$LEFT_PANE" \
  "ros2 run pva_control pva_smooth_feedback_control_node ${DRONE_ID} ${TOTAL_DRONES} --ros-args --params-file ${PARAMS_FILE_ESCAPED}" C-m

# RIGHT: sync_goto state machine with same trajectory directory
tmux send-keys -t "$RIGHT_PANE" "cd $WS_DIR" C-m
tmux send-keys -t "$RIGHT_PANE" "export ROS_LOCALHOST_ONLY=1" C-m
tmux send-keys -t "$RIGHT_PANE" "sleep 5 && source /opt/ros/humble/setup.bash && source $WS_DIR/install/setup.bash" C-m
tmux send-keys -t "$RIGHT_PANE" \
  "ros2 launch offboard_state_machine sync_goto.launch.py drone_id:=${DRONE_ID} traj_base_dir:=${TRAJ_DIR_ESCAPED}" C-m

tmux attach-session -t "$SESSION_NAME"

#!/usr/bin/env bash
set -euo pipefail

# GCS launcher: ZMQ bridge + swarm coordinator (+ optional VRPN client).

DRONE_IDS="${DRONE_IDS:-}"
USE_VICON_ALTITUDE="${USE_VICON_ALTITUDE:-true}"
SESSION_NAME="gcs_mq"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TOPICS_FILE="$SCRIPT_DIR/topics_record.yaml"

usage() {
  cat <<EOF
Usage: gcs_mq_start.sh [OPTIONS]

Options:
  -i, --drone-ids IDS     Comma-separated drone IDs (default: from YAML)
  -h, --help              Show this help and exit

Environment:
  USE_VICON_ALTITUDE      true/false (default: ${USE_VICON_ALTITUDE})
  VICON_IP                required if USE_VICON_ALTITUDE=true
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--drone-ids)
      DRONE_IDS="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift; break ;;
    *)
      echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "${USE_VICON_ALTITUDE}" == "true" ]] && [[ -z "${VICON_IP:-}" ]]; then
  echo "ERROR: VICON_IP is required when USE_VICON_ALTITUDE=true" >&2
  exit 1
fi

if [[ ! -f "$TOPICS_FILE" ]]; then
  echo "ERROR: topics file not found: $TOPICS_FILE" >&2
  exit 1
fi

set +u
source /opt/ros/humble/setup.bash
source "$WS_DIR/install/setup.bash"
set -u

tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
tmux new-session -d -s "$SESSION_NAME" -c "$WS_DIR"
tmux split-window -h -t "$SESSION_NAME" -c "$WS_DIR"
readarray -t PANES < <(tmux list-panes -t "$SESSION_NAME" -F '#{pane_id}')
LEFT_PANE="${PANES[0]}"
RIGHT_PANE="${PANES[1]}"
VRPN_PANE=$(tmux split-window -v -t "$RIGHT_PANE" -c "$WS_DIR" -P -F '#{pane_id}')
BAG_PANE=$(tmux split-window -v -t "$LEFT_PANE" -c "$WS_DIR" -P -F '#{pane_id}')

tmux send-keys -t "$LEFT_PANE" "cd $WS_DIR" C-m
tmux send-keys -t "$LEFT_PANE" "export ROS_LOCALHOST_ONLY=1" C-m
tmux send-keys -t "$LEFT_PANE" "source /opt/ros/humble/setup.bash && source $WS_DIR/install/setup.bash" C-m
ZMQ_CMD="ros2 run zmq_state_bridge zmq_state_bridge_node --ros-args \
  --params-file $WS_DIR/src/zmq_state_bridge/config/zmq_state_bridge_shared.yaml \
  --params-file $WS_DIR/src/zmq_state_bridge/config/zmq_state_bridge_gcs.yaml"
if [[ -n "$DRONE_IDS" ]]; then
  ZMQ_CMD="$ZMQ_CMD -p drone_ids_csv:=${DRONE_IDS}"
fi
tmux send-keys -t "$LEFT_PANE" "$ZMQ_CMD" C-m

tmux send-keys -t "$RIGHT_PANE" "cd $WS_DIR" C-m
tmux send-keys -t "$RIGHT_PANE" "export ROS_LOCALHOST_ONLY=1" C-m
tmux send-keys -t "$RIGHT_PANE" "source /opt/ros/humble/setup.bash && source $WS_DIR/install/setup.bash" C-m
COORD_CMD="ros2 run zmq_state_bridge gcs_swarm_coordinator_node --ros-args \
  --params-file $WS_DIR/src/zmq_state_bridge/config/gcs_swarm_coordinator.yaml \
  -p use_vicon_altitude:=${USE_VICON_ALTITUDE}"
if [[ -n "$DRONE_IDS" ]]; then
  COORD_CMD="$COORD_CMD -p drone_ids_csv:=${DRONE_IDS}"
fi
tmux send-keys -t "$RIGHT_PANE" "$COORD_CMD" C-m

tmux send-keys -t "$VRPN_PANE" "cd $WS_DIR" C-m
tmux send-keys -t "$VRPN_PANE" "export ROS_LOCALHOST_ONLY=1" C-m
tmux send-keys -t "$VRPN_PANE" "source /opt/ros/humble/setup.bash && source $WS_DIR/install/setup.bash" C-m
if [[ "${USE_VICON_ALTITUDE}" == "true" ]]; then
  tmux send-keys -t "$VRPN_PANE" \
    "ros2 launch vrpn_mocap client.launch.yaml server:=${VICON_IP} port:=3883" C-m
else
  tmux send-keys -t "$VRPN_PANE" "echo \"VRPN client disabled (USE_VICON_ALTITUDE=false)\"" C-m
fi

BAG_DIR="$WS_DIR/rosbags"
mkdir -p "$BAG_DIR"
BAG_PATH="${BAG_DIR}/vrpn_pose_$(date +%Y%m%d_%H%M%S)"

readarray -t TOPICS < <(
  awk '/^[[:space:]]*-[[:space:]]+/{sub(/^[[:space:]]*-[[:space:]]+/,"");print}' "$TOPICS_FILE"
)
if [[ ${#TOPICS[@]} -eq 0 ]]; then
  echo "ERROR: no topics found in $TOPICS_FILE" >&2
  exit 1
fi

TOPICS_ESCAPED=()
for topic in "${TOPICS[@]}"; do
  printf -v topic_escaped '%q' "$topic"
  TOPICS_ESCAPED+=("$topic_escaped")
done

USE_REGEX="false"
for topic in "${TOPICS[@]}"; do
  if [[ "$topic" =~ [\*\?\|\[\]\(\)\+\^\$] ]]; then
    USE_REGEX="true"
    break
  fi
done

RECORD_CMD=""
if [[ "$USE_REGEX" == "true" ]]; then
  REGEX=""
  for topic in "${TOPICS[@]}"; do
    if [[ -z "$REGEX" ]]; then
      REGEX="(${topic})"
    else
      REGEX="${REGEX}|(${topic})"
    fi
  done
  RECORD_CMD="ros2 bag record -o ${BAG_PATH} -e '${REGEX}'"
else
  RECORD_CMD="ros2 bag record -o ${BAG_PATH} ${TOPICS_ESCAPED[*]}"
fi

tmux send-keys -t "$BAG_PANE" "cd $WS_DIR" C-m
tmux send-keys -t "$BAG_PANE" "export ROS_LOCALHOST_ONLY=1" C-m
tmux send-keys -t "$BAG_PANE" "source /opt/ros/humble/setup.bash && source $WS_DIR/install/setup.bash" C-m
tmux send-keys -t "$BAG_PANE" "echo \"Recording ROS 2 bag: VRPN pose topics\"" C-m
tmux send-keys -t "$BAG_PANE" "echo \"Topics file: ${TOPICS_FILE}\"" C-m
tmux send-keys -t "$BAG_PANE" "echo \"Topics loaded: ${#TOPICS[@]}\"" C-m
tmux send-keys -t "$BAG_PANE" "echo \"Output: ${BAG_PATH}\"" C-m
tmux send-keys -t "$BAG_PANE" "${RECORD_CMD}" C-m

tmux attach-session -t "$SESSION_NAME"

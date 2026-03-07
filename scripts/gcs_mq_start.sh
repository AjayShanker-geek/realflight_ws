#!/usr/bin/env bash
set -euo pipefail

# GCS launcher: ZMQ bridge + swarm coordinator (+ optional VRPN client).

DRONE_IDS="${DRONE_IDS:-}"
TOTAL_DRONES="${TOTAL_DRONES:-}"
USE_VICON_ALTITUDE="${USE_VICON_ALTITUDE:-true}"
SESSION_NAME="gcs_mq"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TOPICS_FILE="$SCRIPT_DIR/topics_record.yaml"
GCS_CFG_FILE="$WS_DIR/src/zmq_state_bridge/config/zmq_state_bridge_gcs.yaml"

usage() {
  cat <<EOF
Usage: gcs_mq_start.sh [OPTIONS]

Options:
  -i, --drone-ids IDS     Comma-separated drone IDs (default: from YAML)
  -n, --num-drones N      Use IDs 0..N-1 (if --drone-ids not set)
  -h, --help              Show this help and exit

Environment:
  TOTAL_DRONES            Same as --num-drones
  USE_VICON_ALTITUDE      true/false (default: ${USE_VICON_ALTITUDE})
  VICON_IP                required if USE_VICON_ALTITUDE=true
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--drone-ids)
      DRONE_IDS="$2"; shift 2 ;;
    -n|--num-drones)
      TOTAL_DRONES="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift; break ;;
    *)
      echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if ! [[ "$TOTAL_DRONES" =~ ^$|^[0-9]+$ ]] || { [[ -n "$TOTAL_DRONES" ]] && (( TOTAL_DRONES < 1 )); }; then
  echo "ERROR: --num-drones must be a positive integer" >&2
  exit 1
fi

if [[ "${USE_VICON_ALTITUDE}" == "true" ]] && [[ -z "${VICON_IP:-}" ]]; then
  echo "ERROR: VICON_IP is required when USE_VICON_ALTITUDE=true" >&2
  exit 1
fi

if [[ ! -f "$GCS_CFG_FILE" ]]; then
  echo "ERROR: GCS config not found: $GCS_CFG_FILE" >&2
  exit 1
fi

if [[ ! -f "$TOPICS_FILE" ]]; then
  echo "ERROR: topics file not found: $TOPICS_FILE" >&2
  exit 1
fi

read_yaml_csv() {
  local file="$1"
  local key="$2"
  sed -nE "s/^[[:space:]]*${key}:[[:space:]]*\"([^\"]*)\"[[:space:]]*$/\1/p" "$file" | head -n1
}

parse_csv_ids() {
  local csv="$1"
  local -n out_arr="$2"
  out_arr=()
  local token
  IFS=',' read -r -a __tmp <<< "${csv// /}"
  for token in "${__tmp[@]}"; do
    [[ -n "$token" ]] && out_arr+=("$token")
  done
}

DEFAULT_IDS_CSV="$(read_yaml_csv "$GCS_CFG_FILE" "drone_ids_csv")"
DEFAULT_IPS_CSV="$(read_yaml_csv "$GCS_CFG_FILE" "udp_state_drone_ips_csv")"

if [[ -z "$DRONE_IDS" ]]; then
  if [[ -n "$TOTAL_DRONES" ]]; then
    DRONE_IDS="$(seq -s, 0 $((TOTAL_DRONES - 1)))"
  else
    DRONE_IDS="$DEFAULT_IDS_CSV"
  fi
fi

if [[ -z "$DRONE_IDS" ]]; then
  echo "ERROR: no drone IDs set (pass -i/--drone-ids or set drone_ids_csv in $GCS_CFG_FILE)" >&2
  exit 1
fi

DRONE_IDS="${DRONE_IDS// /}"
parse_csv_ids "$DRONE_IDS" DRONE_IDS_ARR
if [[ ${#DRONE_IDS_ARR[@]} -eq 0 ]]; then
  echo "ERROR: parsed drone ID list is empty" >&2
  exit 1
fi

declare -A SEEN_IDS=()
for id in "${DRONE_IDS_ARR[@]}"; do
  if ! [[ "$id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid drone id '$id' (must be non-negative integer)" >&2
    exit 1
  fi
  if [[ -n "${SEEN_IDS[$id]:-}" ]]; then
    echo "ERROR: duplicate drone id '$id'" >&2
    exit 1
  fi
  SEEN_IDS[$id]=1
done

if [[ -n "$TOTAL_DRONES" ]] && (( ${#DRONE_IDS_ARR[@]} != TOTAL_DRONES )); then
  echo "ERROR: --num-drones=$TOTAL_DRONES but --drone-ids has ${#DRONE_IDS_ARR[@]} IDs" >&2
  exit 1
fi

if [[ -z "$TOTAL_DRONES" ]]; then
  TOTAL_DRONES="${#DRONE_IDS_ARR[@]}"
fi

SELECTED_IPS_CSV=""
if [[ -n "$DEFAULT_IPS_CSV" ]]; then
  parse_csv_ids "$DEFAULT_IDS_CSV" DEFAULT_IDS_ARR
  parse_csv_ids "$DEFAULT_IPS_CSV" DEFAULT_IPS_ARR

  if (( ${#DEFAULT_IDS_ARR[@]} != ${#DEFAULT_IPS_ARR[@]} )); then
    echo "ERROR: in $GCS_CFG_FILE, drone_ids_csv count (${#DEFAULT_IDS_ARR[@]}) does not match udp_state_drone_ips_csv count (${#DEFAULT_IPS_ARR[@]})" >&2
    exit 1
  fi

  declare -A IP_BY_ID=()
  for i in "${!DEFAULT_IDS_ARR[@]}"; do
    IP_BY_ID["${DEFAULT_IDS_ARR[$i]}"]="${DEFAULT_IPS_ARR[$i]}"
  done

  SELECTED_IPS_ARR=()
  for id in "${DRONE_IDS_ARR[@]}"; do
    if [[ -z "${IP_BY_ID[$id]:-}" ]]; then
      echo "ERROR: no UDP IP mapping for drone ID '$id' in $GCS_CFG_FILE (drone_ids_csv/udp_state_drone_ips_csv)" >&2
      exit 1
    fi
    SELECTED_IPS_ARR+=("${IP_BY_ID[$id]}")
  done
  SELECTED_IPS_CSV="$(IFS=','; echo "${SELECTED_IPS_ARR[*]}")"
fi

echo "Using drone IDs: ${DRONE_IDS} (total: ${TOTAL_DRONES})"
if [[ -n "$SELECTED_IPS_CSV" ]]; then
  echo "Using UDP state target IPs: ${SELECTED_IPS_CSV}"
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
ZMQ_CMD="$ZMQ_CMD -p drone_ids_csv:=${DRONE_IDS}"
if [[ -n "$SELECTED_IPS_CSV" ]]; then
  ZMQ_CMD="$ZMQ_CMD -p udp_state_drone_ips_csv:=${SELECTED_IPS_CSV}"
fi
tmux send-keys -t "$LEFT_PANE" "$ZMQ_CMD" C-m

tmux send-keys -t "$RIGHT_PANE" "cd $WS_DIR" C-m
tmux send-keys -t "$RIGHT_PANE" "export ROS_LOCALHOST_ONLY=1" C-m
tmux send-keys -t "$RIGHT_PANE" "source /opt/ros/humble/setup.bash && source $WS_DIR/install/setup.bash" C-m
COORD_CMD="ros2 run zmq_state_bridge gcs_swarm_coordinator_node --ros-args \
  --params-file $WS_DIR/src/zmq_state_bridge/config/gcs_swarm_coordinator.yaml \
  -p use_vicon_altitude:=${USE_VICON_ALTITUDE} \
  -p drone_ids_csv:=${DRONE_IDS}"
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

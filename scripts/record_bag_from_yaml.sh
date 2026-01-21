#!/usr/bin/env bash
set -euo pipefail

# Record a ROS 2 bag using topics declared in a YAML file.
# Usage:
#   DRONE_IDS="0,1" BAG_DIR=/tmp/rosbags BAG_NAME=my_run ./scripts/record_bag_from_yaml.sh [topics.yaml]
# - If no YAML file is provided, scripts/bag_topics.yaml is used.
# - YAML supports:
#     * top-level list of topics
#     * mapping with "topics" and/or "common_topics"
#     * "per_drone_topics" entries with {id} placeholder expanded with "drone_ids" (or DRONE_IDS env)
# - Set DRY_RUN=1 to only print the resolved topics.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

TOPIC_FILE="${1:-$SCRIPT_DIR/bag_topics.yaml}"
BAG_DIR="${BAG_DIR:-$WS_DIR/rosbags}"
BAG_NAME="${BAG_NAME:-bag_$(date +%Y%m%d_%H%M%S)}"
DRONE_IDS_ENV="${DRONE_IDS:-}"

if [[ ! -f "$TOPIC_FILE" ]]; then
  echo "Topic file not found: $TOPIC_FILE" >&2
  exit 1
fi

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ros2 is not on PATH. Source your ROS 2 environment first." >&2
  exit 1
fi

# Source ROS without nounset to avoid unbound variable issues from upstream scripts
set +u
source /opt/ros/humble/setup.bash
source "$WS_DIR/install/setup.bash"
set -u

mapfile -t TOPICS < <(python3 - <<'PY' "$TOPIC_FILE" "$DRONE_IDS_ENV")
import sys
from pathlib import Path

topic_file = Path(sys.argv[1])
drone_ids_override = sys.argv[2]

try:
    import yaml
except ImportError:
    sys.stderr.write("PyYAML is required to parse the topic file. Install with `pip install pyyaml`.\n")
    sys.exit(1)

if not topic_file.exists():
    sys.stderr.write(f"Topic file not found: {topic_file}\n")
    sys.exit(1)

data = yaml.safe_load(topic_file.read_text()) or {}


def normalize_list(value, label):
    items = []
    if value is None:
        return items
    if not isinstance(value, list):
        sys.stderr.write(f"{label} must be a list\n")
        sys.exit(1)
    for item in value:
        if not isinstance(item, str):
            sys.stderr.write(f"{label} entries must be strings\n")
            sys.exit(1)
        stripped = item.strip()
        if stripped:
            items.append(stripped)
    return items


topics = []
per_drone_topics = []
drone_ids = []

if isinstance(data, list):
    topics.extend(normalize_list(data, "top-level topics"))
elif isinstance(data, dict):
    topics.extend(normalize_list(data.get("topics", []), "topics"))
    topics.extend(normalize_list(data.get("common_topics", []), "common_topics"))
    per_drone_topics = normalize_list(
        data.get("per_drone_topics") or data.get("per_drone") or [], "per_drone_topics"
    )

    raw_ids = data.get("drone_ids") or []
    if drone_ids_override:
        raw_ids = [part for part in drone_ids_override.split(",") if part.strip()]
    if raw_ids:
        if not isinstance(raw_ids, list):
            sys.stderr.write("drone_ids must be a list\n")
            sys.exit(1)
        drone_ids = [str(item).strip() for item in raw_ids if str(item).strip()]
else:
    sys.stderr.write("Topic file must be a list or mapping\n")
    sys.exit(1)

if per_drone_topics and not drone_ids:
    sys.stderr.write("per_drone_topics set but no drone_ids found; skipping per-drone expansion.\n")
else:
    for drone_id in drone_ids:
        for template in per_drone_topics:
            topics.append(template.format(id=drone_id))

final = []
seen = set()
for topic in topics:
    if topic in seen:
        continue
    seen.add(topic)
    final.append(topic)

if not final:
    sys.stderr.write("No topics found in the YAML file.\n")
    sys.exit(1)

for topic in final:
    print(topic)
PY

if [[ "${#TOPICS[@]}" -eq 0 ]]; then
  echo "No topics found after parsing $TOPIC_FILE" >&2
  exit 1
fi

echo "Resolved ${#TOPICS[@]} topic(s) from $TOPIC_FILE:"
for t in "${TOPICS[@]}"; do
  echo "  - $t"
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 set; not recording."
  exit 0
fi

mkdir -p "$BAG_DIR"
BAG_PATH="$BAG_DIR/$BAG_NAME"

echo "Recording to $BAG_PATH"
ros2 bag record -o "$BAG_PATH" "${TOPICS[@]}"

# geom_multilift

Per-drone controllers for the multilift realflight setup with corrected ENU/NED/FRD transforms.

- `geom_multilift_node`: C++ port of the Python geometric multilift controller. Reads preprocessed offline data (payload/cable/Kfb), computes cable direction from mocap payload pose + drone odometry, and publishes attitude/thrust like `geom_multilift.py`, but one node per drone (`DRONE_ID`).
- `realflight_traj_node`: simple trajectory streamer kept for reference.

## What it does
- Loads the pre-processed CSV from `px4_offboard/save_traj.py` (`time,x,y,z,vx,vy,vz` in NED).
- Waits until the offboard FSM publishes `TRAJ` on `/state/state_drone_<id>`, optionally synchronizing with the rest of the swarm.
- Streams `px4_msgs/TrajectorySetpoint` for *only this drone* to its PX4 namespace (`/fmu/` or `/px4_<id>/fmu/`).
- Sends `END_TRAJ` state commands when the offline trajectory is finished.

## Realflight vs. simulation
- This node assumes motion-capture local position already uses the mocap/world origin.  
  No origin re-centering is applied (SITL odometry typically resets around the arming pose).
- Trajectory values are sent as-is from the CSV so they should already be expressed in the mocap frame you fly in.

## Build
From the workspace root:
```bash
colcon build --packages-select geom_multilift
source install/setup.bash
```

## Launch (geom_multilift)
Preprocess once (already done here for the default 3-quad set):
```bash
python3 src/geom_multilift/scripts/preprocess_traj.py
```

Then run one process per vehicle (DRONE_ID must match PX4 namespace). The node waits for `/state/state_drone_<id>` == `TRAJ` and uses the payload PoseStamped topic (default `/vrpn_mocap/multilift_payload/pose`) + vehicle odometry to build the control state.
```bash
export DRONE_ID=0
export TOTAL_DRONES=3
ros2 launch geom_multilift geom_multilift.launch.py \
  data_root:=data/preprocessed/rg_001_-001_3quad_l1m_100hz \
  timer_period:=0.01
```
Repeat per drone with a different `DRONE_ID`.

## Launch (trajectory-only)
Use one process per vehicle; the launch file also starts the offboard FSM with takeoff/goto pulled from the CSV:
```bash
export DRONE_ID=0          # unique per vehicle
export TOTAL_DRONES=3      # swarm size used for FSM sync
ros2 launch geom_multilift realflight_traj.launch.py \
  csv_base_path:=data/3drone_trajectories_001_-001 \
  timer_period:=0.01 \
  yaw_setpoint:=3.1415926
```
Repeat in separate shells for each drone with a different `DRONE_ID`. The launch reads `drone_<id>_traj.csv` under `csv_base_path` and passes the first point to the FSM for takeoff/goto.

## Run the trajectory node only
If the FSM is already running elsewhere:
```bash
export DRONE_ID=1
ros2 run geom_multilift realflight_traj_node 1 3 --ros-args \
  -p csv_base_path:=data/3drone_trajectories_001_-001 \
  -p timer_period:=0.01 \
  -p yaw_setpoint:=3.1415926
```

## Topics and parameters (geom_multilift_node)
- Subscribes: `payload_pose_topic` (default `/vrpn_mocap/multilift_payload/pose`), `/fmu/out/vehicle_odometry` (or `/px4_<id>/...`), `/state/state_drone_<id>`, `/fmu/out/vehicle_local_position_setpoint` (for accel fusion).
- Publishes: `/fmu/in/vehicle_attitude_setpoint_v1`, `/fmu/in/trajectory_setpoint`, `/state/command_drone_<id>` (END_TRAJ).
- Params: `data_root` (preprocessed directory), `timer_period`, `l` (cable length), gains `kq/kw/alpha_gain/z_weight/thrust_bias`.

## Notes / assumptions
- Preprocessed data lives under `data/preprocessed/rg_001_-001_3quad_l1m_100hz` (from the 3-quad set in `3quad_traj/...`). Adjust `data_root` to switch scenarios.
- Payload state is taken from `payload_pose_topic`; vel/acc/omega are finite-differenced (replace later with your Kalman filter if available).
- Cable direction is recomputed from payload pose + drone odometry in realflight (no SITL origin reset).

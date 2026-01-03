#!/usr/bin/env python3
"""
Summarize PVA debug logs (CSV).
Supports both pva_control and pva_feedback_control debug logs.
"""

import argparse
from typing import Dict, Optional

import numpy as np


STATE_NAMES = {
    0: "INIT",
    1: "ARMING",
    2: "TAKEOFF",
    3: "GOTO",
    4: "HOVER",
    5: "TRAJ",
    6: "END_TRAJ",
    7: "LAND",
    8: "DONE",
}


def load_log(path: str) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        raise RuntimeError(f"No data rows in {path}")
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def get_col(data: np.ndarray, name: str) -> Optional[np.ndarray]:
    if name not in data.dtype.names:
        return None
    return data[name]


def ratio_true(vec: Optional[np.ndarray]) -> Optional[float]:
    if vec is None:
        return None
    vec = np.asarray(vec)
    vec = vec[np.isfinite(vec)]
    if vec.size == 0:
        return None
    return float(np.mean(vec > 0.5))


def fmt_ratio(r: Optional[float]) -> str:
    if r is None:
        return "missing"
    return f"{r * 100.0:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PVA debug log CSV")
    parser.add_argument("log_path", help="Path to PVA debug CSV")
    args = parser.parse_args()

    data = load_log(args.log_path)
    t = get_col(data, "t")
    if t is not None and t.size > 1:
        duration = float(t[-1] - t[0])
        dt = np.diff(t)
        dt = dt[np.isfinite(dt)]
        dt_mean = float(np.mean(dt)) if dt.size else float("nan")
    else:
        duration = float("nan")
        dt_mean = float("nan")

    state = get_col(data, "state")
    state_duration = {}
    if state is not None and t is not None and t.size > 1:
        dt = np.diff(t)
        dt = np.append(dt, dt[-1]) if dt.size else np.array([0.0])
        for s, ds in zip(state.astype(int), dt):
            state_duration[s] = state_duration.get(s, 0.0) + float(ds)

    traj_started = ratio_true(get_col(data, "traj_started"))
    waiting_swarm = ratio_true(get_col(data, "waiting_swarm"))
    traj_completed = ratio_true(get_col(data, "traj_completed"))
    odom_ready = ratio_true(get_col(data, "odom_ready"))
    payload_ready = ratio_true(get_col(data, "payload_ready"))
    traj_loaded = ratio_true(get_col(data, "traj_loaded"))
    cable_loaded = ratio_true(get_col(data, "cable_loaded"))

    print(f"samples: {int(data.shape[0])}  duration: {duration:.3f}s  mean_dt: {dt_mean:.4f}s")
    print(f"traj_started: {fmt_ratio(traj_started)}  waiting_swarm: {fmt_ratio(waiting_swarm)}  traj_completed: {fmt_ratio(traj_completed)}")
    print(f"odom_ready: {fmt_ratio(odom_ready)}  payload_ready: {fmt_ratio(payload_ready)}")
    print(f"traj_loaded: {fmt_ratio(traj_loaded)}  cable_loaded: {fmt_ratio(cable_loaded)}")

    if state_duration:
        print("state_durations:")
        for key in sorted(state_duration.keys()):
            name = STATE_NAMES.get(key, str(key))
            print(f"  {name}: {state_duration[key]:.2f}s")


if __name__ == "__main__":
    main()

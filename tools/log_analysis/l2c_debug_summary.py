#!/usr/bin/env python3
"""
Summarize L2C debug logs (CSV).
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


def stats(vec: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
    if vec is None:
        return None
    vec = np.asarray(vec)
    vec = vec[np.isfinite(vec)]
    if vec.size == 0:
        return None
    return {
        "count": int(vec.size),
        "mean": float(np.mean(vec)),
        "min": float(np.min(vec)),
        "max": float(np.max(vec)),
        "p95": float(np.percentile(vec, 95)),
    }


def fmt_ratio(r: Optional[float]) -> str:
    if r is None:
        return "missing"
    return f"{r * 100.0:.1f}%"


def fmt_stats(s: Optional[Dict[str, float]]) -> str:
    if not s:
        return "missing"
    return f"mean={s['mean']:.4f} min={s['min']:.4f} max={s['max']:.4f} p95={s['p95']:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize L2C debug log CSV")
    parser.add_argument("log_path", help="Path to l2c debug CSV")
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

    payload_ready = ratio_true(get_col(data, "payload_ready"))
    odom_ready = ratio_true(get_col(data, "odom_ready"))
    local_pos_ready = ratio_true(get_col(data, "local_pos_ready"))
    init_pos_ready = ratio_true(get_col(data, "init_pos_ready"))

    xy_valid = ratio_true(get_col(data, "xy_valid"))
    z_valid = ratio_true(get_col(data, "z_valid"))
    vxy_valid = ratio_true(get_col(data, "v_xy_valid"))
    vz_valid = ratio_true(get_col(data, "v_z_valid"))
    acc_sp = ratio_true(get_col(data, "acc_sp_from_setpoint"))

    vec_norm_stats = stats(get_col(data, "vec_norm"))
    u_total_stats = stats(get_col(data, "u_total_norm"))

    print(f"samples: {int(data.shape[0])}  duration: {duration:.3f}s  mean_dt: {dt_mean:.4f}s")
    print(f"payload_ready: {fmt_ratio(payload_ready)}  odom_ready: {fmt_ratio(odom_ready)}  local_pos_ready: {fmt_ratio(local_pos_ready)}")
    print(f"init_pos_ready: {fmt_ratio(init_pos_ready)}")
    print(f"xy_valid: {fmt_ratio(xy_valid)}  z_valid: {fmt_ratio(z_valid)}  v_xy_valid: {fmt_ratio(vxy_valid)}  v_z_valid: {fmt_ratio(vz_valid)}")
    print(f"acc_sp_from_setpoint: {fmt_ratio(acc_sp)}")
    print(f"vec_norm: {fmt_stats(vec_norm_stats)}")
    print(f"u_total_norm: {fmt_stats(u_total_stats)}")

    if state_duration:
        print("state_durations:")
        for key in sorted(state_duration.keys()):
            name = STATE_NAMES.get(key, str(key))
            print(f"  {name}: {state_duration[key]:.2f}s")


if __name__ == "__main__":
    main()

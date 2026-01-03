#!/usr/bin/env python3
"""
Summarize PVA controller logs (CSV).
Works with both pva_control and pva_feedback_control logs.
"""

import argparse
import json
from typing import Dict, Optional, Tuple

import numpy as np


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


def safe_derivative(values: np.ndarray, t: np.ndarray) -> np.ndarray:
  values = np.asarray(values, dtype=float)
  t = np.asarray(t, dtype=float)
  out = np.full_like(values, np.nan, dtype=float)
  if values.size < 2 or t.size != values.size:
    return out
  deriv = np.gradient(values, t)
  deriv[~np.isfinite(deriv)] = np.nan
  return deriv


def vec_from_prefix(data: np.ndarray, prefix: str, suffix: str = "") -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
  cols = [f"{prefix}_x{suffix}", f"{prefix}_y{suffix}", f"{prefix}_z{suffix}"]
  vals = [get_col(data, c) for c in cols]
  if any(v is None for v in vals):
    return None
  return vals[0], vals[1], vals[2]


def vector_norm(data: np.ndarray, prefix: str) -> Optional[np.ndarray]:
  vec = vec_from_prefix(data, prefix)
  if vec is None:
    return None
  stacked = np.vstack(vec).T
  return np.linalg.norm(stacked, axis=1)


def stats(vec: np.ndarray) -> Optional[Dict[str, float]]:
    vec = np.asarray(vec)
    vec = vec[np.isfinite(vec)]
    if vec.size == 0:
        return None
    return {
        "count": int(vec.size),
        "mean": float(np.mean(vec)),
        "rms": float(np.sqrt(np.mean(vec ** 2))),
        "max": float(np.max(np.abs(vec))),
        "p95": float(np.percentile(np.abs(vec), 95)),
    }


def maybe_stats(vec: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
    if vec is None:
        return None
    return stats(vec)


def axis_stats(data: np.ndarray, prefix: str) -> Optional[Dict[str, Dict[str, float]]]:
  out: Dict[str, Dict[str, float]] = {}
  for axis in ("x", "y", "z"):
        col = get_col(data, f"{prefix}_{axis}")
        if col is None:
            return None
        s = stats(col)
        if s is None:
            return None
        out[axis] = s
  return out


def jerk_from_acc(data: np.ndarray, t: Optional[np.ndarray]) -> Optional[np.ndarray]:
  if t is None:
    return None
  acc = vec_from_prefix(data, "acc")
  if acc is None:
    return None
  jx = safe_derivative(acc[0], t)
  jy = safe_derivative(acc[1], t)
  jz = safe_derivative(acc[2], t)
  stacked = np.vstack([jx, jy, jz]).T
  return np.linalg.norm(stacked, axis=1)


def jerk_from_velocity(data: np.ndarray,
                       t: Optional[np.ndarray],
                       prefix: str,
                       suffix: str = "") -> Optional[np.ndarray]:
  if t is None:
    return None
  vel = vec_from_prefix(data, prefix, suffix)
  if vel is None:
    return None
  ax = safe_derivative(vel[0], t)
  ay = safe_derivative(vel[1], t)
  az = safe_derivative(vel[2], t)
  jx = safe_derivative(ax, t)
  jy = safe_derivative(ay, t)
  jz = safe_derivative(az, t)
  stacked = np.vstack([jx, jy, jz]).T
  return np.linalg.norm(stacked, axis=1)


def jerk_from_position(data: np.ndarray,
                       t: Optional[np.ndarray],
                       prefix: str,
                       suffix: str = "") -> Optional[np.ndarray]:
  if t is None:
    return None
  pos = vec_from_prefix(data, prefix, suffix)
  if pos is None:
    return None
  vx = safe_derivative(pos[0], t)
  vy = safe_derivative(pos[1], t)
  vz = safe_derivative(pos[2], t)
  ax = safe_derivative(vx, t)
  ay = safe_derivative(vy, t)
  az = safe_derivative(vz, t)
  jx = safe_derivative(ax, t)
  jy = safe_derivative(ay, t)
  jz = safe_derivative(az, t)
  stacked = np.vstack([jx, jy, jz]).T
  return np.linalg.norm(stacked, axis=1)


def actual_jerk(data: np.ndarray, t: Optional[np.ndarray]) -> Optional[np.ndarray]:
  jerk = jerk_from_velocity(data, t, "payload_v", "_enu")
  if jerk is not None:
    return jerk
  jerk = jerk_from_velocity(data, t, "odom_v")
  if jerk is not None:
    return jerk
  jerk = jerk_from_position(data, t, "payload", "_enu")
  if jerk is not None:
    return jerk
  return jerk_from_position(data, t, "odom")


def fmt(s: Optional[Dict[str, float]]) -> str:
  if not s:
    return "missing"
  return (
        f"count={s['count']} mean={s['mean']:.4f} rms={s['rms']:.4f} "
        f"p95={s['p95']:.4f} max={s['max']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PVA controller log CSV")
    parser.add_argument("log_path", help="Path to PVA log CSV")
    parser.add_argument("--json", dest="json_path", default="",
                        help="Optional path to write JSON summary")
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

    odom_x = get_col(data, "odom_x")
    sp_x = get_col(data, "sp_x")
    pos_err = None
    if odom_x is not None and sp_x is not None:
        odom = np.vstack([get_col(data, "odom_x"),
                          get_col(data, "odom_y"),
                          get_col(data, "odom_z")]).T
        sp = np.vstack([get_col(data, "sp_x"),
                        get_col(data, "sp_y"),
                        get_col(data, "sp_z")]).T
        pos_err = np.linalg.norm(odom - sp, axis=1)

  payload_err = None
  if ("payload_x_enu" in data.dtype.names and
            "payload_des_x" in data.dtype.names):
        payload = np.vstack([get_col(data, "payload_x_enu"),
                             get_col(data, "payload_y_enu"),
                             get_col(data, "payload_z_enu")]).T
        payload_des = np.vstack([get_col(data, "payload_des_x"),
                                 get_col(data, "payload_des_y"),
                                 get_col(data, "payload_des_z")]).T
        payload_err = np.linalg.norm(payload - payload_des, axis=1)

  summary = {
    "samples": int(data.shape[0]),
    "duration_s": duration,
    "mean_dt_s": dt_mean,
    "acc_norm": maybe_stats(vector_norm(data, "acc")),
    "cable_mu": maybe_stats(get_col(data, "cable_mu")),
    "pos_err_norm": maybe_stats(pos_err),
    "payload_err_norm": maybe_stats(payload_err),
    "sp_pos_axis": axis_stats(data, "sp"),
  }

  # Feedback log extras.
  summary["mu_ff_norm"] = maybe_stats(vector_norm(data, "mu_ff"))
  summary["mu_fb_norm"] = maybe_stats(vector_norm(data, "mu_fb"))
  summary["ex_enu_norm"] = maybe_stats(vector_norm(data, "ex_enu"))
  summary["ev_enu_norm"] = maybe_stats(vector_norm(data, "ev_enu"))
  summary["jerk_cmd_norm"] = maybe_stats(jerk_from_acc(data, t))
  summary["jerk_actual_norm"] = maybe_stats(actual_jerk(data, t))

  print(f"samples: {summary['samples']}  duration: {summary['duration_s']:.3f}s  mean_dt: {summary['mean_dt_s']:.4f}s")
  print(f"acc_norm: {fmt(summary['acc_norm'])}")
  print(f"cable_mu: {fmt(summary['cable_mu'])}")
  print(f"pos_err_norm: {fmt(summary['pos_err_norm'])}")
  print(f"payload_err_norm: {fmt(summary['payload_err_norm'])}")
  print(f"mu_ff_norm: {fmt(summary['mu_ff_norm'])}")
  print(f"mu_fb_norm: {fmt(summary['mu_fb_norm'])}")
  print(f"ex_enu_norm: {fmt(summary['ex_enu_norm'])}")
  print(f"ev_enu_norm: {fmt(summary['ev_enu_norm'])}")
  print(f"jerk_cmd_norm: {fmt(summary['jerk_cmd_norm'])}")
  print(f"jerk_actual_norm: {fmt(summary['jerk_actual_norm'])}")

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

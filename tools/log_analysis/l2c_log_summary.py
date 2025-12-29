#!/usr/bin/env python3
"""
Summarize L2C controller logs (CSV).
"""

import argparse
import json
from typing import Dict, Optional

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


def vector_norm(data: np.ndarray, prefix: str) -> Optional[np.ndarray]:
    cols = [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]
    vals = [get_col(data, c) for c in cols]
    if any(v is None for v in vals):
        return None
    stacked = np.vstack(vals).T
    return np.linalg.norm(stacked, axis=1)


def quat_norm(data: np.ndarray, prefix: str) -> Optional[np.ndarray]:
    cols = [f"{prefix}_w", f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]
    vals = [get_col(data, c) for c in cols]
    if any(v is None for v in vals):
        return None
    stacked = np.vstack(vals).T
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


def fmt(s: Optional[Dict[str, float]]) -> str:
    if not s:
        return "missing"
    return (
        f"count={s['count']} mean={s['mean']:.4f} rms={s['rms']:.4f} "
        f"p95={s['p95']:.4f} max={s['max']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize L2C controller log CSV")
    parser.add_argument("log_path", help="Path to l2c log CSV")
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

    summary = {
        "samples": int(data.shape[0]),
        "duration_s": duration,
        "mean_dt_s": dt_mean,
        "ex_ned_norm": maybe_stats(vector_norm(data, "ex_ned")),
        "ev_ned_norm": maybe_stats(vector_norm(data, "ev_ned")),
        "ex_enu_norm": maybe_stats(vector_norm(data, "ex_enu")),
        "ev_enu_norm": maybe_stats(vector_norm(data, "ev_enu")),
        "eq_wxyz_norm": maybe_stats(quat_norm(data, "eq")),
        "eomega_enu_norm": maybe_stats(vector_norm(data, "eomega_enu")),
        "eq_cable_norm": maybe_stats(vector_norm(data, "eq_cable")),
        "ew_cable_norm": maybe_stats(vector_norm(data, "ew_cable")),
        "ex_ned_axis": axis_stats(data, "ex_ned"),
        "ev_ned_axis": axis_stats(data, "ev_ned"),
    }

    print(f"samples: {summary['samples']}  duration: {summary['duration_s']:.3f}s  mean_dt: {summary['mean_dt_s']:.4f}s")
    print(f"ex_ned_norm: {fmt(summary['ex_ned_norm'])}")
    print(f"ev_ned_norm: {fmt(summary['ev_ned_norm'])}")
    print(f"ex_enu_norm: {fmt(summary['ex_enu_norm'])}")
    print(f"ev_enu_norm: {fmt(summary['ev_enu_norm'])}")
    print(f"eq_wxyz_norm: {fmt(summary['eq_wxyz_norm'])}")
    print(f"eomega_enu_norm: {fmt(summary['eomega_enu_norm'])}")
    print(f"eq_cable_norm: {fmt(summary['eq_cable_norm'])}")
    print(f"ew_cable_norm: {fmt(summary['ew_cable_norm'])}")

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot PVA logs from a file or a SITL run directory.
If a directory is given, all pva_log_*_drone_*.csv and pva_feedback_log_*_drone_*.csv
in that directory are plotted together.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_log(path: Path) -> np.ndarray:
    data = np.genfromtxt(str(path), delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def list_log_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]

    log_files = sorted(path.glob("pva_log_*_drone_*.csv"))
    log_files += sorted(path.glob("pva_feedback_log_*_drone_*.csv"))
    return log_files


def find_latest_run_dir(base_dir: Path) -> Path:
    candidates = [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("exp_")]
    if not candidates:
        return base_dir
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_multi(path: Path) -> List[Tuple[str, np.ndarray]]:
    if path.is_dir():
        log_dir = path
        log_files = list_log_files(log_dir)
        if not log_files:
            log_dir = find_latest_run_dir(path)
            log_files = list_log_files(log_dir)
        if not log_files:
            raise RuntimeError(f"No PVA logs found in {path}")
    else:
        log_files = [path]

    out = []
    for f in log_files:
        out.append((f.name, load_log(f)))
    return out


def vec_norm(data: np.ndarray, prefix: str) -> np.ndarray:
    return np.linalg.norm(
        np.vstack([data[f"{prefix}_x"], data[f"{prefix}_y"], data[f"{prefix}_z"]]).T,
        axis=1,
    )


def safe_derivative(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.full_like(values, np.nan, dtype=float)
    if values.size < 2 or t.size != values.size:
        return out
    deriv = np.gradient(values, t)
    deriv[~np.isfinite(deriv)] = np.nan
    return deriv


def vec_from_prefix(data: np.ndarray, prefix: str, suffix: str = "") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        data[f"{prefix}_x{suffix}"],
        data[f"{prefix}_y{suffix}"],
        data[f"{prefix}_z{suffix}"],
    )


def jerk_from_acc(data: np.ndarray) -> np.ndarray:
    t = data["t"]
    ax, ay, az = vec_from_prefix(data, "acc")
    jx = safe_derivative(ax, t)
    jy = safe_derivative(ay, t)
    jz = safe_derivative(az, t)
    return np.linalg.norm(np.vstack([jx, jy, jz]).T, axis=1)


def jerk_from_velocity(data: np.ndarray, prefix: str, suffix: str = "") -> np.ndarray:
    t = data["t"]
    vx, vy, vz = vec_from_prefix(data, prefix, suffix)
    ax = safe_derivative(vx, t)
    ay = safe_derivative(vy, t)
    az = safe_derivative(vz, t)
    jx = safe_derivative(ax, t)
    jy = safe_derivative(ay, t)
    jz = safe_derivative(az, t)
    return np.linalg.norm(np.vstack([jx, jy, jz]).T, axis=1)


def jerk_from_position(data: np.ndarray, prefix: str, suffix: str = "") -> np.ndarray:
    t = data["t"]
    px, py, pz = vec_from_prefix(data, prefix, suffix)
    vx = safe_derivative(px, t)
    vy = safe_derivative(py, t)
    vz = safe_derivative(pz, t)
    ax = safe_derivative(vx, t)
    ay = safe_derivative(vy, t)
    az = safe_derivative(vz, t)
    jx = safe_derivative(ax, t)
    jy = safe_derivative(ay, t)
    jz = safe_derivative(az, t)
    return np.linalg.norm(np.vstack([jx, jy, jz]).T, axis=1)


def actual_jerk(data: np.ndarray) -> np.ndarray:
    names = data.dtype.names
    if "payload_vx_enu" in names:
        return jerk_from_velocity(data, "payload_v", "_enu")
    if "odom_vx" in names:
        return jerk_from_velocity(data, "odom_v")
    if "payload_x_enu" in names:
        return jerk_from_position(data, "payload", "_enu")
    return jerk_from_position(data, "odom")


def plot_position(data_list: List[Tuple[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].set_title("Position: setpoint/desired (dashed) vs measured (solid)")
    for name, data in data_list:
        t = data["t"]
        if "odom_x" in data.dtype.names:
            sp_prefix = "sp"
            meas_prefix = "odom"
            meas_suffix = ""
        elif "payload_x_enu" in data.dtype.names:
            sp_prefix = "payload_des"
            meas_prefix = "payload"
            meas_suffix = "_enu"
        else:
            continue
        for ax, axis in zip(axes, ["x", "y", "z"]):
            ax.plot(t, data[f"{sp_prefix}_{axis}"], "--", label=f"{name} {sp_prefix}_{axis}")
            ax.plot(t, data[f"{meas_prefix}_{axis}{meas_suffix}"], "-", label=f"{name} {meas_prefix}_{axis}")
            ax.set_ylabel(axis)
    axes[-1].set_xlabel("time [s]")
    for ax in axes:
        ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()


def plot_acc(data_list: List[Tuple[str, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Acceleration magnitude")
    for name, data in data_list:
        t = data["t"]
        acc_norm = vec_norm(data, "acc")
        ax.plot(t, acc_norm, label=name)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("||acc|| [m/s^2]")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()


def plot_feedback(data_list: List[Tuple[str, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Cable force norms (mu_ff, mu_fb)")
    has_any = False
    for name, data in data_list:
        if "mu_ff_x" in data.dtype.names:
            t = data["t"]
            mu_ff = vec_norm(data, "mu_ff")
            ax.plot(t, mu_ff, label=f"{name} mu_ff")
            has_any = True
        if "mu_fb_x" in data.dtype.names:
            t = data["t"]
            mu_fb = vec_norm(data, "mu_fb")
            ax.plot(t, mu_fb, label=f"{name} mu_fb")
            has_any = True
    if not has_any:
        plt.close(fig)
        return
    ax.set_xlabel("time [s]")
    ax.set_ylabel("||mu|| [N]")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()


def plot_jerk(data_list: List[Tuple[str, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Jerk magnitude (commanded vs actual)")
    has_any = False
    for name, data in data_list:
        if "acc_x" in data.dtype.names:
            t = data["t"]
            jerk_cmd = jerk_from_acc(data)
            ax.plot(t, jerk_cmd, label=f"{name} jerk_cmd")
            has_any = True
        if "payload_x_enu" in data.dtype.names or "odom_x" in data.dtype.names:
            t = data["t"]
            jerk_act = actual_jerk(data)
            ax.plot(t, jerk_act, label=f"{name} jerk_actual")
            has_any = True
    if not has_any:
        plt.close(fig)
        return
    ax.set_xlabel("time [s]")
    ax.set_ylabel("||jerk|| [m/s^3]")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PVA logs (single or multi-drone)")
    parser.add_argument("path", help="CSV log file or base log directory")
    args = parser.parse_args()

    data_list = load_multi(Path(args.path))
    plot_position(data_list)
    plot_acc(data_list)
    plot_feedback(data_list)
    plot_jerk(data_list)
    plt.show()


if __name__ == "__main__":
    main()

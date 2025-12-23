#!/usr/bin/env python3
"""
Plot all CSV outputs produced by tools/preprocess_traj_new.py.
"""

import argparse
import re
import signal
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def signal_handler(sig, frame):
    print("\nExiting...")
    plt.close("all")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def read_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    config = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.match(r"^([A-Za-z0-9_]+)\s*:\s*(.+)$", line)
        if match:
            key = match.group(1)
            value = match.group(2).strip().strip("\"'")
            config[key] = value
    return config


def scenario_suffix(name: str) -> str:
    marker = "COM_Dyn"
    if marker not in name:
        return ""
    return name.split(marker, 1)[1]


def default_output_dir(repo_root: Path) -> Path:
    config_path = repo_root / "tools" / "preprocess_traj_new.yaml"
    config = read_config(config_path)
    base_dir = config.get("base_dir")
    if base_dir:
        suffix = scenario_suffix(Path(base_dir).name)
        return Path(f"data/realflight_traj_new{suffix}")
    return Path("data/realflight_traj_new")


def resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root / path


def set_equal_aspect_3d(ax, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray):
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    z_min, z_max = np.min(zs), np.max(zs)
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    half = max_range / 2.0
    x_mid = (x_max + x_min) / 2.0
    y_mid = (y_max + y_min) / 2.0
    z_mid = (z_max + z_min) / 2.0
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)


def plot_components(ax, time, df, cols, title, ylabel):
    labels = []
    for col in cols:
        if col in df.columns:
            ax.plot(time, df[col].to_numpy())
            labels.append(col)
    if labels:
        ax.legend(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True)


def plot_payload(payload_path: Path):
    df = pd.read_csv(payload_path)
    time = df["time"].to_numpy()

    fig1, axs1 = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    plot_components(axs1[0], time, df, ["x", "y", "z"], "Payload position", "m")
    plot_components(axs1[1], time, df, ["vx", "vy", "vz"], "Payload velocity", "m/s")
    plot_components(axs1[2], time, df, ["ax", "ay", "az"], "Payload acceleration", "m/s^2")
    plot_components(axs1[3], time, df, ["jx", "jy", "jz"], "Payload jerk", "m/s^3")
    plot_components(axs1[4], time, df, ["sx", "sy", "sz"], "Payload snap", "m/s^4")
    axs1[-1].set_xlabel("time [s]")

    fig2, axs2 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    plot_components(axs2[0], time, df, ["qw", "qx", "qy", "qz"], "Payload quaternion", "unit")
    if {"qw", "qx", "qy", "qz"}.issubset(df.columns):
        q_norm = np.linalg.norm(df[["qw", "qx", "qy", "qz"]].to_numpy(), axis=1)
        axs2[0].plot(time, q_norm, linestyle="--")
        axs2[0].legend(["qw", "qx", "qy", "qz", "q_norm"])
    plot_components(axs2[1], time, df, ["wx", "wy", "wz"], "Payload omega", "rad/s")
    axs2[-1].set_xlabel("time [s]")


def plot_drone(drone_path: Path, drone_id: int):
    df = pd.read_csv(drone_path)
    time = df["time"].to_numpy()

    fig1, axs1 = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    plot_components(axs1[0], time, df, ["x", "y", "z"], f"Drone {drone_id} position", "m")
    plot_components(axs1[1], time, df, ["vx", "vy", "vz"], f"Drone {drone_id} velocity", "m/s")
    plot_components(axs1[2], time, df, ["ax", "ay", "az"], f"Drone {drone_id} acceleration", "m/s^2")
    plot_components(axs1[3], time, df, ["jx", "jy", "jz"], f"Drone {drone_id} jerk", "m/s^3")
    plot_components(axs1[4], time, df, ["sx", "sy", "sz"], f"Drone {drone_id} snap", "m/s^4")
    axs1[-1].set_xlabel("time [s]")

    fig2, axs2 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    plot_components(axs2[0], time, df, ["roll", "pitch", "yaw"], f"Drone {drone_id} attitude", "rad")
    plot_components(axs2[1], time, df, ["p", "q", "r"], f"Drone {drone_id} body rates", "rad/s")
    axs2[-1].set_xlabel("time [s]")


def plot_cable(cable_path: Path, cable_id: int):
    df = pd.read_csv(cable_path)
    time = df["time"].to_numpy()

    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    plot_components(axs[0], time, df, ["dir_x", "dir_y", "dir_z"], f"Cable {cable_id} direction", "unit")
    plot_components(axs[1], time, df, ["omega_x", "omega_y", "omega_z"], f"Cable {cable_id} omega", "rad/s")
    plot_components(axs[2], time, df, ["omega_dot_x", "omega_dot_y", "omega_dot_z"], f"Cable {cable_id} omega_dot", "rad/s^2")
    plot_components(axs[3], time, df, ["mu"], f"Cable {cable_id} mu", "N")
    axs[-1].set_xlabel("time [s]")


def plot_kfb(kfb_path: Path):
    df = pd.read_csv(kfb_path)
    time = df["time"].to_numpy()
    data = df.drop(columns=["time"]).to_numpy().T

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=[time[0], time[-1], 0, data.shape[0] - 1],
        cmap="coolwarm",
    )
    ax.set_title("Kfb coefficients (rows: k index)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("coefficient index")
    fig.colorbar(im, ax=ax, label="value")


def plot_paths_3d(payload_path: Path, drone_paths: list[tuple[int, Path]]):
    payload = pd.read_csv(payload_path)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(payload["x"].to_numpy(), payload["y"].to_numpy(), payload["z"].to_numpy(), label="payload")

    xs, ys, zs = [payload["x"].to_numpy()], [payload["y"].to_numpy()], [payload["z"].to_numpy()]
    for drone_id, path in drone_paths:
        df = pd.read_csv(path)
        ax.plot(df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy(), label=f"drone {drone_id}")
        xs.append(df["x"].to_numpy())
        ys.append(df["y"].to_numpy())
        zs.append(df["z"].to_numpy())

    ax.set_title("3D trajectories (NED)")
    ax.set_xlabel("North [m]")
    ax.set_ylabel("East [m]")
    ax.set_zlabel("Down [m]")
    set_equal_aspect_3d(ax, np.concatenate(xs), np.concatenate(ys), np.concatenate(zs))
    ax.legend()
    ax.grid(True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Plot all CSV outputs from tools/preprocess_traj_new.py"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=default_output_dir(repo_root),
        help="Directory containing payload.csv, cable_*.csv, kfb.csv, drone_*_traj_smoothed_100hz.csv",
    )
    parser.add_argument(
        "--no-3d",
        action="store_true",
        help="Skip 3D path plot",
    )
    args = parser.parse_args()

    base_dir = resolve_path(args.base_dir, repo_root)
    if not base_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {base_dir}")

    payload_path = base_dir / "payload.csv"
    if payload_path.exists():
        plot_payload(payload_path)
    else:
        print(f"Skipping payload.csv (missing): {payload_path}")

    drone_paths = []
    for path in sorted(base_dir.glob("drone_*_traj_smoothed_100hz.csv")):
        match = re.match(r"drone_(\d+)_traj_smoothed_100hz\.csv", path.name)
        if match:
            drone_id = int(match.group(1))
            drone_paths.append((drone_id, path))
            plot_drone(path, drone_id)

    for path in sorted(base_dir.glob("cable_*.csv")):
        match = re.match(r"cable_(\d+)\.csv", path.name)
        if match:
            cable_id = int(match.group(1))
            plot_cable(path, cable_id)

    kfb_path = base_dir / "kfb.csv"
    if kfb_path.exists():
        plot_kfb(kfb_path)
    else:
        print(f"Skipping kfb.csv (missing): {kfb_path}")

    if payload_path.exists() and drone_paths and not args.no_3d:
        plot_paths_3d(payload_path, drone_paths)

    print("Displaying plots. Press Ctrl+C to exit or close all plot windows.")
    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    main()

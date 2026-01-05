#!/usr/bin/env python3
"""
Plot all CSV outputs produced by tools/preprocess_traj_new.py.
"""

import argparse
import re
import signal
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d import proj3d  # noqa: F401

PAYLOAD_RADIUS_M = 0.13
PAYLOAD_RING_POINTS = 80


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


def enu_to_ned(data: np.ndarray) -> np.ndarray:
    transform = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    flat = data.reshape(-1, 3)
    return (transform @ flat.T).T.reshape(data.shape)


def data_radius_to_points(ax, radius_m: float) -> float:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x_mid = 0.5 * (x0 + x1)
    y_mid = 0.5 * (y0 + y1)
    p0 = ax.transData.transform((x_mid, y_mid))
    p1 = ax.transData.transform((x_mid + radius_m, y_mid))
    pixel_radius = np.hypot(*(p1 - p0))
    return pixel_radius * 72.0 / ax.figure.dpi


def data_radius_to_points_3d(ax, radius_m: float) -> float:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    z0, z1 = ax.get_zlim()
    x_mid = 0.5 * (x0 + x1)
    y_mid = 0.5 * (y0 + y1)
    z_mid = 0.5 * (z0 + z1)
    x_proj, y_proj, _ = proj3d.proj_transform(x_mid, y_mid, z_mid, ax.get_proj())
    x_proj_r, y_proj_r, _ = proj3d.proj_transform(
        x_mid + radius_m,
        y_mid,
        z_mid,
        ax.get_proj(),
    )
    p0 = ax.transData.transform((x_proj, y_proj))
    p1 = ax.transData.transform((x_proj_r, y_proj_r))
    pixel_radius = np.hypot(*(p1 - p0))
    return pixel_radius * 72.0 / ax.figure.dpi


def circle_xy(center: np.ndarray, radius: float, samples: int) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, samples)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return x, y


def plot_components(ax, time, df, cols, title, ylabel, compare=None, compare_label="diff"):
    for col in cols:
        if col in df.columns:
            ax.plot(time, df[col].to_numpy(), label=col)
        if compare is not None:
            time_c, df_c = compare
            if col in df_c.columns:
                ax.plot(time_c, df_c[col].to_numpy(), linestyle="--", label=f"{col}_{compare_label}")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(handles, labels)
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


def plot_drone(drone_path: Path, drone_id: int, diff_path: Optional[Path] = None):
    df = pd.read_csv(drone_path)
    time = df["time"].to_numpy()
    compare = None
    if diff_path and diff_path.exists():
        df_diff = pd.read_csv(diff_path)
        if "time" in df_diff.columns:
            time_diff = df_diff["time"].to_numpy()
        else:
            time_diff = time
        compare = (time_diff, df_diff)

    fig1, axs1 = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    plot_components(axs1[0], time, df, ["x", "y", "z"], f"Drone {drone_id} position", "m", compare=compare)
    plot_components(axs1[1], time, df, ["vx", "vy", "vz"], f"Drone {drone_id} velocity", "m/s", compare=compare)
    plot_components(axs1[2], time, df, ["ax", "ay", "az"], f"Drone {drone_id} acceleration", "m/s^2", compare=compare)
    plot_components(axs1[3], time, df, ["jx", "jy", "jz"], f"Drone {drone_id} jerk", "m/s^3", compare=compare)
    plot_components(axs1[4], time, df, ["sx", "sy", "sz"], f"Drone {drone_id} snap", "m/s^4", compare=compare)
    axs1[-1].set_xlabel("time [s]")

    fig2, axs2 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    plot_components(axs2[0], time, df, ["roll", "pitch", "yaw"], f"Drone {drone_id} attitude", "rad", compare=compare)
    plot_components(axs2[1], time, df, ["p", "q", "r"], f"Drone {drone_id} body rates", "rad/s", compare=compare)
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
    kfb_flat = df.drop(columns=["time"]).to_numpy()
    if kfb_flat.shape[1] % 6 != 0:
        raise ValueError("Unexpected Kfb shape; expected 6 x N columns.")
    num_rows = 6
    num_cols = kfb_flat.shape[1] // num_rows
    kfb = kfb_flat.reshape(len(time), num_rows, num_cols)
    coeff_idx = np.arange(kfb_flat.shape[1])

    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(bottom=0.18, wspace=0.25)
    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")

    vmin = float(np.min(kfb))
    vmax = float(np.max(kfb))
    img = ax2d.imshow(
        kfb[0],
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
    )
    ax2d.set_title(f"Kfb at t={time[0]:.2f} s")
    ax2d.set_xlabel("column")
    ax2d.set_ylabel("row")
    fig.colorbar(img, ax=ax2d, fraction=0.046, pad=0.04, label="value")

    time_grid = time[:, None, None]
    row_grid = np.arange(num_rows)[None, :, None]
    col_grid = np.arange(num_cols)[None, None, :]
    time_flat = np.broadcast_to(time_grid, kfb.shape).ravel()
    row_flat = np.broadcast_to(row_grid, kfb.shape).ravel()
    col_flat = np.broadcast_to(col_grid, kfb.shape).ravel()
    val_flat = kfb.ravel()

    scatter = ax3d.scatter(
        time_flat,
        row_flat,
        col_flat,
        c=val_flat,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        s=6,
        alpha=0.8,
    )
    slice_scatter = ax3d.scatter(
        np.full(num_rows * num_cols, time[0], dtype=float),
        np.repeat(np.arange(num_rows), num_cols),
        np.tile(np.arange(num_cols), num_rows),
        c=kfb[0].ravel(),
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        s=18,
        edgecolors="black",
        linewidths=0.3,
    )
    ax3d.set_title("Kfb over time (6 x N x time)")
    ax3d.set_xlabel("time [s]")
    ax3d.set_ylabel("row")
    ax3d.set_zlabel("column")
    fig.colorbar(scatter, ax=ax3d, shrink=0.6, pad=0.1, label="value")

    ax_slider = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    slider = Slider(
        ax=ax_slider,
        label="time [s]",
        valmin=float(time[0]),
        valmax=float(time[-1]),
        valinit=float(time[0]),
        valfmt="%.2f",
    )
    time_text = fig.text(0.5, 0.02, f"t = {time[0]:.2f} s", ha="center", va="center")

    def update_time(val):
        t = float(slider.val)
        idx = int(np.searchsorted(time, t))
        if idx >= len(time):
            idx = len(time) - 1
        img.set_data(kfb[idx])
        slice_scatter._offsets3d = (
            np.full(num_rows * num_cols, time[idx], dtype=float),
            np.repeat(np.arange(num_rows), num_cols),
            np.tile(np.arange(num_cols), num_rows),
        )
        slice_scatter.set_array(kfb[idx].ravel())
        ax2d.set_title(f"Kfb at t={time[idx]:.2f} s")
        time_text.set_text(f"t = {time[idx]:.2f} s")
        fig.canvas.draw_idle()

    slider.on_changed(update_time)
    update_time(time[0])
    fig._kfb_slider = slider
    fig._kfb_update = update_time


def plot_paths_3d(payload_path: Path, drone_paths: list[tuple[int, Path]]):
    payload = pd.read_csv(payload_path)
    payload_time = payload["time"].to_numpy()
    payload_xyz_enu = payload[["x", "y", "z"]].to_numpy()
    payload_xyz_ned = enu_to_ned(payload_xyz_enu)

    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(bottom=0.18, wspace=0.25)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_xy = fig.add_subplot(1, 2, 2)

    xs = [payload_xyz_ned[:, 0]]
    ys = [payload_xyz_ned[:, 1]]
    zs = [payload_xyz_ned[:, 2]]

    ax3d.plot(
        payload_xyz_ned[:, 0],
        payload_xyz_ned[:, 1],
        payload_xyz_ned[:, 2],
        label="payload",
        color="black",
        linewidth=2,
    )
    ax_xy.plot(
        payload_xyz_ned[:, 0],
        payload_xyz_ned[:, 1],
        label="payload",
        color="black",
        linewidth=2,
    )
    ring_x, ring_y = circle_xy(payload_xyz_ned[0], PAYLOAD_RADIUS_M, PAYLOAD_RING_POINTS)
    ring_z = np.full_like(ring_x, payload_xyz_ned[0, 2])
    payload_ring3d, = ax3d.plot(
        ring_x,
        ring_y,
        ring_z,
        color="black",
        linestyle="--",
        linewidth=1,
    )
    payload_ring2d, = ax_xy.plot(
        ring_x,
        ring_y,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    marker_entries = []
    payload_dot3d, = ax3d.plot(
        [np.nan],
        [np.nan],
        [np.nan],
        marker="o",
        markersize=1,
        linestyle="None",
        color="black",
    )
    payload_dot2d, = ax_xy.plot(
        [np.nan],
        [np.nan],
        marker="o",
        markersize=1,
        linestyle="None",
        color="black",
    )
    marker_entries.append(
        {
            "time": payload_time,
            "x": payload_xyz_ned[:, 0],
            "y": payload_xyz_ned[:, 1],
            "z": payload_xyz_ned[:, 2],
            "t_min": payload_time[0],
            "t_max": payload_time[-1],
            "dot3d": payload_dot3d,
            "dot2d": payload_dot2d,
        }
    )

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(drone_paths), 1)))
    for (drone_id, path), color in zip(drone_paths, colors):
        df = pd.read_csv(path)
        time = df["time"].to_numpy()
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        z = df["z"].to_numpy()

        ax3d.plot(x, y, z, label=f"drone {drone_id}", color=color)
        ax_xy.plot(x, y, label=f"drone {drone_id}", color=color)
        xs.append(x)
        ys.append(y)
        zs.append(z)

        dot3d, = ax3d.plot(
            [np.nan],
            [np.nan],
            [np.nan],
            marker="o",
            markersize=1,
            linestyle="None",
            color=color,
        )
        dot2d, = ax_xy.plot(
            [np.nan],
            [np.nan],
            marker="o",
            markersize=1,
            linestyle="None",
            color=color,
        )
        marker_entries.append(
            {
                "time": time,
                "x": x,
                "y": y,
                "z": z,
                "t_min": time[0],
                "t_max": time[-1],
                "dot3d": dot3d,
                "dot2d": dot2d,
            }
        )

    ax3d.set_title("3D trajectories (NED)")
    ax3d.set_xlabel("North [m]")
    ax3d.set_ylabel("East [m]")
    ax3d.set_zlabel("Down [m]")
    set_equal_aspect_3d(ax3d, np.concatenate(xs), np.concatenate(ys), np.concatenate(zs))
    ax3d.legend()
    ax3d.grid(True)

    ax_xy.set_title("XY projection (NED)")
    ax_xy.set_xlabel("North [m]")
    ax_xy.set_ylabel("East [m]")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.legend(ncol=2, fontsize="small")
    ax_xy.grid(True)

    fig.canvas.draw()
    marker_radius_m = 0.125
    marker_size_2d = 2.0 * data_radius_to_points(ax_xy, marker_radius_m)
    marker_size_3d = 2.0 * data_radius_to_points_3d(ax3d, marker_radius_m)
    for entry in marker_entries:
        entry["dot2d"].set_markersize(marker_size_2d)
        entry["dot3d"].set_markersize(marker_size_3d)

    all_times = np.concatenate([entry["time"] for entry in marker_entries])
    t_min = float(np.min(all_times))
    t_max = float(np.max(all_times))

    ax_slider = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    slider = Slider(
        ax=ax_slider,
        label="time [s]",
        valmin=t_min,
        valmax=t_max,
        valinit=t_min,
        valfmt="%.2f",
    )
    time_text = fig.text(0.5, 0.02, f"t = {t_min:.2f} s", ha="center", va="center")

    def update_time(val):
        t = slider.val
        for entry in marker_entries:
            if t < entry["t_min"] or t > entry["t_max"]:
                x = y = z = np.nan
            else:
                x = float(np.interp(t, entry["time"], entry["x"]))
                y = float(np.interp(t, entry["time"], entry["y"]))
                z = float(np.interp(t, entry["time"], entry["z"]))
            entry["dot3d"].set_data_3d([x], [y], [z])
            entry["dot2d"].set_data([x], [y])
        if t < payload_time[0] or t > payload_time[-1]:
            ring_x = ring_y = ring_z = np.full(PAYLOAD_RING_POINTS, np.nan)
        else:
            px = float(np.interp(t, payload_time, payload_xyz_ned[:, 0]))
            py = float(np.interp(t, payload_time, payload_xyz_ned[:, 1]))
            pz = float(np.interp(t, payload_time, payload_xyz_ned[:, 2]))
            ring_x, ring_y = circle_xy(np.array([px, py, pz]), PAYLOAD_RADIUS_M, PAYLOAD_RING_POINTS)
            ring_z = np.full_like(ring_x, pz)
        payload_ring3d.set_data_3d(ring_x, ring_y, ring_z)
        payload_ring2d.set_data(ring_x, ring_y)
        time_text.set_text(f"t = {t:.2f} s")
        fig.canvas.draw_idle()

    slider.on_changed(update_time)
    update_time(t_min)
    fig._paths_slider = slider
    fig._paths_update = update_time


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
            diff_path = path.with_name(
                path.name.replace("_traj_smoothed_100hz.csv", "_traj_smoothed_100hz_diff.csv")
            )
            plot_drone(path, drone_id, diff_path=diff_path)

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

#!/usr/bin/env python3
"""
Compare logged acceleration vectors against planned cable direction/mu in 3D.
Generates per-drone 3D plots (ff_acc, acc_cmd, acc_dir) plus a multi-drone 3D plot.
Planned cable data is assumed to be ENU in cable_*.csv.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider
import numpy as np

_SLIDERS: List[Slider] = []


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


def parse_drone_id(name: str) -> Optional[int]:
    match = re.search(r"drone_(\d+)", name)
    if not match:
        return None
    return int(match.group(1))


def ned_to_enu(vec: np.ndarray) -> np.ndarray:
    return np.array([vec[1], vec[0], -vec[2]], dtype=float)


def ned_to_enu_series(series: np.ndarray) -> np.ndarray:
    return np.vstack([ned_to_enu(v) for v in series])


def normalize_series(series: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(series, axis=1)
    out = np.zeros_like(series)
    valid = norms > 1e-9
    out[valid] = series[valid] / norms[valid, None]
    return out


def interpolate_series(time_src: np.ndarray,
                       series_src: np.ndarray,
                       time_target: np.ndarray) -> np.ndarray:
    out = np.zeros((len(time_target), 3), dtype=float)
    for axis in range(3):
        out[:, axis] = np.interp(time_target, time_src, series_src[:, axis],
                                 left=series_src[0, axis], right=series_src[-1, axis])
    return out


def load_planned_cable(planned_dir: Path, drone_id: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    cable_path = planned_dir / f"cable_{drone_id}.csv"
    if not cable_path.exists():
        return None
    data = np.genfromtxt(str(cable_path), delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    if "time" not in data.dtype.names:
        return None
    if not all(k in data.dtype.names for k in ("dir_x", "dir_y", "dir_z", "mu")):
        return None
    t = data["time"].astype(float)
    dirs = np.vstack([data["dir_x"], data["dir_y"], data["dir_z"]]).T.astype(float)
    mu = data["mu"].astype(float)
    return t, dirs, mu


def extract_log_vectors(data: np.ndarray) -> Dict[str, np.ndarray]:
    names = data.dtype.names
    if "traj_t" in names:
        t = data["traj_t"].astype(float)
    else:
        t = data["t"].astype(float)
    t_rel = t - t[0]

    acc_ned = np.vstack([data["acc_x"], data["acc_y"], data["acc_z"]]).T.astype(float)
    ff_ned = None
    if "ff_acc_x" in names:
        ff_ned = np.vstack([data["ff_acc_x"], data["ff_acc_y"], data["ff_acc_z"]]).T.astype(float)

    if "acc_dir_x" in names:
        acc_dir_ned = np.vstack([data["acc_dir_x"], data["acc_dir_y"], data["acc_dir_z"]]).T.astype(float)
    else:
        acc_dir_ned = normalize_series(acc_ned)

    acc_enu = ned_to_enu_series(acc_ned)
    ff_enu = ned_to_enu_series(ff_ned) if ff_ned is not None else None
    acc_dir_enu = ned_to_enu_series(acc_dir_ned)

    return {
        "t_rel": t_rel,
        "acc_enu": acc_enu,
        "ff_enu": ff_enu,
        "acc_dir_enu": acc_dir_enu,
    }


def plot_drone_3d(name: str,
                  t_rel: np.ndarray,
                  acc_enu: np.ndarray,
                  acc_dir_enu: np.ndarray,
                  ff_enu: Optional[np.ndarray],
                  ff_plan_enu: Optional[np.ndarray]) -> None:
    max_mag = float(np.max(np.linalg.norm(acc_enu, axis=1)))
    if ff_enu is not None:
        max_mag = max(max_mag, float(np.max(np.linalg.norm(ff_enu, axis=1))))
    if ff_plan_enu is not None:
        max_mag = max(max_mag, float(np.max(np.linalg.norm(ff_plan_enu, axis=1))))
    axis_lim = max(1.0, max_mag * 1.1)
    dir_scale = max(1.0, axis_lim * 0.5)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(bottom=0.2)
    ax.set_title(f"{name} Acc vectors (ENU)")
    ax.set_xlabel("E")
    ax.set_ylabel("N")
    ax.set_zlabel("U")
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_zlim(-axis_lim, axis_lim)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass

    idx0 = 0
    vec_acc = acc_enu[idx0]
    line_acc, = ax.plot(np.array([0.0, vec_acc[0]]),
                        np.array([0.0, vec_acc[1]]),
                        np.array([0.0, vec_acc[2]]),
                        color="tab:green", marker="^", label="acc_cmd")

    vec_dir = acc_dir_enu[idx0] * dir_scale
    line_dir, = ax.plot(np.array([0.0, vec_dir[0]]),
                        np.array([0.0, vec_dir[1]]),
                        np.array([0.0, vec_dir[2]]),
                        color="tab:orange", marker="s", label="acc_dir")

    line_ff = None
    if ff_enu is not None:
        vec_ff = ff_enu[idx0]
        line_ff, = ax.plot(np.array([0.0, vec_ff[0]]),
                           np.array([0.0, vec_ff[1]]),
                           np.array([0.0, vec_ff[2]]),
                           color="tab:blue", marker="o", label="ff_acc")

    line_plan = None
    if ff_plan_enu is not None:
        vec_plan = ff_plan_enu[idx0]
        line_plan, = ax.plot(np.array([0.0, vec_plan[0]]),
                             np.array([0.0, vec_plan[1]]),
                             np.array([0.0, vec_plan[2]]),
                             color="tab:blue", linestyle="--", marker="x", label="ff_plan")

    ax.legend(ncol=2, fontsize=8)

    slider_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
    time_slider = Slider(slider_ax, "t", 0.0, float(t_rel[-1]), valinit=0.0)
    _SLIDERS.append(time_slider)

    def update(val: float) -> None:
        idx = int(np.searchsorted(t_rel, val, side="left"))
        if idx >= len(t_rel):
            idx = len(t_rel) - 1

        vec_acc_u = acc_enu[idx]
        line_acc.set_data(np.array([0.0, vec_acc_u[0]]), np.array([0.0, vec_acc_u[1]]))
        line_acc.set_3d_properties(np.array([0.0, vec_acc_u[2]]))

        vec_dir_u = acc_dir_enu[idx] * dir_scale
        line_dir.set_data(np.array([0.0, vec_dir_u[0]]), np.array([0.0, vec_dir_u[1]]))
        line_dir.set_3d_properties(np.array([0.0, vec_dir_u[2]]))

        parts = [f"{name} t={t_rel[idx]:.2f}s"]
        if line_ff is not None and ff_enu is not None:
            vec_ff_u = ff_enu[idx]
            line_ff.set_data(np.array([0.0, vec_ff_u[0]]), np.array([0.0, vec_ff_u[1]]))
            line_ff.set_3d_properties(np.array([0.0, vec_ff_u[2]]))
            parts.append(f"|ff|={np.linalg.norm(vec_ff_u):.3f}")
        if line_plan is not None and ff_plan_enu is not None:
            vec_plan_u = ff_plan_enu[idx]
            line_plan.set_data(np.array([0.0, vec_plan_u[0]]), np.array([0.0, vec_plan_u[1]]))
            line_plan.set_3d_properties(np.array([0.0, vec_plan_u[2]]))
            parts.append(f"|plan|={np.linalg.norm(vec_plan_u):.3f}")

        ax.set_title(" | ".join(parts))
        fig.canvas.draw_idle()

    time_slider.on_changed(update)


def plot_multi_3d(entries: List[Dict[str, np.ndarray]]) -> None:
    if len(entries) < 2:
        return
    colors = ["tab:blue", "tab:orange", "tab:green"]
    markers = {"ff": "o", "acc": "^", "dir": "s", "plan": "x"}

    max_mag = 0.0
    for entry in entries:
        acc = entry["acc_enu"]
        max_mag = max(max_mag, float(np.max(np.linalg.norm(acc, axis=1))))
        ff = entry.get("ff_enu")
        if ff is not None:
            max_mag = max(max_mag, float(np.max(np.linalg.norm(ff, axis=1))))
        ff_plan = entry.get("ff_plan_enu")
        if ff_plan is not None:
            max_mag = max(max_mag, float(np.max(np.linalg.norm(ff_plan, axis=1))))

    axis_lim = max(1.0, max_mag * 1.1)
    dir_scale = max(1.0, axis_lim * 0.5)
    t_max = max(float(entry["t_rel"][-1]) for entry in entries)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(bottom=0.2)
    ax.set_title("Acc vectors (ENU) - multi-drone")
    ax.set_xlabel("E")
    ax.set_ylabel("N")
    ax.set_zlabel("U")
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_zlim(-axis_lim, axis_lim)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass

    lines: List[Tuple[str, int, plt.Line2D]] = []
    legend_items: List[Line2D] = []

    for idx, entry in enumerate(entries):
        color = colors[idx % len(colors)]
        acc = entry["acc_enu"]
        acc_dir = entry["acc_dir_enu"] * dir_scale
        ff = entry.get("ff_enu")
        ff_plan = entry.get("ff_plan_enu")
        label_prefix = entry["name"]

        line_acc, = ax.plot(np.array([0.0, acc[0, 0]]),
                            np.array([0.0, acc[0, 1]]),
                            np.array([0.0, acc[0, 2]]),
                            color=color, marker=markers["acc"], label=f"{label_prefix} acc_cmd")
        lines.append(("acc", idx, line_acc))

        line_dir, = ax.plot(np.array([0.0, acc_dir[0, 0]]),
                            np.array([0.0, acc_dir[0, 1]]),
                            np.array([0.0, acc_dir[0, 2]]),
                            color=color, marker=markers["dir"], label=f"{label_prefix} acc_dir")
        lines.append(("dir", idx, line_dir))

        if ff is not None:
            line_ff, = ax.plot(np.array([0.0, ff[0, 0]]),
                               np.array([0.0, ff[0, 1]]),
                               np.array([0.0, ff[0, 2]]),
                               color=color, marker=markers["ff"], label=f"{label_prefix} ff_acc")
            lines.append(("ff", idx, line_ff))

        if ff_plan is not None:
            line_plan, = ax.plot(np.array([0.0, ff_plan[0, 0]]),
                                 np.array([0.0, ff_plan[0, 1]]),
                                 np.array([0.0, ff_plan[0, 2]]),
                                 color=color, linestyle="--", marker=markers["plan"],
                                 label=f"{label_prefix} ff_plan")
            lines.append(("plan", idx, line_plan))

    legend_items.append(Line2D([0], [0], color="black", marker=markers["acc"], lw=0, label="acc_cmd"))
    legend_items.append(Line2D([0], [0], color="black", marker=markers["dir"], lw=0, label="acc_dir"))
    legend_items.append(Line2D([0], [0], color="black", marker=markers["ff"], lw=0, label="ff_acc"))
    legend_items.append(Line2D([0], [0], color="black", marker=markers["plan"], lw=0, label="ff_plan"))
    ax.legend(handles=legend_items, ncol=2, fontsize=8)

    slider_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
    time_slider = Slider(slider_ax, "t", 0.0, t_max, valinit=0.0)
    _SLIDERS.append(time_slider)

    def update(val: float) -> None:
        for kind, idx, line in lines:
            entry = entries[idx]
            t_rel = entry["t_rel"]
            i = int(np.searchsorted(t_rel, val, side="left"))
            if i >= len(t_rel):
                i = len(t_rel) - 1
            if kind == "acc":
                vec = entry["acc_enu"][i]
            elif kind == "dir":
                vec = entry["acc_dir_enu"][i] * dir_scale
            elif kind == "ff":
                vec = entry["ff_enu"][i]
            else:
                vec = entry["ff_plan_enu"][i]
            line.set_data(np.array([0.0, vec[0]]), np.array([0.0, vec[1]]))
            line.set_3d_properties(np.array([0.0, vec[2]]))
        ax.set_title(f"Acc vectors (ENU) - multi-drone | t={val:.2f}s")
        fig.canvas.draw_idle()

    time_slider.on_changed(update)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare logged vs planned vectors in 3D")
    parser.add_argument("log_path", help="Log directory or CSV log file")
    parser.add_argument("--planned-dir", required=True, help="Directory containing cable_*.csv")
    parser.add_argument("--drone-mass", type=float, default=0.25, help="Drone mass [kg]")
    parser.add_argument("--ff-weight", type=float, default=1.0, help="Feedforward weight")
    parser.add_argument("--max-drones", type=int, default=3, help="Maximum drones to plot")
    args = parser.parse_args()
    if args.drone_mass <= 0.0:
        raise RuntimeError("drone_mass must be > 0")

    log_files = list_log_files(Path(args.log_path))
    if not log_files:
        raise RuntimeError(f"No log files found in {args.log_path}")

    planned_dir = Path(args.planned_dir)
    if not planned_dir.exists():
        raise RuntimeError(f"Planned dir not found: {planned_dir}")

    entries = []
    for path in log_files:
        data = load_log(path)
        drone_id = parse_drone_id(path.name)
        if drone_id is None:
            continue
        vectors = extract_log_vectors(data)
        planned = load_planned_cable(planned_dir, drone_id)
        ff_plan_enu = None
        if planned is not None:
            t_plan, dirs_enu, mu = planned
            t_plan_rel = t_plan - t_plan[0]
            ff_plan = (args.ff_weight * mu / args.drone_mass)[:, None] * dirs_enu
            ff_plan_enu = interpolate_series(t_plan_rel, ff_plan, vectors["t_rel"])

        entry = {
            "name": path.name,
            "drone_id": drone_id,
            "t_rel": vectors["t_rel"],
            "acc_enu": vectors["acc_enu"],
            "acc_dir_enu": vectors["acc_dir_enu"],
            "ff_enu": vectors["ff_enu"],
            "ff_plan_enu": ff_plan_enu,
        }
        entries.append(entry)

    entries.sort(key=lambda e: e["drone_id"])
    entries = entries[: args.max_drones]

    for entry in entries:
        plot_drone_3d(
            entry["name"],
            entry["t_rel"],
            entry["acc_enu"],
            entry["acc_dir_enu"],
            entry["ff_enu"],
            entry["ff_plan_enu"],
        )

    plot_multi_3d(entries)
    plt.show()


if __name__ == "__main__":
    main()

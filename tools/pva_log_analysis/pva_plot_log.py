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
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider
import numpy as np

_SLIDERS = []


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


def plot_acc_dir(data_list: List[Tuple[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].set_title("Acceleration direction (NED)")
    has_any = False
    for name, data in data_list:
        if "acc_dir_x" not in data.dtype.names:
            continue
        t = data["t"]
        for ax, axis in zip(axes, ["x", "y", "z"]):
            ax.plot(t, data[f"acc_dir_{axis}"], label=f"{name} acc_dir_{axis}")
            ax.set_ylabel(axis)
        has_any = True
    if not has_any:
        plt.close(fig)
        return
    axes[-1].set_xlabel("time [s]")
    for ax in axes:
        ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()


def plot_feedforward_acc(data_list: List[Tuple[str, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Feedforward acceleration magnitude")
    has_any = False
    for name, data in data_list:
        if "ff_acc_x" in data.dtype.names:
            t = data["t"]
            ff_norm = vec_norm(data, "ff_acc")
            ax.plot(t, ff_norm, label=f"{name} ff_acc")
            has_any = True
    if not has_any:
        plt.close(fig)
        return
    ax.set_xlabel("time [s]")
    ax.set_ylabel("||ff_acc|| [m/s^2]")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()


def ned_to_enu(vec: np.ndarray) -> np.ndarray:
    return np.array([vec[1], vec[0], -vec[2]], dtype=float)


def plot_ff_acc_3d(data_list: List[Tuple[str, np.ndarray]]) -> None:
    for name, data in data_list:
        names = data.dtype.names
        if "ff_acc_x" not in names and "acc_x" not in names:
            continue
        t = data["t"]
        t_rel = t - t[0]
        ff_ned = np.vstack([data["ff_acc_x"], data["ff_acc_y"], data["ff_acc_z"]]).T if "ff_acc_x" in names else None
        acc_ned = np.vstack([data["acc_x"], data["acc_y"], data["acc_z"]]).T if "acc_x" in names else None
        if ff_ned is None and acc_ned is None:
            continue

        if "acc_dir_x" in names:
            acc_dir_ned = np.vstack([data["acc_dir_x"], data["acc_dir_y"], data["acc_dir_z"]]).T
        elif acc_ned is not None:
            acc_dir_ned = np.zeros_like(acc_ned)
            norms = np.linalg.norm(acc_ned, axis=1)
            valid = norms > 1e-9
            acc_dir_ned[valid] = acc_ned[valid] / norms[valid, None]
        else:
            acc_dir_ned = None

        ff_enu = np.vstack([ned_to_enu(v) for v in ff_ned]) if ff_ned is not None else None
        acc_enu = np.vstack([ned_to_enu(v) for v in acc_ned]) if acc_ned is not None else None
        acc_dir_enu = np.vstack([ned_to_enu(v) for v in acc_dir_ned]) if acc_dir_ned is not None else None

        mag_ff = np.linalg.norm(ff_enu, axis=1) if ff_enu is not None else None
        mag_acc = np.linalg.norm(acc_enu, axis=1) if acc_enu is not None else None
        max_mag = 0.0
        if mag_ff is not None and mag_ff.size:
            max_mag = max(max_mag, float(np.max(mag_ff)))
        if mag_acc is not None and mag_acc.size:
            max_mag = max(max_mag, float(np.max(mag_acc)))
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
        quiv_ff = None
        quiv_acc = None
        quiv_dir = None
        if ff_enu is not None:
            vec_ff = ff_enu[idx0]
            mag_ff0 = mag_ff[idx0]
            quiv_ff = ax.quiver(
                0.0, 0.0, 0.0,
                vec_ff[0], vec_ff[1], vec_ff[2],
                length=mag_ff0 if mag_ff0 > 0.0 else 0.0,
                normalize=mag_ff0 > 0.0,
                color="tab:blue",
            )
        if acc_enu is not None:
            vec_acc = acc_enu[idx0]
            mag_acc0 = mag_acc[idx0]
            quiv_acc = ax.quiver(
                0.0, 0.0, 0.0,
                vec_acc[0], vec_acc[1], vec_acc[2],
                length=mag_acc0 if mag_acc0 > 0.0 else 0.0,
                normalize=mag_acc0 > 0.0,
                color="tab:green",
            )
        if acc_dir_enu is not None:
            vec_dir = acc_dir_enu[idx0]
            mag_dir = np.linalg.norm(vec_dir)
            quiv_dir = ax.quiver(
                0.0, 0.0, 0.0,
                vec_dir[0], vec_dir[1], vec_dir[2],
                length=dir_scale if mag_dir > 0.0 else 0.0,
                normalize=mag_dir > 0.0,
                color="tab:orange",
            )

        legend_items = []
        if ff_enu is not None:
            legend_items.append(Line2D([0], [0], color="tab:blue", lw=2, label="ff_acc"))
        if acc_enu is not None:
            legend_items.append(Line2D([0], [0], color="tab:green", lw=2, label="acc_cmd"))
        if acc_dir_enu is not None:
            legend_items.append(Line2D([0], [0], color="tab:orange", lw=2, label="acc_dir"))
        if legend_items:
            ax.legend(handles=legend_items, loc="upper right")

        slider_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
        time_slider = Slider(slider_ax, "t", 0.0, float(t_rel[-1]), valinit=0.0)
        _SLIDERS.append(time_slider)

        quiv_state = {"ff": quiv_ff, "acc": quiv_acc, "dir": quiv_dir}

        def update(val: float,
                   t_rel=t_rel,
                   name=name,
                   ff_enu=ff_enu,
                   acc_enu=acc_enu,
                   acc_dir_enu=acc_dir_enu,
                   mag_ff=mag_ff,
                   mag_acc=mag_acc,
                   dir_scale=dir_scale,
                   ax=ax,
                   fig=fig,
                   quiv_state=quiv_state) -> None:
            idx = int(np.searchsorted(t_rel, val, side="left"))
            if idx >= len(t_rel):
                idx = len(t_rel) - 1
            parts = [f"{name} Acc vectors (ENU) t={t_rel[idx]:.2f}s"]
            if ff_enu is not None:
                vec_ff = ff_enu[idx]
                mag_ff_v = mag_ff[idx]
                if quiv_state["ff"] is not None:
                    quiv_state["ff"].remove()
                quiv_state["ff"] = ax.quiver(
                    0.0, 0.0, 0.0,
                    vec_ff[0], vec_ff[1], vec_ff[2],
                    length=mag_ff_v if mag_ff_v > 0.0 else 0.0,
                    normalize=mag_ff_v > 0.0,
                    color="tab:blue",
                )
                parts.append(f"|ff|={mag_ff_v:.3f}")
            if acc_enu is not None:
                vec_acc = acc_enu[idx]
                mag_acc_v = mag_acc[idx]
                if quiv_state["acc"] is not None:
                    quiv_state["acc"].remove()
                quiv_state["acc"] = ax.quiver(
                    0.0, 0.0, 0.0,
                    vec_acc[0], vec_acc[1], vec_acc[2],
                    length=mag_acc_v if mag_acc_v > 0.0 else 0.0,
                    normalize=mag_acc_v > 0.0,
                    color="tab:green",
                )
                parts.append(f"|acc|={mag_acc_v:.3f}")
            if acc_dir_enu is not None:
                vec_dir = acc_dir_enu[idx]
                mag_dir = np.linalg.norm(vec_dir)
                if quiv_state["dir"] is not None:
                    quiv_state["dir"].remove()
                quiv_state["dir"] = ax.quiver(
                    0.0, 0.0, 0.0,
                    vec_dir[0], vec_dir[1], vec_dir[2],
                    length=dir_scale if mag_dir > 0.0 else 0.0,
                    normalize=mag_dir > 0.0,
                    color="tab:orange",
                )
            ax.set_title(" | ".join(parts))
            fig.canvas.draw_idle()

        time_slider.on_changed(update)


def plot_ff_acc_3d_multi(data_list: List[Tuple[str, np.ndarray]]) -> None:
    entries = []
    for name, data in data_list:
        names = data.dtype.names
        if "ff_acc_x" not in names and "acc_x" not in names:
            continue
        entries.append((name, data))
        if len(entries) >= 3:
            break
    if len(entries) < 2:
        return

    colors = ["tab:blue", "tab:orange", "tab:green"]
    markers = {"ff": "o", "acc": "^", "dir": "s"}
    t_refs = [d["t"] - d["t"][0] for _, d in entries]
    t_max = max(float(t[-1]) for t in t_refs if t.size)

    ff_enu_all = []
    acc_enu_all = []
    acc_dir_enu_all = []
    max_mag = 0.0

    for _, data in entries:
        names = data.dtype.names
        ff_ned = np.vstack([data["ff_acc_x"], data["ff_acc_y"], data["ff_acc_z"]]).T if "ff_acc_x" in names else None
        acc_ned = np.vstack([data["acc_x"], data["acc_y"], data["acc_z"]]).T if "acc_x" in names else None
        if "acc_dir_x" in names:
            acc_dir_ned = np.vstack([data["acc_dir_x"], data["acc_dir_y"], data["acc_dir_z"]]).T
        elif acc_ned is not None:
            acc_dir_ned = np.zeros_like(acc_ned)
            norms = np.linalg.norm(acc_ned, axis=1)
            valid = norms > 1e-9
            acc_dir_ned[valid] = acc_ned[valid] / norms[valid, None]
        else:
            acc_dir_ned = None

        ff_enu = np.vstack([ned_to_enu(v) for v in ff_ned]) if ff_ned is not None else None
        acc_enu = np.vstack([ned_to_enu(v) for v in acc_ned]) if acc_ned is not None else None
        acc_dir_enu = np.vstack([ned_to_enu(v) for v in acc_dir_ned]) if acc_dir_ned is not None else None

        if ff_enu is not None and ff_enu.size:
            max_mag = max(max_mag, float(np.max(np.linalg.norm(ff_enu, axis=1))))
        if acc_enu is not None and acc_enu.size:
            max_mag = max(max_mag, float(np.max(np.linalg.norm(acc_enu, axis=1))))

        ff_enu_all.append(ff_enu)
        acc_enu_all.append(acc_enu)
        acc_dir_enu_all.append(acc_dir_enu)

    axis_lim = max(1.0, max_mag * 1.1)
    dir_scale = max(1.0, axis_lim * 0.5)

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

    lines = []
    for idx, (name, _) in enumerate(entries):
        color = colors[idx % len(colors)]
        ff_vec = ff_enu_all[idx]
        acc_vec = acc_enu_all[idx]
        dir_vec = acc_dir_enu_all[idx]

        if ff_vec is not None:
            line_ff, = ax.plot([0.0, ff_vec[0, 0]],
                               [0.0, ff_vec[0, 1]],
                               [0.0, ff_vec[0, 2]],
                               color=color, marker=markers["ff"],
                               label=f"{name} ff_acc")
            lines.append(("ff", idx, line_ff))
        if acc_vec is not None:
            line_acc, = ax.plot([0.0, acc_vec[0, 0]],
                                [0.0, acc_vec[0, 1]],
                                [0.0, acc_vec[0, 2]],
                                color=color, marker=markers["acc"],
                                label=f"{name} acc_cmd")
            lines.append(("acc", idx, line_acc))
        if dir_vec is not None:
            vec0 = dir_vec[0] * dir_scale
            line_dir, = ax.plot([0.0, vec0[0]],
                                [0.0, vec0[1]],
                                [0.0, vec0[2]],
                                color=color, marker=markers["dir"],
                                label=f"{name} acc_dir")
            lines.append(("dir", idx, line_dir))

    ax.legend(ncol=2, fontsize=8)

    slider_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
    time_slider = Slider(slider_ax, "t", 0.0, t_max, valinit=0.0)
    _SLIDERS.append(time_slider)

    def update(val: float) -> None:
        parts = [f"t={val:.2f}s"]
        for kind, idx, line in lines:
            t_rel = t_refs[idx]
            i = int(np.searchsorted(t_rel, val, side="left"))
            if i >= len(t_rel):
                i = len(t_rel) - 1
            if kind == "ff":
                vec = ff_enu_all[idx][i]
            elif kind == "acc":
                vec = acc_enu_all[idx][i]
            else:
                vec = acc_dir_enu_all[idx][i] * dir_scale
            line.set_data(np.array([0.0, vec[0]]), np.array([0.0, vec[1]]))
            line.set_3d_properties(np.array([0.0, vec[2]]))
        ax.set_title("Acc vectors (ENU) - multi-drone | " + " ".join(parts))
        fig.canvas.draw_idle()

    time_slider.on_changed(update)


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
    plot_acc_dir(data_list)
    plot_feedforward_acc(data_list)
    plot_ff_acc_3d(data_list)
    plot_ff_acc_3d_multi(data_list)
    plot_feedback(data_list)
    plot_jerk(data_list)
    plt.show()


if __name__ == "__main__":
    main()

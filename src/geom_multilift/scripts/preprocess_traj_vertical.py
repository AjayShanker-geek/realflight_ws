#!/usr/bin/env python3
"""
Convert vertical-obstacle 3-quad trajectory .npy files into CSV for C++ consumption.
Scenario: Planning_plots_multiagent_meta_evaluation_COM_Dyn_V(m2=0.1,rg=[0.022,0.016])
Outputs 100 Hz trajectories using local high-order polynomial fits.
"""

from pathlib import Path
import csv

import numpy as np


TARGET_DT = 0.01
WINDOW = 8
ORDER = 7
DEFAULT_DT = 0.05


def extend_time_series(time_raw: np.ndarray, series_raw: np.ndarray, target_time: float, time_axis: int = 0):
    if time_raw[-1] >= target_time:
        return time_raw, series_raw
    pad_slice = [slice(None)] * series_raw.ndim
    pad_slice[time_axis] = slice(-1, None)
    series_pad = series_raw[tuple(pad_slice)]
    series_ext = np.concatenate([series_raw, series_pad], axis=time_axis)
    time_ext = np.append(time_raw, target_time)
    return time_ext, series_ext


def resample_polynomial_series(
    series_raw: np.ndarray,
    time_raw: np.ndarray,
    time_dense: np.ndarray,
    window: int = WINDOW,
    order: int = ORDER,
) -> np.ndarray:
    num_tracks, _, dims = series_raw.shape
    out = np.zeros((num_tracks, len(time_dense), dims))
    half_w = window // 2

    for i in range(num_tracks):
        for axis in range(dims):
            values = series_raw[i, :, axis]
            for idx_t, t in enumerate(time_dense):
                close_mask = np.isclose(time_raw, t)
                if np.any(close_mask):
                    raw_idx = int(np.argmax(close_mask))
                    is_raw = True
                else:
                    raw_idx = int(np.searchsorted(time_raw, t))
                    is_raw = False
                    if raw_idx >= len(time_raw):
                        raw_idx = len(time_raw) - 1
                center = raw_idx
                start = max(0, center - half_w)
                end = min(len(time_raw), start + window)
                start = max(0, end - window)
                t_win = time_raw[start:end]
                v_win = values[start:end]
                order_fit = min(order, len(t_win) - 1)
                if order_fit < 1:
                    out[i, idx_t, axis] = values[center]
                    continue
                t_shift = t_win - t
                coeffs = np.polyfit(t_shift, v_win, order_fit)
                val = np.polyval(coeffs, 0.0)
                if is_raw:
                    val = values[raw_idx]
                out[i, idx_t, axis] = val

    return out


def resample_quaternion(
    q_raw: np.ndarray,
    time_raw: np.ndarray,
    time_dense: np.ndarray,
    window: int = WINDOW,
    order: int = ORDER,
) -> np.ndarray:
    q_cont = q_raw.copy()
    for i in range(1, q_cont.shape[0]):
        if np.dot(q_cont[i - 1], q_cont[i]) < 0.0:
            q_cont[i] *= -1.0
    q_series = q_cont[None, :, :]
    q_resampled = resample_polynomial_series(q_series, time_raw, time_dense, window, order)[0]
    norms = np.linalg.norm(q_resampled, axis=1, keepdims=True)
    return q_resampled / np.clip(norms, 1e-9, None)


def normalize_vectors(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=2, keepdims=True)
    return vecs / np.clip(norms, 1e-9, None)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    scenario = root / "raw_data" / "3quad_traj" / "Planning_plots_multiagent_meta_evaluation_COM_Dyn_V(m2=0.1,rg=[0.022,0.016])"
    out_dir = root / "data" / "preprocessed" / "rg_0022_0016_3quad_vertical_100hz"
    out_dir.mkdir(parents=True, exist_ok=True)

    dt = DEFAULT_DT

    xl = np.load(scenario / "xl_traj_0_3_a_3.npy", allow_pickle=True)  # (T,13)
    xq = np.load(scenario / "xq_traj_0_3_a_3.npy", allow_pickle=True)  # (N,T,14)
    kb = np.load(scenario / "Kfb_traj_0_3_a_3.npy", allow_pickle=True)  # (T-1,6,13)

    T = xl.shape[0]
    N = xq.shape[0]
    time_raw = np.arange(T) * dt
    time_dense = np.arange(0.0, time_raw[-1] + TARGET_DT / 2.0, TARGET_DT)

    payload_pos = resample_polynomial_series(xl[:, 0:3][None, :, :], time_raw, time_dense)[0]
    payload_vel = resample_polynomial_series(xl[:, 3:6][None, :, :], time_raw, time_dense)[0]
    payload_q = resample_quaternion(xl[:, 6:10], time_raw, time_dense)
    payload_omega = resample_polynomial_series(xl[:, 10:13][None, :, :], time_raw, time_dense)[0]

    cable_dir = resample_polynomial_series(xq[:, :, 0:3], time_raw, time_dense)
    cable_dir = normalize_vectors(cable_dir)
    cable_omega = resample_polynomial_series(xq[:, :, 3:6], time_raw, time_dense)
    cable_omega_dot = resample_polynomial_series(xq[:, :, 6:9], time_raw, time_dense)
    cable_mu = resample_polynomial_series(xq[:, :, 12:13], time_raw, time_dense)[..., 0]

    time_k = np.arange(kb.shape[0]) * dt
    time_k, kb = extend_time_series(time_k, kb, time_raw[-1], time_axis=0)
    kfb_flat = kb.reshape(kb.shape[0], -1)
    kfb_series = kfb_flat.T[:, :, None]
    kfb_resampled = resample_polynomial_series(kfb_series, time_k, time_dense)
    kfb_out = kfb_resampled[:, :, 0].T.reshape(len(time_dense), 6, 13)

    # payload.csv
    with open(out_dir / "payload.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "x", "y", "z", "vx", "vy", "vz",
                         "qw", "qx", "qy", "qz", "wx", "wy", "wz"])
        for t_idx, t in enumerate(time_dense):
            row = [t]
            row += payload_pos[t_idx].tolist()
            row += payload_vel[t_idx].tolist()
            row += payload_q[t_idx].tolist()
            row += payload_omega[t_idx].tolist()
            writer.writerow(row)

    # per-drone cable files
    for i in range(N):
        with open(out_dir / f"cable_{i}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time",
                "dir_x", "dir_y", "dir_z",
                "omega_x", "omega_y", "omega_z",
                "mu",
                "omega_dot_x", "omega_dot_y", "omega_dot_z",
            ])
            for t_idx, t in enumerate(time_dense):
                row = [t]
                row += cable_dir[i, t_idx].tolist()
                row += cable_omega[i, t_idx].tolist()
                row += [cable_mu[i, t_idx]]
                row += cable_omega_dot[i, t_idx].tolist()
                writer.writerow(row)

    # Kfb flatten
    with open(out_dir / "kfb.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["time"] + [f"k{r}_{c}" for r in range(6) for c in range(13)]
        writer.writerow(header)
        for t_idx, t in enumerate(time_dense):
            row = [t]
            row += kfb_out[t_idx].reshape(-1).tolist()
            writer.writerow(row)

    print(f"Wrote preprocessed data to {out_dir}")


if __name__ == "__main__":
    main()

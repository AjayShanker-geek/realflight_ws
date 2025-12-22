#!/usr/bin/env python3
"""
Save the vertical-obstacle 3-quad trajectory to per-drone CSV files.
- Raw 20 Hz data from COM_Dyn_V .npy files.
- Smoothed 100 Hz version via local high-order polynomial fits to the offline data
  (positions preserved at raw samples) with analytic derivatives up to snap,
  plus attitude and body rates from differential flatness.
"""

import csv
import math
from pathlib import Path
from typing import Optional, Union

import numpy as np
try:
    import matplotlib.pyplot as plt
except Exception as exc:  # matplotlib may be missing or mismatched locally
    plt = None
    _PLOT_IMPORT_ERROR = exc
else:
    _PLOT_IMPORT_ERROR = None


FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FILE_DIR.parent
SCENARIO_DIR = (
    REPO_ROOT
    / "raw_data"
    / "3quad_traj"
    / "Planning_plots_multiagent_meta_evaluation_COM_Dyn_V(m2=0.1,rg=[0.022,0.016])"
)


class DataLoaderVertical:
    """
    Load planned trajectory data for the COM_Dyn_V evaluation.
    """

    def __init__(self, scenario_dir: Path = SCENARIO_DIR):
        self.path = scenario_dir
        # Keep defaults aligned with COM_Dyn_V config (dt=0.05, rl=0.13, cl0=1.0)
        self.dt = 0.05
        self.rl = 0.13
        self.cable_length = 1.0
        self.g = 9.81
        self.ez = np.array([0.0, 0.0, 1.0]).reshape(3, 1)

        self.xl_traj = np.load(self.path / "xl_traj_0_3_a_3.npy", allow_pickle=True)
        self.payload_x = self.xl_traj[:, 0:3]
        self.payload_v = self.xl_traj[:, 3:6]
        self.payload_q = self.xl_traj[:, 6:10]
        self.payload_w = self.xl_traj[:, 10:13]

        self.xq_traj = np.load(self.path / "xq_traj_0_3_a_3.npy", allow_pickle=True)
        self.num_drones = self.xq_traj.shape[0]
        self.alpha = 2 * math.pi / self.num_drones
        self.cable_direction = self.xq_traj[:, :, 0:3]
        self.cable_omega = self.xq_traj[:, :, 3:6]
        self.cable_omega_dot = self.xq_traj[:, :, 6:9]
        self.cable_mu = self.xq_traj[:, :, 12]
        self.cable_mu_dot = self.xq_traj[:, :, 13]


class TrajectorySaverVertical:
    def __init__(self, output_dir: Union[str, Path] = "./data/3drone_trajectories_new_vertical"):
        self.loader = DataLoaderVertical()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_drones = self.loader.num_drones
        self.dt = self.loader.dt
        self.target_dt = 0.01  # 100 Hz output
        self.rl = self.loader.rl
        self.cable_length = self.loader.cable_length
        self.alpha = self.loader.alpha
        self.T_enu2ned = np.array([[0, 1, 0],
                                   [1, 0, 0],
                                   [0, 0, -1]])
        self.g = self.loader.g
        self.ez = self.loader.ez.reshape(3)

    def enu_to_ned(self, data: np.ndarray) -> np.ndarray:
        flat = data.reshape(-1, 3)
        transformed = (self.T_enu2ned @ flat.T).T
        return transformed.reshape(data.shape)

    def compute_positions(self) -> np.ndarray:
        """Compute drone positions over time in ENU (shape: num_drones x N x 3)."""
        N = self.loader.payload_x.shape[0]
        pos_enu = np.zeros((self.num_drones, N, 3))

        ri = np.array(
            [
                [
                    self.rl * np.cos(i * self.alpha),
                    self.rl * np.sin(i * self.alpha),
                    0.0,
                ]
                for i in range(self.num_drones)
            ]
        )

        payload_enu = self.loader.payload_x
        cable_dir_enu = self.loader.cable_direction
        for i in range(self.num_drones):
            pos_enu[i] = payload_enu + ri[i] + self.cable_length * cable_dir_enu[i]
        return pos_enu

    def compute_velocities(self, pos: np.ndarray) -> np.ndarray:
        return np.gradient(pos, self.dt, axis=1)

    def compute_accelerations(self, vel: np.ndarray) -> np.ndarray:
        return np.gradient(vel, self.dt, axis=1)

    def resample_polynomial(
        self,
        pos_raw: np.ndarray,
        time_raw: np.ndarray,
        time_dense: np.ndarray,
        window: int = 8,
        order: int = 7,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        num_drones, _, _ = pos_raw.shape
        pos_out = np.zeros((num_drones, len(time_dense), 3))
        vel_out = np.zeros_like(pos_out)
        acc_out = np.zeros_like(pos_out)
        jerk_out = np.zeros_like(pos_out)
        snap_out = np.zeros_like(pos_out)

        half_w = window // 2
        for i in range(num_drones):
            for axis in range(3):
                values = pos_raw[i, :, axis]
                for idx_t, t in enumerate(time_dense):
                    close_mask = np.isclose(time_raw, t)
                    raw_idx = int(np.argmax(close_mask)) if np.any(close_mask) else np.searchsorted(time_raw, t)
                    is_raw = raw_idx < len(time_raw) and np.isclose(time_raw[raw_idx], t)
                    center = raw_idx
                    start = max(0, center - half_w)
                    end = min(len(time_raw), start + window)
                    start = max(0, end - window)
                    t_win = time_raw[start:end]
                    v_win = values[start:end]
                    order_fit = min(order, len(t_win) - 1)
                    if order_fit < 1:
                        pos_out[i, idx_t, axis] = values[center if center < len(values) else -1]
                        continue
                    t_shift = t_win - t
                    coeffs = np.polyfit(t_shift, v_win, order_fit)
                    pos_val = np.polyval(coeffs, 0.0)
                    if is_raw:
                        pos_val = values[raw_idx]
                    vel_coeffs = np.polyder(coeffs, 1)
                    acc_coeffs = np.polyder(coeffs, 2)
                    jerk_coeffs = np.polyder(coeffs, 3)
                    snap_coeffs = np.polyder(coeffs, 4)
                    vel_val = np.polyval(vel_coeffs, 0.0)
                    acc_val = np.polyval(acc_coeffs, 0.0)
                    jerk_val = np.polyval(jerk_coeffs, 0.0)
                    snap_val = np.polyval(snap_coeffs, 0.0)

                    pos_out[i, idx_t, axis] = pos_val
                    vel_out[i, idx_t, axis] = vel_val
                    acc_out[i, idx_t, axis] = acc_val
                    jerk_out[i, idx_t, axis] = jerk_val
                    snap_out[i, idx_t, axis] = snap_val

        return pos_out, vel_out, acc_out, jerk_out, snap_out

    def save_csv(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        time: np.ndarray,
        suffix: str,
        accel: Optional[np.ndarray] = None,
        jerk: Optional[np.ndarray] = None,
        snap: Optional[np.ndarray] = None,
        roll: Optional[np.ndarray] = None,
        pitch: Optional[np.ndarray] = None,
        yaw: Optional[np.ndarray] = None,
        p_rates: Optional[np.ndarray] = None,
        q_rates: Optional[np.ndarray] = None,
        r_rates: Optional[np.ndarray] = None,
    ) -> None:
        for i in range(self.num_drones):
            out_file = self.output_dir / f"drone_{i}_traj_{suffix}.csv"
            with out_file.open("w", newline="") as f:
                writer = csv.writer(f)
                headers = ["time", "x", "y", "z", "vx", "vy", "vz"]
                cols = [
                    time,
                    pos[i, :, 0],
                    pos[i, :, 1],
                    pos[i, :, 2],
                    vel[i, :, 0],
                    vel[i, :, 1],
                    vel[i, :, 2],
                ]

                if accel is not None:
                    headers += ["ax", "ay", "az"]
                    cols += [accel[i, :, 0], accel[i, :, 1], accel[i, :, 2]]
                if jerk is not None:
                    headers += ["jx", "jy", "jz"]
                    cols += [jerk[i, :, 0], jerk[i, :, 1], jerk[i, :, 2]]
                if snap is not None:
                    headers += ["sx", "sy", "sz"]
                    cols += [snap[i, :, 0], snap[i, :, 1], snap[i, :, 2]]
                if roll is not None and pitch is not None and yaw is not None:
                    headers += ["roll", "pitch", "yaw"]
                    cols += [roll[i, :], pitch[i, :], yaw[i, :]]
                if p_rates is not None and q_rates is not None and r_rates is not None:
                    headers += ["p", "q", "r"]
                    cols += [p_rates[i, :], q_rates[i, :], r_rates[i, :]]

                writer.writerow(headers)
                data = np.column_stack(cols)
                for row in data:
                    writer.writerow([f"{val:.6f}" for val in row])
            print(f"Saved {out_file}")

    def plot_comparison(
        self,
        pos_raw: np.ndarray,
        vel_raw: np.ndarray,
        time_raw: np.ndarray,
        pos_smooth: np.ndarray,
        vel_smooth: np.ndarray,
        time_smooth: np.ndarray,
        drone_idx: int = 0,
    ) -> None:
        if plt is None:
            print(f"Skipping plots; matplotlib not available: {_PLOT_IMPORT_ERROR}")
            return

        labels = ["x", "y", "z"]
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex="row")

        for j, label in enumerate(labels):
            axes[0, j].plot(time_smooth, pos_smooth[drone_idx, :, j], label="smoothed 100 Hz")
            axes[0, j].plot(
                time_raw,
                pos_raw[drone_idx, :, j],
                "o",
                markersize=3,
                alpha=0.6,
                label="raw 20 Hz" if j == 0 else None,
            )
            axes[0, j].set_title(f"{label}-pos [m]")

            axes[1, j].plot(time_smooth, vel_smooth[drone_idx, :, j], label="smoothed 100 Hz")
            axes[1, j].plot(
                time_raw,
                vel_raw[drone_idx, :, j],
                "o",
                markersize=3,
                alpha=0.6,
                label="raw 20 Hz" if j == 0 else None,
            )
            axes[1, j].set_title(f"{label}-vel [m/s]")

        axes[0, 0].set_ylabel("Position")
        axes[1, 0].set_ylabel("Velocity")
        axes[1, 1].set_xlabel("Time [s]")
        axes[1, 2].set_xlabel("Time [s]")
        handles, labels_legend = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels_legend, loc="upper center", ncol=2)
        fig.tight_layout()
        out_file = self.output_dir / f"traj_compare_drone{drone_idx}.png"
        fig.savefig(out_file, dpi=200)
        plt.close(fig)
        print(f"Saved comparison plot to {out_file}")

    def compute_attitude_and_body_rates(
        self, accel: np.ndarray, jerk: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        N = accel.shape[1]
        roll = np.zeros((self.num_drones, N))
        pitch = np.zeros_like(roll)
        yaw = np.zeros_like(roll)
        p_rates = np.zeros_like(roll)
        q_rates = np.zeros_like(roll)
        r_rates = np.zeros_like(roll)

        g_vec = self.g * self.ez
        psi_des = 0.0
        x_c = np.array([np.cos(psi_des), np.sin(psi_des), 0.0])
        y_c = np.array([-np.sin(psi_des), np.cos(psi_des), 0.0])

        for i in range(self.num_drones):
            for k in range(N):
                a_k = accel[i, k, :]
                j_k = jerk[i, k, :]

                f_vec = g_vec - a_k
                T = np.linalg.norm(f_vec)
                if T < 1e-6:
                    T = 1e-6
                z_b = f_vec / T

                x_b = np.cross(y_c, z_b)
                norm_xb = np.linalg.norm(x_b)
                if norm_xb < 1e-6:
                    x_b = np.array([1.0, 0.0, 0.0])
                    norm_xb = 1.0
                x_b /= norm_xb
                y_b = np.cross(z_b, x_b)
                y_b /= np.linalg.norm(y_b)
                R = np.column_stack((x_b, y_b, z_b))

                roll[i, k] = np.arctan2(R[2, 1], R[2, 2])
                pitch[i, k] = np.arcsin(-R[2, 0])
                yaw[i, k] = np.arctan2(R[1, 0], R[0, 0])

                f_dot = -j_k
                T_dot = np.dot(f_dot, z_b)
                z_b_dot = (f_dot - T_dot * z_b) / T
                omega_world = np.cross(z_b, z_b_dot)
                omega_body = R.T @ omega_world

                p_rates[i, k] = omega_body[0]
                q_rates[i, k] = omega_body[1]
                r_rates[i, k] = omega_body[2]

        return roll, pitch, yaw, p_rates, q_rates, r_rates

    def run(self) -> None:
        time_raw = np.arange(self.loader.payload_x.shape[0]) * self.dt
        time_dense_end = time_raw[-1]
        time_dense = np.arange(0.0, time_dense_end + self.target_dt / 2, self.target_dt)

        pos_raw_enu = self.compute_positions()
        vel_raw_enu = self.compute_velocities(pos_raw_enu)
        acc_raw_enu = self.compute_accelerations(vel_raw_enu)
        jerk_raw_enu = np.gradient(acc_raw_enu, self.dt, axis=1)
        snap_raw_enu = np.gradient(jerk_raw_enu, self.dt, axis=1)

        pos_raw = self.enu_to_ned(pos_raw_enu)
        vel_raw = self.enu_to_ned(vel_raw_enu)
        acc_raw = self.enu_to_ned(acc_raw_enu)
        jerk_raw = self.enu_to_ned(jerk_raw_enu)
        snap_raw = self.enu_to_ned(snap_raw_enu)

        self.save_csv(
            pos_raw_enu,
            vel_raw_enu,
            time_raw,
            suffix="raw_20hz_enu",
            accel=acc_raw_enu,
            jerk=jerk_raw_enu,
            snap=snap_raw_enu,
        )

        (
            roll_raw,
            pitch_raw,
            yaw_raw,
            p_rates_raw,
            q_rates_raw,
            r_rates_raw,
        ) = self.compute_attitude_and_body_rates(acc_raw, jerk_raw)
        self.save_csv(
            pos_raw,
            vel_raw,
            time_raw,
            suffix="raw_20hz",
            accel=acc_raw,
            jerk=jerk_raw,
            snap=snap_raw,
            roll=roll_raw,
            pitch=pitch_raw,
            yaw=yaw_raw,
            p_rates=p_rates_raw,
            q_rates=q_rates_raw,
            r_rates=r_rates_raw,
        )

        (
            pos_smooth_enu,
            vel_smooth_enu,
            acc_smooth_enu,
            jerk_smooth_enu,
            snap_smooth_enu,
        ) = self.resample_polynomial(pos_raw_enu, time_raw, time_dense)

        pos_smooth = self.enu_to_ned(pos_smooth_enu)
        vel_smooth = self.enu_to_ned(vel_smooth_enu)
        acc_smooth = self.enu_to_ned(acc_smooth_enu)
        jerk_smooth = self.enu_to_ned(jerk_smooth_enu)
        snap_smooth = self.enu_to_ned(snap_smooth_enu)

        self.save_csv(
            pos_smooth_enu,
            vel_smooth_enu,
            time_dense,
            suffix="smoothed_100hz_enu",
            accel=acc_smooth_enu,
            jerk=jerk_smooth_enu,
            snap=snap_smooth_enu,
        )

        (
            roll_smooth,
            pitch_smooth,
            yaw_smooth,
            p_rates_smooth,
            q_rates_smooth,
            r_rates_smooth,
        ) = self.compute_attitude_and_body_rates(acc_smooth, jerk_smooth)
        self.save_csv(
            pos_smooth,
            vel_smooth,
            time_dense,
            suffix="smoothed_100hz",
            accel=acc_smooth,
            jerk=jerk_smooth,
            snap=snap_smooth,
            roll=roll_smooth,
            pitch=pitch_smooth,
            yaw=yaw_smooth,
            p_rates=p_rates_smooth,
            q_rates=q_rates_smooth,
            r_rates=r_rates_smooth,
        )

        self.plot_comparison(pos_raw, vel_raw, time_raw, pos_smooth, vel_smooth, time_dense)


def main() -> None:
    saver = TrajectorySaverVertical()
    saver.run()


if __name__ == "__main__":
    main()

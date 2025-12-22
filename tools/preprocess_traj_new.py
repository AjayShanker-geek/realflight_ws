#!/usr/bin/env python3
"""
Convert the new COM_Dyn_H trajectory to CSV for:
- Per-drone follow trajectories (NED, raw 20 Hz + smoothed 100 Hz).
- Geometric controller data (payload/cable/Kfb in ENU at 100 Hz).

Column naming follows tools/save_traj_new.py.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np

FILE_DIR = Path(__file__).resolve().parent
sys.path.append(str(FILE_DIR))

from get_data_new import DataLoader


class TrajectoryConverterNew:
    def __init__(
        self,
        output_dir: Path,
        target_dt: float = 0.01,
        window: int = 8,
        order: int = 7,
    ) -> None:
        self.loader = DataLoader()
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_drones = self.loader.num_drones
        self.dt = self.loader.dt
        self.target_dt = target_dt
        self.window = window
        self.order = order
        self.rl = self.loader.rl
        self.cable_length = self.loader.cable_length
        self.alpha = self.loader.alpha
        self.g = self.loader.g
        self.ez = self.loader.ez.reshape(3)
        self.T_enu2ned = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )

    def enu_to_ned(self, data: np.ndarray) -> np.ndarray:
        flat = data.reshape(-1, 3)
        transformed = (self.T_enu2ned @ flat.T).T
        return transformed.reshape(data.shape)

    def compute_positions(self) -> np.ndarray:
        N = self.loader.payload_x.shape[0]
        pos_enu = np.zeros((self.num_drones, N, 3))
        ri = np.array(
            [
                [self.rl * np.cos(i * self.alpha), self.rl * np.sin(i * self.alpha), 0.0]
                for i in range(self.num_drones)
            ]
        )
        payload_enu = self.loader.payload_x
        cable_dir_enu = self.loader.cable_direction
        for i in range(self.num_drones):
            pos_enu[i] = payload_enu + ri[i] + self.cable_length * cable_dir_enu[i]
        return pos_enu

    @staticmethod
    def compute_velocities(pos: np.ndarray, dt: float) -> np.ndarray:
        return np.gradient(pos, dt, axis=1)

    @staticmethod
    def compute_accelerations(vel: np.ndarray, dt: float) -> np.ndarray:
        return np.gradient(vel, dt, axis=1)

    @staticmethod
    def normalize_vectors(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=2, keepdims=True)
        return vecs / np.clip(norms, 1e-9, None)

    def resample_polynomial(
        self,
        series_raw: np.ndarray,
        time_raw: np.ndarray,
        time_dense: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        num_tracks, _, _ = series_raw.shape
        pos_out = np.zeros((num_tracks, len(time_dense), 3))
        vel_out = np.zeros_like(pos_out)
        acc_out = np.zeros_like(pos_out)
        jerk_out = np.zeros_like(pos_out)
        snap_out = np.zeros_like(pos_out)

        half_w = self.window // 2
        for i in range(num_tracks):
            for axis in range(3):
                values = series_raw[i, :, axis]
                for idx_t, t in enumerate(time_dense):
                    close_mask = np.isclose(time_raw, t)
                    raw_idx = int(np.argmax(close_mask)) if np.any(close_mask) else np.searchsorted(time_raw, t)
                    is_raw = raw_idx < len(time_raw) and np.isclose(time_raw[raw_idx], t)
                    center = raw_idx
                    start = max(0, center - half_w)
                    end = min(len(time_raw), start + self.window)
                    start = max(0, end - self.window)
                    t_win = time_raw[start:end]
                    v_win = values[start:end]
                    order_fit = min(self.order, len(t_win) - 1)
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

    @staticmethod
    def resample_vector_linear(
        series_raw: np.ndarray, time_raw: np.ndarray, time_dense: np.ndarray
    ) -> np.ndarray:
        num_tracks = series_raw.shape[0]
        out = np.zeros((num_tracks, len(time_dense), 3))
        for i in range(num_tracks):
            for axis in range(3):
                out[i, :, axis] = np.interp(time_dense, time_raw, series_raw[i, :, axis])
        return out

    @staticmethod
    def resample_scalar_linear(
        series_raw: np.ndarray, time_raw: np.ndarray, time_dense: np.ndarray
    ) -> np.ndarray:
        num_tracks = series_raw.shape[0]
        out = np.zeros((num_tracks, len(time_dense)))
        for i in range(num_tracks):
            out[i, :] = np.interp(time_dense, time_raw, series_raw[i, :])
        return out

    @staticmethod
    def resample_quaternion(
        q_raw: np.ndarray, time_raw: np.ndarray, time_dense: np.ndarray
    ) -> np.ndarray:
        q_out = np.zeros((len(time_dense), 4))
        q_cont = q_raw.copy()
        for i in range(1, q_cont.shape[0]):
            if np.dot(q_cont[i - 1], q_cont[i]) < 0.0:
                q_cont[i] *= -1.0
        for k in range(4):
            q_out[:, k] = np.interp(time_dense, time_raw, q_cont[:, k])
        norms = np.linalg.norm(q_out, axis=1, keepdims=True)
        q_out = q_out / np.clip(norms, 1e-9, None)
        return q_out

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
                thrust = np.linalg.norm(f_vec)
                if thrust < 1e-6:
                    thrust = 1e-6
                z_b = f_vec / thrust

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
                thrust_dot = np.dot(f_dot, z_b)
                z_b_dot = (f_dot - thrust_dot * z_b) / thrust
                omega_world = np.cross(z_b, z_b_dot)
                omega_body = R.T @ omega_world

                p_rates[i, k] = omega_body[0]
                q_rates[i, k] = omega_body[1]
                r_rates[i, k] = omega_body[2]

        return roll, pitch, yaw, p_rates, q_rates, r_rates

    def save_drone_csv(
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

    def save_payload_csv(
        self,
        time: np.ndarray,
        pos: np.ndarray,
        vel: np.ndarray,
        acc: np.ndarray,
        jerk: np.ndarray,
        snap: np.ndarray,
        q: np.ndarray,
        omega: np.ndarray,
    ) -> None:
        out_file = self.output_dir / "payload.csv"
        headers = [
            "time",
            "x", "y", "z",
            "vx", "vy", "vz",
            "ax", "ay", "az",
            "jx", "jy", "jz",
            "sx", "sy", "sz",
            "qw", "qx", "qy", "qz",
            "wx", "wy", "wz",
        ]
        cols = [
            time,
            pos[:, 0], pos[:, 1], pos[:, 2],
            vel[:, 0], vel[:, 1], vel[:, 2],
            acc[:, 0], acc[:, 1], acc[:, 2],
            jerk[:, 0], jerk[:, 1], jerk[:, 2],
            snap[:, 0], snap[:, 1], snap[:, 2],
            q[:, 0], q[:, 1], q[:, 2], q[:, 3],
            omega[:, 0], omega[:, 1], omega[:, 2],
        ]
        with out_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            data = np.column_stack(cols)
            for row in data:
                writer.writerow([f"{val:.6f}" for val in row])

    def save_cable_csv(
        self,
        time: np.ndarray,
        cable_dir: np.ndarray,
        cable_omega: np.ndarray,
        cable_omega_dot: np.ndarray,
        cable_mu: np.ndarray,
    ) -> None:
        headers = [
            "time",
            "dir_x", "dir_y", "dir_z",
            "omega_x", "omega_y", "omega_z",
            "omega_dot_x", "omega_dot_y", "omega_dot_z",
            "mu",
        ]
        for i in range(self.num_drones):
            out_file = self.output_dir / f"cable_{i}.csv"
            cols = [
                time,
                cable_dir[i, :, 0], cable_dir[i, :, 1], cable_dir[i, :, 2],
                cable_omega[i, :, 0], cable_omega[i, :, 1], cable_omega[i, :, 2],
                cable_omega_dot[i, :, 0], cable_omega_dot[i, :, 1], cable_omega_dot[i, :, 2],
                cable_mu[i, :],
            ]
            with out_file.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                data = np.column_stack(cols)
                for row in data:
                    writer.writerow([f"{val:.6f}" for val in row])

    @staticmethod
    def save_kfb_csv(time: np.ndarray, kfb: np.ndarray, out_path: Path) -> None:
        header = ["time"] + [f"k{r}_{c}" for r in range(6) for c in range(13)]
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for idx, t in enumerate(time):
                row = [t] + kfb[idx].reshape(-1).tolist()
                writer.writerow([f"{val:.6f}" for val in row])

    def run(self) -> None:
        time_raw = np.arange(self.loader.payload_x.shape[0]) * self.dt
        time_dense_end = time_raw[-1]
        time_dense = np.arange(0.0, time_dense_end + self.target_dt / 2.0, self.target_dt)

        # Per-drone trajectories (NED)
        pos_raw_enu = self.compute_positions()
        vel_raw_enu = self.compute_velocities(pos_raw_enu, self.dt)
        acc_raw_enu = self.compute_accelerations(vel_raw_enu, self.dt)
        jerk_raw_enu = np.gradient(acc_raw_enu, self.dt, axis=1)
        snap_raw_enu = np.gradient(jerk_raw_enu, self.dt, axis=1)

        pos_raw = self.enu_to_ned(pos_raw_enu)
        vel_raw = self.enu_to_ned(vel_raw_enu)
        acc_raw = self.enu_to_ned(acc_raw_enu)
        jerk_raw = self.enu_to_ned(jerk_raw_enu)
        snap_raw = self.enu_to_ned(snap_raw_enu)

        self.save_drone_csv(
            pos_raw,
            vel_raw,
            time_raw,
            suffix="raw_20hz",
            accel=acc_raw,
            jerk=jerk_raw,
            snap=snap_raw,
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

        roll, pitch, yaw, p_rates, q_rates, r_rates = self.compute_attitude_and_body_rates(
            acc_smooth, jerk_smooth
        )
        self.save_drone_csv(
            pos_smooth,
            vel_smooth,
            time_dense,
            suffix="smoothed_100hz",
            accel=acc_smooth,
            jerk=jerk_smooth,
            snap=snap_smooth,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            p_rates=p_rates,
            q_rates=q_rates,
            r_rates=r_rates,
        )

        # Payload trajectory (ENU) for geometric controller
        payload_pos_raw = self.loader.payload_x.reshape(1, -1, 3)
        (
            payload_pos_s,
            payload_vel_s,
            payload_acc_s,
            payload_jerk_s,
            payload_snap_s,
        ) = self.resample_polynomial(payload_pos_raw, time_raw, time_dense)

        payload_pos_s = payload_pos_s[0]
        payload_vel_s = payload_vel_s[0]
        payload_acc_s = payload_acc_s[0]
        payload_jerk_s = payload_jerk_s[0]
        payload_snap_s = payload_snap_s[0]

        payload_q_s = self.resample_quaternion(self.loader.payload_q, time_raw, time_dense)
        payload_w_s = np.column_stack([
            np.interp(time_dense, time_raw, self.loader.payload_w[:, 0]),
            np.interp(time_dense, time_raw, self.loader.payload_w[:, 1]),
            np.interp(time_dense, time_raw, self.loader.payload_w[:, 2]),
        ])

        self.save_payload_csv(
            time_dense,
            payload_pos_s,
            payload_vel_s,
            payload_acc_s,
            payload_jerk_s,
            payload_snap_s,
            payload_q_s,
            payload_w_s,
        )

        # Cable data (ENU)
        cable_dir_raw = self.loader.cable_direction
        (
            cable_dir_s,
            _,
            _,
            _,
            _,
        ) = self.resample_polynomial(cable_dir_raw, time_raw, time_dense)
        cable_dir_s = self.normalize_vectors(cable_dir_s)

        cable_omega_s = self.resample_vector_linear(self.loader.cable_omega, time_raw, time_dense)
        cable_omega_dot_s = self.resample_vector_linear(
            self.loader.cable_omega_dot, time_raw, time_dense
        )
        cable_mu_s = self.resample_scalar_linear(self.loader.cable_mu, time_raw, time_dense)

        self.save_cable_csv(
            time_dense,
            cable_dir_s,
            cable_omega_s,
            cable_omega_dot_s,
            cable_mu_s,
        )

        # Kfb gains (upsampled to match time_dense)
        kfb_path = self.loader.path / "Kfb_traj_0_3_a_3.npy"
        if not kfb_path.exists():
            print(f"ERROR: Kfb file not found: {kfb_path}", file=sys.stderr)
            raise FileNotFoundError(f"Kfb file not found: {kfb_path}")
        kfb_raw = np.load(kfb_path, allow_pickle=True)
        if kfb_raw.ndim != 3 or kfb_raw.shape[1:] != (6, 13):
            raise ValueError(f"Unexpected Kfb shape: {kfb_raw.shape}")
        time_k = np.arange(kfb_raw.shape[0]) * self.dt
        kfb_flat = kfb_raw.reshape(kfb_raw.shape[0], -1)
        kfb_interp = np.zeros((len(time_dense), kfb_flat.shape[1]))
        for col in range(kfb_flat.shape[1]):
            kfb_interp[:, col] = np.interp(time_dense, time_k, kfb_flat[:, col])
        kfb_out = kfb_interp.reshape(len(time_dense), 6, 13)
        self.save_kfb_csv(time_dense, kfb_out, self.output_dir / "kfb.csv")

        print(f"Wrote converted data to {self.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert new trajectory to CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/realflight_traj_new"),
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--target-dt",
        type=float,
        default=0.01,
        help="Target sample period for smoothed outputs",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=8,
        help="Polynomial fit window (raw samples)",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=7,
        help="Polynomial fit order",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    converter = TrajectoryConverterNew(
        args.output_dir,
        target_dt=args.target_dt,
        window=args.window,
        order=args.order,
    )
    converter.run()


if __name__ == "__main__":
    main()

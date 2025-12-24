#!/usr/bin/env python3
"""
Convert the vertical COM_Dyn_V trajectory to CSV for:
- Per-drone follow trajectories (NED, smoothed 100 Hz).
- Geometric controller data (payload/cable/Kfb in ENU at 100 Hz).

Column naming follows tools/save_traj_vertical.py.
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

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

    def __init__(self, scenario_dir: Optional[Path] = None):
        if scenario_dir is None:
            scenario_dir = SCENARIO_DIR
        scenario_dir = Path(scenario_dir)
        if not scenario_dir.is_absolute():
            scenario_dir = REPO_ROOT / scenario_dir
        if not scenario_dir.exists():
            raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")
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


class TrajectoryConverterVertical:
    def __init__(
        self,
        output_dir: Path,
        base_dir: Optional[Path] = None,
        target_dt: float = 0.01,
        window: int = 8,
        order: int = 7,
    ) -> None:
        self.loader = DataLoaderVertical(base_dir)
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

    def resample_polynomial_series(
        self,
        series_raw: np.ndarray,
        time_raw: np.ndarray,
        time_dense: np.ndarray,
    ) -> np.ndarray:
        num_tracks, _, dims = series_raw.shape
        out = np.zeros((num_tracks, len(time_dense), dims))
        half_w = self.window // 2

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
                    end = min(len(time_raw), start + self.window)
                    start = max(0, end - self.window)
                    t_win = time_raw[start:end]
                    v_win = values[start:end]
                    order_fit = min(self.order, len(t_win) - 1)
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

    @staticmethod
    def quat_normalize(q: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(q)
        if norm < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / norm

    @staticmethod
    def quat_conjugate(q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def quat_multiply(q: np.ndarray, r: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q
        w2, x2, y2, z2 = r
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])

    def quat_inverse(self, q: np.ndarray) -> np.ndarray:
        q = self.quat_normalize(q)
        return self.quat_conjugate(q)

    @staticmethod
    def quat_log(q: np.ndarray) -> np.ndarray:
        w = np.clip(q[0], -1.0, 1.0)
        v = q[1:]
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-9:
            return np.zeros(4)
        angle = np.arctan2(v_norm, w)
        return np.array([0.0, *(v / v_norm * angle)])

    @staticmethod
    def quat_exp(q: np.ndarray) -> np.ndarray:
        v = q[1:]
        angle = np.linalg.norm(v)
        if angle < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return np.array([np.cos(angle), *(v / angle * np.sin(angle))])

    def quat_slerp(self, q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        q0 = self.quat_normalize(q0)
        q1 = self.quat_normalize(q1)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            return self.quat_normalize(q0 + t * (q1 - q0))
        theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        s0 = np.sin(theta_0 - theta) / sin_theta_0
        s1 = np.sin(theta) / sin_theta_0
        return s0 * q0 + s1 * q1

    def quat_squad(
        self, q0: np.ndarray, q1: np.ndarray, a0: np.ndarray, a1: np.ndarray, t: float
    ) -> np.ndarray:
        slerp_q = self.quat_slerp(q0, q1, t)
        slerp_a = self.quat_slerp(a0, a1, t)
        return self.quat_slerp(slerp_q, slerp_a, 2.0 * t * (1.0 - t))

    @staticmethod
    def slerp_vectors(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
        v0_norm = np.linalg.norm(v0)
        v1_norm = np.linalg.norm(v1)
        if v0_norm < 1e-9 or v1_norm < 1e-9:
            return v0
        v0_u = v0 / v0_norm
        v1_u = v1 / v1_norm
        dot = float(np.clip(np.dot(v0_u, v1_u), -1.0, 1.0))
        if dot > 0.9995:
            v = v0_u + t * (v1_u - v0_u)
            return v / np.clip(np.linalg.norm(v), 1e-9, None)
        if dot < -0.9995:
            axis = np.array([1.0, 0.0, 0.0]) if abs(v0_u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            ortho = np.cross(v0_u, axis)
            ortho /= np.clip(np.linalg.norm(ortho), 1e-9, None)
            return np.cos(np.pi * t) * v0_u + np.sin(np.pi * t) * ortho
        theta = np.arccos(dot)
        sin_theta = np.sin(theta)
        return (
            np.sin((1.0 - t) * theta) / sin_theta * v0_u
            + np.sin(t * theta) / sin_theta * v1_u
        )

    @staticmethod
    def sphere_log(base: np.ndarray, target: np.ndarray) -> np.ndarray:
        base_u = base / np.clip(np.linalg.norm(base), 1e-9, None)
        tgt_u = target / np.clip(np.linalg.norm(target), 1e-9, None)
        dot = float(np.clip(np.dot(base_u, tgt_u), -1.0, 1.0))
        if dot > 0.999999:
            return np.zeros(3)
        if dot < -0.999999:
            axis = np.array([1.0, 0.0, 0.0]) if abs(base_u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            ortho = np.cross(base_u, axis)
            ortho /= np.clip(np.linalg.norm(ortho), 1e-9, None)
            return ortho * np.pi
        v = tgt_u - dot * base_u
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-9:
            return np.zeros(3)
        theta = np.arccos(dot)
        return v * (theta / v_norm)

    @staticmethod
    def sphere_exp(base: np.ndarray, tangent: np.ndarray) -> np.ndarray:
        base_u = base / np.clip(np.linalg.norm(base), 1e-9, None)
        theta = np.linalg.norm(tangent)
        if theta < 1e-9:
            return base_u
        direction = tangent / theta
        return np.cos(theta) * base_u + np.sin(theta) * direction

    def sphere_squad(
        self,
        v0: np.ndarray,
        v1: np.ndarray,
        a0: np.ndarray,
        a1: np.ndarray,
        t: float,
    ) -> np.ndarray:
        slerp_v = self.slerp_vectors(v0, v1, t)
        slerp_a = self.slerp_vectors(a0, a1, t)
        return self.slerp_vectors(slerp_v, slerp_a, 2.0 * t * (1.0 - t))

    def resample_unit_vectors_squad(
        self,
        series_raw: np.ndarray,
        time_raw: np.ndarray,
        time_dense: np.ndarray,
    ) -> np.ndarray:
        series_norm = self.normalize_vectors(series_raw)
        num_tracks = series_norm.shape[0]
        out = np.zeros((num_tracks, len(time_dense), 3))
        if series_norm.shape[1] == 1:
            out[:] = series_norm[:, 0:1, :]
            return out

        ctrl = np.zeros_like(series_norm)
        ctrl[:, 0] = series_norm[:, 0]
        ctrl[:, -1] = series_norm[:, -1]
        for i in range(1, series_norm.shape[1] - 1):
            for n in range(num_tracks):
                log_prev = self.sphere_log(series_norm[n, i], series_norm[n, i - 1])
                log_next = self.sphere_log(series_norm[n, i], series_norm[n, i + 1])
                ctrl[n, i] = self.sphere_exp(series_norm[n, i], -0.25 * (log_prev + log_next))

        for n in range(num_tracks):
            for idx_t, t in enumerate(time_dense):
                if t <= time_raw[0]:
                    out[n, idx_t] = series_norm[n, 0]
                    continue
                if t >= time_raw[-1]:
                    out[n, idx_t] = series_norm[n, -1]
                    continue
                seg = np.searchsorted(time_raw, t) - 1
                seg = max(0, min(seg, series_norm.shape[1] - 2))
                t0 = time_raw[seg]
                t1 = time_raw[seg + 1]
                u = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
                out[n, idx_t] = self.sphere_squad(
                    series_norm[n, seg],
                    series_norm[n, seg + 1],
                    ctrl[n, seg],
                    ctrl[n, seg + 1],
                    u,
                )
        return out

    def resample_quaternion(
        self, q_raw: np.ndarray, time_raw: np.ndarray, time_dense: np.ndarray
    ) -> np.ndarray:
        q_cont = q_raw.copy()
        for i in range(1, q_cont.shape[0]):
            if np.dot(q_cont[i - 1], q_cont[i]) < 0.0:
                q_cont[i] *= -1.0
        q_cont = np.array([self.quat_normalize(q) for q in q_cont])
        if q_cont.shape[0] == 1:
            return np.repeat(q_cont, len(time_dense), axis=0)

        ctrl = np.zeros_like(q_cont)
        ctrl[0] = q_cont[0]
        ctrl[-1] = q_cont[-1]
        for i in range(1, q_cont.shape[0] - 1):
            q_inv = self.quat_inverse(q_cont[i])
            log1 = self.quat_log(self.quat_multiply(q_inv, q_cont[i - 1]))
            log2 = self.quat_log(self.quat_multiply(q_inv, q_cont[i + 1]))
            exp_term = self.quat_exp(-0.25 * (log1 + log2))
            ctrl[i] = self.quat_multiply(q_cont[i], exp_term)

        out = np.zeros((len(time_dense), 4))
        for idx_t, t in enumerate(time_dense):
            if t <= time_raw[0]:
                out[idx_t] = q_cont[0]
                continue
            if t >= time_raw[-1]:
                out[idx_t] = q_cont[-1]
                continue
            seg = np.searchsorted(time_raw, t) - 1
            seg = max(0, min(seg, q_cont.shape[0] - 2))
            t0 = time_raw[seg]
            t1 = time_raw[seg + 1]
            u = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            out[idx_t] = self.quat_squad(q_cont[seg], q_cont[seg + 1], ctrl[seg], ctrl[seg + 1], u)
        return out

    @staticmethod
    def extend_time_series(
        time_raw: np.ndarray, series_raw: np.ndarray, target_time: float, time_axis: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        if time_raw[-1] >= target_time:
            return time_raw, series_raw
        pad_slice = [slice(None)] * series_raw.ndim
        pad_slice[time_axis] = slice(-1, None)
        series_pad = series_raw[tuple(pad_slice)]
        series_ext = np.concatenate([series_raw, series_pad], axis=time_axis)
        time_ext = np.append(time_raw, target_time)
        return time_ext, series_ext

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
        payload_w_raw = self.loader.payload_w.reshape(1, -1, 3)
        payload_w_s = self.resample_polynomial_series(payload_w_raw, time_raw, time_dense)[0]

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
        cable_dir_s = self.resample_unit_vectors_squad(cable_dir_raw, time_raw, time_dense)

        cable_omega_s = self.resample_polynomial_series(self.loader.cable_omega, time_raw, time_dense)
        cable_omega_dot_s = self.resample_polynomial_series(
            self.loader.cable_omega_dot, time_raw, time_dense
        )
        cable_mu_raw = self.loader.cable_mu[:, :, None]
        cable_mu_s = self.resample_polynomial_series(cable_mu_raw, time_raw, time_dense)[:, :, 0]

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
        time_k, kfb_raw = self.extend_time_series(time_k, kfb_raw, time_dense_end, time_axis=0)
        kfb_flat = kfb_raw.reshape(kfb_raw.shape[0], -1)
        kfb_series = kfb_flat[None, :, :]
        kfb_resampled = self.resample_polynomial_series(kfb_series, time_k, time_dense)[0]
        kfb_out = kfb_resampled.reshape(len(time_dense), 6, 13)
        self.save_kfb_csv(time_dense, kfb_out, self.output_dir / "kfb.csv")

        print(f"Wrote converted data to {self.output_dir}")


def scenario_suffix(scenario_dir: Path) -> str:
    marker = "COM_Dyn"
    name = scenario_dir.name
    if marker not in name:
        return ""
    suffix = name.split(marker, 1)[1]
    return suffix


def resolve_scenario_dir(base_dir: Optional[Path]) -> Path:
    if base_dir is None:
        return SCENARIO_DIR
    base_dir = Path(base_dir)
    if base_dir.is_absolute():
        return base_dir
    return REPO_ROOT / base_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert vertical trajectory to CSV")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Raw-data scenario directory (default: COM_Dyn_V vertical data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for CSV files (default: data/realflight_traj_vertical + scenario suffix)",
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
    scenario_dir = resolve_scenario_dir(args.base_dir)
    output_dir = args.output_dir
    if output_dir is None:
        base = Path("data/realflight_traj_vertical")
        suffix = scenario_suffix(scenario_dir)
        output_dir = base.parent / f"{base.name}{suffix}"
    converter = TrajectoryConverterVertical(
        output_dir,
        base_dir=scenario_dir,
        target_dt=args.target_dt,
        window=args.window,
        order=args.order,
    )
    converter.run()


if __name__ == "__main__":
    main()

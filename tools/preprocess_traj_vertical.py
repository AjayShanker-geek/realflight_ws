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
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FILE_DIR.parent
CONFIG_PATH = FILE_DIR / "preprocess_traj_vertical.yaml"
SCENARIO_DIR = (
    REPO_ROOT
    / Path("raw_data/Planning_plots_multiagent_meta_evaluation_COM_Dyn_V (m2=0.1,rg=[0.03,0],4.6rq)")
)


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


def _parse_float(config: dict, key: str) -> Optional[float]:
    value = config.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_bool(config: dict, key: str, default: bool = False) -> bool:
    value = config.get(key)
    if value is None:
        return default
    val = str(value).strip().lower()
    if val in ("1", "true", "yes", "y", "on"):
        return True
    if val in ("0", "false", "no", "n", "off"):
        return False
    return default


def _parse_int(config: dict, key: str) -> Optional[int]:
    value = config.get(key)
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        return None


class DataLoaderVertical:
    """
    Load planned trajectory data for the COM_Dyn_V evaluation.
    """

    @staticmethod
    def _extract_suffix(file_path: Path, prefix: str) -> str:
        name = file_path.name
        prefix_with_sep = f"{prefix}_"
        if name.startswith(prefix_with_sep):
            return name[len(prefix_with_sep):]
        return name

    def _find_traj_file(
        self,
        prefix: str,
        num_drones: Optional[int],
        suffix_hint: Optional[str] = None,
    ) -> Path:
        """
        Locate the first matching trajectory file for the given prefix.
        Priority: exact suffix_hint (if provided) -> num_drones-specific -> any.
        """
        if suffix_hint:
            suffix_clean = suffix_hint
            if suffix_clean.startswith(f"{prefix}_"):
                suffix_clean = suffix_clean[len(prefix) + 1 :]
            suffix_clean = suffix_clean.lstrip("_")
            if not suffix_clean.endswith(".npy"):
                suffix_clean = f"{suffix_clean}.npy"
            candidate = self.path / f"{prefix}_{suffix_clean}"
            if candidate.exists():
                return candidate
        patterns = []
        if num_drones is not None:
            patterns.append(f"{prefix}_*_a_{num_drones}.npy")
        patterns.append(f"{prefix}_*_a_*.npy")
        for pattern in patterns:
            matches = sorted(self.path.glob(pattern))
            if matches:
                return matches[0]
        hint = suffix_hint or f"* (num_drones={num_drones or 'any'})"
        raise FileNotFoundError(f"No {prefix} file matching {hint} in {self.path}")

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
        self.payload_attitude_identity = False

        config = read_config(CONFIG_PATH)
        self.num_drones_config = _parse_int(config, "num_drones")
        self.traj_suffix = None
        dt_val = _parse_float(config, "dt")
        if dt_val is not None:
            self.dt = dt_val
        cl0_val = _parse_float(config, "cl0")
        if cl0_val is not None:
            self.cable_length = cl0_val
        payload_radius_val = _parse_float(config, "payload_radius")
        rl_val = _parse_float(config, "rl")
        self.payload_attitude_identity = _parse_bool(config, "payload_attitude_identity", False)
        if payload_radius_val is not None:
            self.rl = payload_radius_val
        elif rl_val is not None:
            self.rl = rl_val

        xq_file = self._find_traj_file("xq_traj", self.num_drones_config, config.get("traj_suffix"))
        self.traj_suffix = self._extract_suffix(xq_file, "xq_traj")
        self.xq_traj = np.load(xq_file, allow_pickle=True)
        self.num_drones = self.xq_traj.shape[0]
        if self.num_drones_config is not None and self.num_drones_config != self.num_drones:
            raise ValueError(
                f"Configured num_drones={self.num_drones_config} but data has {self.num_drones} drones ({xq_file.name})"
            )
        self.alpha = 2 * math.pi / self.num_drones
        self.cable_direction = self.xq_traj[:, :, 0:3]
        self.cable_omega = self.xq_traj[:, :, 3:6]
        self.cable_omega_dot = self.xq_traj[:, :, 6:9]
        self.cable_mu = self.xq_traj[:, :, 12]
        self.cable_mu_dot = self.xq_traj[:, :, 13]
        self.kfb_path = self._find_traj_file("Kfb_traj", self.num_drones, self.traj_suffix)

        xl_file = self._find_traj_file("xl_traj", self.num_drones, self.traj_suffix)
        self.xl_traj = np.load(xl_file, allow_pickle=True)
        self.payload_x = self.xl_traj[:, 0:3]
        self.payload_v = self.xl_traj[:, 3:6]
        self.payload_q = self.xl_traj[:, 6:10]
        self.payload_w = self.xl_traj[:, 10:13]


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
        self._fact = [math.factorial(i) for i in range(self.order + 1)]
        self._c5_inv = self._build_c5_inv()
        self.payload_attitude_identity = self.loader.payload_attitude_identity
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

    def compute_drone_kinematics(
        self,
        payload_pos: np.ndarray,
        payload_vel: np.ndarray,
        payload_acc: np.ndarray,
        payload_q: np.ndarray,
        payload_omega: np.ndarray,
        payload_omega_dot: np.ndarray,
        cable_dir: np.ndarray,
        cable_omega: np.ndarray,
        cable_omega_dot: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r_attach_body = np.array(
            [
                [self.rl * np.cos(i * self.alpha), self.rl * np.sin(i * self.alpha), 0.0]
                for i in range(self.num_drones)
            ]
        )
        rot = self.quat_to_rotmat(payload_q)
        r_attach_world = np.einsum("tij,dj->tdi", rot, r_attach_body)
        r_attach_world = np.transpose(r_attach_world, (1, 0, 2))

        omega_world = np.einsum("tij,tj->ti", rot, payload_omega)
        omega_dot_world = np.einsum("tij,tj->ti", rot, payload_omega_dot)

        qdot = np.cross(cable_omega, cable_dir, axis=2)
        qddot = np.cross(cable_omega_dot, cable_dir, axis=2) + np.cross(cable_omega, qdot, axis=2)

        omega_world_b = omega_world[None, :, :]
        omega_dot_world_b = omega_dot_world[None, :, :]
        omega_cross_r = np.cross(omega_world_b, r_attach_world, axis=2)
        omega_cross_omega_cross_r = np.cross(omega_world_b, omega_cross_r, axis=2)
        omega_dot_cross_r = np.cross(omega_dot_world_b, r_attach_world, axis=2)

        pos = payload_pos[None, :, :] + r_attach_world + self.cable_length * cable_dir
        vel = payload_vel[None, :, :] + omega_cross_r + self.cable_length * qdot
        acc = payload_acc[None, :, :] + omega_dot_cross_r + omega_cross_omega_cross_r + self.cable_length * qddot
        return pos, vel, acc

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
    def _build_c5_inv() -> np.ndarray:
        order = 11
        size = order + 1
        mat = np.zeros((size, size), dtype=float)
        for k in range(6):
            mat[k, k] = math.factorial(k)
        for k in range(6):
            row = 6 + k
            for m in range(k, order + 1):
                mat[row, m] = math.factorial(m) / math.factorial(m - k)
        return np.linalg.inv(mat)

    def estimate_derivatives_raw(
        self,
        series_raw: np.ndarray,
        time_raw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        num_tracks, _, _ = series_raw.shape
        pos_out = np.zeros((num_tracks, len(time_raw), 3))
        vel_out = np.zeros_like(pos_out)
        acc_out = np.zeros_like(pos_out)
        jerk_out = np.zeros_like(pos_out)
        snap_out = np.zeros_like(pos_out)
        crackle_out = np.zeros_like(pos_out)

        half_w = self.window // 2
        for i in range(num_tracks):
            for axis in range(3):
                values = series_raw[i, :, axis]
                for idx_t, t in enumerate(time_raw):
                    raw_idx = idx_t
                    center = raw_idx
                    start = max(0, center - half_w)
                    end = min(len(time_raw), start + self.window)
                    start = max(0, end - self.window)
                    t_win = time_raw[start:end]
                    v_win = values[start:end]
                    order_fit = min(self.order, len(t_win) - 1)
                    if order_fit < 1:
                        pos_out[i, idx_t, axis] = values[raw_idx]
                        continue
                    t_shift = t_win - t
                    coeffs = np.polyfit(t_shift, v_win, order_fit)
                    pos_out[i, idx_t, axis] = values[raw_idx]
                    vel_out[i, idx_t, axis] = np.polyval(np.polyder(coeffs, 1), 0.0) if order_fit >= 1 else 0.0
                    acc_out[i, idx_t, axis] = np.polyval(np.polyder(coeffs, 2), 0.0) if order_fit >= 2 else 0.0
                    jerk_out[i, idx_t, axis] = np.polyval(np.polyder(coeffs, 3), 0.0) if order_fit >= 3 else 0.0
                    snap_out[i, idx_t, axis] = np.polyval(np.polyder(coeffs, 4), 0.0) if order_fit >= 4 else 0.0
                    crackle_out[i, idx_t, axis] = np.polyval(np.polyder(coeffs, 5), 0.0) if order_fit >= 5 else 0.0

        return pos_out, vel_out, acc_out, jerk_out, snap_out, crackle_out

    def resample_piecewise_c5(
        self,
        series_raw: np.ndarray,
        time_raw: np.ndarray,
        time_dense: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (
            pos_raw,
            vel_raw,
            acc_raw,
            jerk_raw,
            snap_raw,
            crackle_raw,
        ) = self.estimate_derivatives_raw(series_raw, time_raw)

        num_tracks, _, _ = series_raw.shape
        pos_out = np.zeros((num_tracks, len(time_dense), 3))
        vel_out = np.zeros_like(pos_out)
        acc_out = np.zeros_like(pos_out)
        jerk_out = np.zeros_like(pos_out)
        snap_out = np.zeros_like(pos_out)

        for seg in range(len(time_raw) - 1):
            t0 = time_raw[seg]
            t1 = time_raw[seg + 1]
            h = t1 - t0
            if h <= 0.0:
                continue
            idx_start = int(np.searchsorted(time_dense, t0, side="left"))
            if seg == len(time_raw) - 2:
                idx_end = int(np.searchsorted(time_dense, t1, side="right"))
            else:
                idx_end = int(np.searchsorted(time_dense, t1, side="left"))
            if idx_end <= idx_start:
                continue

            tau = (time_dense[idx_start:idx_end] - t0) / h
            tau_powers = np.vstack([tau ** k for k in range(12)])
            tau_powers1 = tau_powers[:-1]
            tau_powers2 = tau_powers[:-2]
            tau_powers3 = tau_powers[:-3]
            tau_powers4 = tau_powers[:-4]

            d0 = np.stack([pos_raw[:, seg, :],
                           vel_raw[:, seg, :],
                           acc_raw[:, seg, :],
                           jerk_raw[:, seg, :],
                           snap_raw[:, seg, :],
                           crackle_raw[:, seg, :]], axis=-1)
            d1 = np.stack([pos_raw[:, seg + 1, :],
                           vel_raw[:, seg + 1, :],
                           acc_raw[:, seg + 1, :],
                           jerk_raw[:, seg + 1, :],
                           snap_raw[:, seg + 1, :],
                           crackle_raw[:, seg + 1, :]], axis=-1)
            scale = np.array([1.0, h, h ** 2, h ** 3, h ** 4, h ** 5], dtype=float)
            b0 = d0 * scale
            b1 = d1 * scale
            b = np.concatenate([b0, b1], axis=-1)
            coeff = np.tensordot(b, self._c5_inv.T, axes=([2], [0]))

            c1 = np.zeros((num_tracks, 3, 11))
            c2 = np.zeros((num_tracks, 3, 10))
            c3 = np.zeros((num_tracks, 3, 9))
            c4 = np.zeros((num_tracks, 3, 8))
            for m in range(1, 12):
                c1[..., m - 1] = m * coeff[..., m]
            for m in range(2, 12):
                c2[..., m - 2] = m * (m - 1) * coeff[..., m]
            for m in range(3, 12):
                c3[..., m - 3] = m * (m - 1) * (m - 2) * coeff[..., m]
            for m in range(4, 12):
                c4[..., m - 4] = m * (m - 1) * (m - 2) * (m - 3) * coeff[..., m]

            pos_seg = np.tensordot(coeff, tau_powers, axes=([2], [0]))
            vel_seg = np.tensordot(c1, tau_powers1, axes=([2], [0])) / h
            acc_seg = np.tensordot(c2, tau_powers2, axes=([2], [0])) / (h ** 2)
            jerk_seg = np.tensordot(c3, tau_powers3, axes=([2], [0])) / (h ** 3)
            snap_seg = np.tensordot(c4, tau_powers4, axes=([2], [0])) / (h ** 4)

            pos_out[:, idx_start:idx_end, :] = np.transpose(pos_seg, (0, 2, 1))
            vel_out[:, idx_start:idx_end, :] = np.transpose(vel_seg, (0, 2, 1))
            acc_out[:, idx_start:idx_end, :] = np.transpose(acc_seg, (0, 2, 1))
            jerk_out[:, idx_start:idx_end, :] = np.transpose(jerk_seg, (0, 2, 1))
            snap_out[:, idx_start:idx_end, :] = np.transpose(snap_seg, (0, 2, 1))

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

    @staticmethod
    def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
        q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-9, None)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        rot = np.empty((q.shape[0], 3, 3))
        rot[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
        rot[:, 0, 1] = 2.0 * (x * y - z * w)
        rot[:, 0, 2] = 2.0 * (x * z + y * w)
        rot[:, 1, 0] = 2.0 * (x * y + z * w)
        rot[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
        rot[:, 1, 2] = 2.0 * (y * z - x * w)
        rot[:, 2, 0] = 2.0 * (x * z - y * w)
        rot[:, 2, 1] = 2.0 * (y * z + x * w)
        rot[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
        return rot

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
                norm_yb = np.linalg.norm(y_b)
                if norm_yb < 1e-6:
                    fallback = x_c if abs(np.dot(z_b, x_c)) < 0.9 else y_c
                    y_b = np.cross(z_b, fallback)
                    norm_yb = np.linalg.norm(y_b)
                    if norm_yb < 1e-6:
                        y_b = np.array([0.0, 1.0, 0.0])
                        norm_yb = 1.0
                y_b /= norm_yb
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

        payload_pos_raw_full = self.loader.payload_x
        payload_vel_raw_full = self.loader.payload_v
        payload_q_raw = self.loader.payload_q
        payload_w_raw_full = self.loader.payload_w
        payload_acc_raw = np.gradient(payload_vel_raw_full, self.dt, axis=0, edge_order=2)
        payload_w_dot_raw = np.gradient(payload_w_raw_full, self.dt, axis=0, edge_order=2)

        if self.payload_attitude_identity:
            payload_q_raw = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (payload_pos_raw_full.shape[0], 1))
            payload_w_raw_full = np.zeros_like(payload_w_raw_full)
            payload_w_dot_raw = np.zeros_like(payload_w_dot_raw)

        # Payload trajectory (ENU) for geometric controller
        payload_pos_raw = payload_pos_raw_full.reshape(1, -1, 3)
        (
            payload_pos_s,
            payload_vel_s,
            payload_acc_s,
            payload_jerk_s,
            payload_snap_s,
        ) = self.resample_piecewise_c5(payload_pos_raw, time_raw, time_dense)

        payload_pos_s = payload_pos_s[0]
        payload_vel_s = payload_vel_s[0]
        payload_acc_s = payload_acc_s[0]
        payload_jerk_s = payload_jerk_s[0]
        payload_snap_s = payload_snap_s[0]

        payload_q_s = self.resample_quaternion(payload_q_raw, time_raw, time_dense)
        payload_w_raw = payload_w_raw_full.reshape(1, -1, 3)
        payload_w_s, payload_w_dot_s, _, _, _ = self.resample_polynomial(
            payload_w_raw, time_raw, time_dense
        )
        payload_w_s = payload_w_s[0]
        payload_w_dot_s = payload_w_dot_s[0]
        if self.payload_attitude_identity:
            payload_q_s = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(time_dense), 1))
            payload_w_s = np.zeros_like(payload_w_s)
            payload_w_dot_s = np.zeros_like(payload_w_dot_s)

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

        # Per-drone trajectories (NED) via analytic kinematics at raw rate + C5 resample
        drone_pos_raw, _, _ = self.compute_drone_kinematics(
            payload_pos_raw_full,
            payload_vel_raw_full,
            payload_acc_raw,
            payload_q_raw,
            payload_w_raw_full,
            payload_w_dot_raw,
            cable_dir_raw,
            self.loader.cable_omega,
            self.loader.cable_omega_dot,
        )
        (
            pos_smooth_enu,
            vel_smooth_enu,
            acc_smooth_enu,
            jerk_smooth_enu,
            snap_smooth_enu,
        ) = self.resample_piecewise_c5(drone_pos_raw, time_raw, time_dense)

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

        vel_diff_enu = np.gradient(pos_smooth_enu, self.target_dt, axis=1)
        acc_diff_enu = np.gradient(vel_diff_enu, self.target_dt, axis=1)
        jerk_diff_enu = np.gradient(acc_diff_enu, self.target_dt, axis=1)
        snap_diff_enu = np.gradient(jerk_diff_enu, self.target_dt, axis=1)

        vel_diff = self.enu_to_ned(vel_diff_enu)
        acc_diff = self.enu_to_ned(acc_diff_enu)
        jerk_diff = self.enu_to_ned(jerk_diff_enu)
        snap_diff = self.enu_to_ned(snap_diff_enu)

        roll_d, pitch_d, yaw_d, p_d, q_d, r_d = self.compute_attitude_and_body_rates(
            acc_diff, jerk_diff
        )
        self.save_drone_csv(
            pos_smooth,
            vel_diff,
            time_dense,
            suffix="smoothed_100hz_diff",
            accel=acc_diff,
            jerk=jerk_diff,
            snap=snap_diff,
            roll=roll_d,
            pitch=pitch_d,
            yaw=yaw_d,
            p_rates=p_d,
            q_rates=q_d,
            r_rates=r_d,
        )

        # Kfb gains (upsampled to match time_dense)
        kfb_path = self.loader.kfb_path
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
    config = read_config(CONFIG_PATH)
    scenario_dir = resolve_scenario_dir(args.base_dir)
    if args.base_dir is None:
        config_base = config.get("base_dir")
        if config_base:
            scenario_dir = resolve_scenario_dir(config_base)
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

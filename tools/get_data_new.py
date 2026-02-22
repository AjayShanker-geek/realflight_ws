#!/usr/bin/env python3
"""
Data loader for COM_Dyn_H preprocessing.

This keeps the same public interface used by preprocess scripts while mirroring
file/config handling from preprocess_traj_vertical.py.
"""

import math
import re
from pathlib import Path
from typing import Optional

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "tools" / "preprocess_traj_new.yaml"


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


def _parse_int(config: dict, key: str) -> Optional[int]:
    value = config.get(key)
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def _parse_vec3_literal(value: str) -> Optional[np.ndarray]:
    nums = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)]
    if len(nums) < 2:
        return None
    if len(nums) == 2:
        nums.append(0.0)
    return np.array(nums[:3], dtype=float)


def _parse_vec3(config: dict, key: str) -> Optional[np.ndarray]:
    value = config.get(key)
    if value is None:
        return None
    return _parse_vec3_literal(str(value))


def resolve_default_scenario_dir(repo_root: Path) -> Path:
    raw_root = repo_root / "raw_data"
    patterns = [
        "Planning_plots_stage2_COM_Dyn_H*",
        "Planning_plots_multiagent_meta_evaluation_COM_Dyn_H*",
    ]
    for pattern in patterns:
        matches = sorted(raw_root.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No COM_Dyn_H scenario directories found under {raw_root} "
        f"(checked: {patterns})"
    )


def resolve_scenario_dir(
    base_dir: Optional[Path] = None,
    config_path: Optional[Path] = None,
    repo_root: Path = BASE_DIR,
) -> Path:
    if base_dir is not None:
        scenario_dir = Path(base_dir)
        if not scenario_dir.is_absolute():
            scenario_dir = repo_root / scenario_dir
        return scenario_dir

    config = read_config(config_path or CONFIG_PATH)
    config_base = config.get("base_dir")
    if config_base:
        scenario_dir = Path(config_base)
        if not scenario_dir.is_absolute():
            scenario_dir = repo_root / scenario_dir
        return scenario_dir

    return resolve_default_scenario_dir(repo_root)


class DataLoader:
    """
    Load planned trajectory data for COM_Dyn_H evaluation.
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
            patterns.append(f"{prefix}_*_{num_drones}.npy")
        patterns.append(f"{prefix}_*_a_*.npy")
        patterns.append(f"{prefix}_*.npy")

        for pattern in patterns:
            matches = sorted(self.path.glob(pattern))
            if matches:
                return matches[0]

        hint = suffix_hint or f"* (num_drones={num_drones or 'any'})"
        raise FileNotFoundError(f"No {prefix} file matching {hint} in {self.path}")

    def __init__(self, scenario_dir: Optional[Path] = None):
        if scenario_dir is None:
            scenario_dir = resolve_scenario_dir(None, config_path=CONFIG_PATH, repo_root=BASE_DIR)
        scenario_dir = Path(scenario_dir)
        if not scenario_dir.is_absolute():
            scenario_dir = BASE_DIR / scenario_dir
        if not scenario_dir.exists():
            raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")
        self.path = scenario_dir

        # Defaults aligned with COM_Dyn_H config.
        self.dt = 0.05
        self.rl = 0.175
        self.cable_length = 1.0
        self.g = 9.81
        self.ez = np.array([0.0, 0.0, 1.0]).reshape(3, 1)
        self.payload_attitude_identity = False
        self.rg = np.zeros(3, dtype=float)

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
        if payload_radius_val is not None:
            self.rl = payload_radius_val
        elif rl_val is not None:
            self.rl = rl_val

        m1 = _parse_float(config, "m1")
        m2 = _parse_float(config, "m2")
        rp = _parse_vec3(config, "rp")
        if m1 is None or m2 is None or rp is None:
            raise ValueError(
                "Missing CoM parameters in preprocess_traj_new.yaml. "
                "Please set m1, m2, and rp: [x, y, z]."
            )
        total_mass = m1 + m2
        if total_mass <= 1e-9:
            raise ValueError(f"Invalid masses: m1 + m2 must be > 0 (got m1={m1}, m2={m2})")
        self.rg = (m2 / total_mass) * rp

        xq_file = self._find_traj_file("xq_traj", self.num_drones_config, config.get("traj_suffix"))
        self.traj_suffix = self._extract_suffix(xq_file, "xq_traj")
        self.xq_traj = np.load(xq_file, allow_pickle=True)
        if self.xq_traj.ndim != 3 or self.xq_traj.shape[2] < 9:
            raise ValueError(f"Unexpected xq_traj shape: {self.xq_traj.shape}")

        self.num_drones = self.xq_traj.shape[0]
        if self.num_drones_config is not None and self.num_drones_config != self.num_drones:
            raise ValueError(
                f"Configured num_drones={self.num_drones_config} but data has {self.num_drones} drones "
                f"({xq_file.name})"
            )
        self.alpha = 2 * math.pi / self.num_drones

        self.cable_direction = self.xq_traj[:, :, 0:3]
        self.cable_omega = self.xq_traj[:, :, 3:6]
        self.cable_omega_dot = self.xq_traj[:, :, 6:9]
        self.cable_mu = self.xq_traj[:, :, 12] if self.xq_traj.shape[2] > 12 else np.zeros(self.xq_traj.shape[:2])
        self.cable_mu_dot = (
            self.xq_traj[:, :, 13] if self.xq_traj.shape[2] > 13 else np.zeros(self.xq_traj.shape[:2])
        )

        xl_file = self._find_traj_file("xl_traj", self.num_drones, self.traj_suffix)
        self.xl_traj = np.load(xl_file, allow_pickle=True)
        if self.xl_traj.ndim != 2 or self.xl_traj.shape[1] < 13:
            raise ValueError(f"Unexpected xl_traj shape: {self.xl_traj.shape}")
        self.payload_x = self.xl_traj[:, 0:3]
        self.payload_v = self.xl_traj[:, 3:6]
        self.payload_q = self.xl_traj[:, 6:10]
        self.payload_w = self.xl_traj[:, 10:13]

        self.kfb_path = self._find_traj_file("Kfb_traj", self.num_drones, self.traj_suffix)
        kfb_raw = np.load(self.kfb_path, allow_pickle=True)
        if kfb_raw.ndim != 3 or kfb_raw.shape[1:] != (6, 13):
            raise ValueError(f"Unexpected Kfb shape: {kfb_raw.shape}")
        self.Kb = kfb_raw

        # Compatibility placeholders with legacy code paths.
        self.uq_traj = np.zeros((self.num_drones, self.xq_traj.shape[1], 3))

    def get_drone_pos(self) -> np.ndarray:
        """
        Compute initial drone positions in ENU using first payload sample.
        """
        ri = np.array(
            [
                [self.rl * math.cos(i * self.alpha), self.rl * math.sin(i * self.alpha), 0.0]
                for i in range(self.num_drones)
            ],
            dtype=float,
        )
        ri = ri - self.rg[None, :]
        return self.payload_x[0][None, :] + ri + self.cable_length * self.cable_direction[:, 0, :]


if __name__ == "__main__":
    loader = DataLoader()
    print("Scenario:", loader.path)
    print("Payload trajectory:", loader.payload_x.shape)
    print("Cable direction:", loader.cable_direction.shape)
    print("Kb shape:", loader.Kb.shape)

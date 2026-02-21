#!/usr/bin/env python3
"""
Diagnose ENU->NED conversion consistency for vertical multi-drone trajectories.

This script now reads `tools/preprocess_traj_vertical.yaml` by default and compares
raw planner inputs (`xq_traj_*.npy`, `xl_traj_*.npy`) against converted CSV outputs.
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Optional

import numpy as np


FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FILE_DIR.parent
DEFAULT_CONFIG = FILE_DIR / "preprocess_traj_vertical.yaml"
DEFAULT_SCENARIO = REPO_ROOT / "raw_data/Planning_plots_stage2_COM_Dyn_6_rpx5"


def read_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    config = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9_]+)\s*:\s*(.+)$", line)
        if m:
            config[m.group(1)] = m.group(2).strip().strip("\"'")
    return config


def parse_vec3(text: str) -> np.ndarray:
    vals = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]
    if len(vals) == 2:
        vals.append(0.0)
    if len(vals) != 3:
        raise ValueError(f"Expected 3 numbers, got: {text}")
    return np.array(vals, dtype=float)


def parse_float(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def parse_int(v: Optional[str]) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def resolve_path(p: Path) -> Path:
    return p if p.is_absolute() else REPO_ROOT / p


def scenario_suffix(scenario_dir: Path) -> str:
    marker = "COM_Dyn"
    name = scenario_dir.name
    if marker not in name:
        return ""
    return name.split(marker, 1)[1]


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-9, None)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rot = np.empty((q.shape[0], 3, 3), dtype=float)
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


def enu_to_ned(arr: np.ndarray) -> np.ndarray:
    t = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=float)
    flat = arr.reshape(-1, 3)
    out = (t @ flat.T).T
    return out.reshape(arr.shape)


def attachment_points_body(num_drones: int, rl: float, rg: np.ndarray) -> np.ndarray:
    alpha = 2.0 * np.pi / float(num_drones)
    points = np.array(
        [[rl * np.cos(i * alpha), rl * np.sin(i * alpha), 0.0] for i in range(num_drones)],
        dtype=float,
    )
    return points - rg[None, :]


def load_drone_positions(csv_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    paths = sorted(csv_dir.glob("drone_*_traj_smoothed_100hz.csv"))
    if not paths:
        raise FileNotFoundError(f"No drone_*_traj_smoothed_100hz.csv found in {csv_dir}")

    def drone_id(path: Path) -> int:
        m = re.match(r"drone_(\d+)_traj_smoothed_100hz\.csv$", path.name)
        if not m:
            raise ValueError(f"Unexpected file name: {path.name}")
        return int(m.group(1))

    pairs = sorted([(drone_id(p), p) for p in paths], key=lambda x: x[0])

    all_xyz = []
    time = None
    for _, path in pairs:
        with path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        t = np.array([float(r["time"]) for r in rows], dtype=float)
        xyz = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=float)
        if time is None:
            time = t
        elif len(time) != len(t) or not np.allclose(time, t, atol=1e-9):
            raise ValueError(f"Inconsistent time arrays in CSVs (failed at {path.name})")
        all_xyz.append(xyz)

    return time, np.stack(all_xyz, axis=0)


def load_payload_positions(csv_dir: Path) -> Optional[tuple[np.ndarray, np.ndarray]]:
    path = csv_dir / "payload.csv"
    if not path.exists():
        return None
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["time"]) for r in rows], dtype=float)
    pos = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=float)
    return t, pos


def load_cable_directions(csv_dir: Path) -> Optional[tuple[np.ndarray, np.ndarray]]:
    paths = sorted(csv_dir.glob("cable_*.csv"))
    if not paths:
        return None

    def cable_id(path: Path) -> int:
        m = re.match(r"cable_(\d+)\.csv$", path.name)
        if not m:
            raise ValueError(f"Unexpected cable file name: {path.name}")
        return int(m.group(1))

    pairs = sorted([(cable_id(p), p) for p in paths], key=lambda x: x[0])
    all_dirs = []
    time = None
    for _, path in pairs:
        with path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        t = np.array([float(r["time"]) for r in rows], dtype=float)
        d = np.array(
            [[float(r["dir_x"]), float(r["dir_y"]), float(r["dir_z"])] for r in rows],
            dtype=float,
        )
        if time is None:
            time = t
        elif len(time) != len(t) or not np.allclose(time, t, atol=1e-9):
            raise ValueError(f"Inconsistent time arrays in cable CSVs (failed at {path.name})")
        all_dirs.append(d)
    return time, np.stack(all_dirs, axis=0)


def raw_indices(csv_time: np.ndarray, n_raw: int, dt_raw: float = 0.05) -> np.ndarray:
    targets = np.arange(n_raw, dtype=float) * dt_raw
    idx = np.array([int(np.argmin(np.abs(csv_time - t))) for t in targets], dtype=int)
    miss = np.max(np.abs(csv_time[idx] - targets))
    if miss > 1e-6:
        raise ValueError(f"CSV time grid does not include expected raw times (max miss {miss:.3e})")
    return idx


def compute_drone_positions_enu(xq: np.ndarray, xl: np.ndarray, rl: float, cl0: float, rg: np.ndarray) -> np.ndarray:
    num_drones = xq.shape[0]
    payload_pos = xl[:, 0:3]
    payload_q = xl[:, 6:10]
    cable_dir = xq[:, :, 0:3]

    ri_body = attachment_points_body(num_drones, rl, rg)
    rot = quat_to_rotmat(payload_q)
    ri_world = np.einsum("tij,dj->tdi", rot, ri_body)
    ri_world = np.transpose(ri_world, (1, 0, 2))
    return payload_pos[None, :, :] + ri_world + cl0 * cable_dir


def list_suffixes(scenario_dir: Path) -> list[str]:
    suffixes = []
    for p in sorted(scenario_dir.glob("xq_traj_*.npy")):
        suffix = p.name[len("xq_traj_") :]
        if (scenario_dir / f"xl_traj_{suffix}").exists():
            suffixes.append(suffix)
    return suffixes


def normalize_suffix(text: str) -> str:
    out = text
    if out.startswith("xq_traj_"):
        out = out[len("xq_traj_") :]
    out = out.lstrip("_")
    if not out.endswith(".npy"):
        out = f"{out}.npy"
    return out


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err * err)))


def evaluate_suffix(
    scenario_dir: Path,
    suffix: str,
    drone_csv_time: np.ndarray,
    drone_csv_pos: np.ndarray,
    payload_csv: Optional[tuple[np.ndarray, np.ndarray]],
    cable_csv: Optional[tuple[np.ndarray, np.ndarray]],
    rl: float,
    cl0: float,
    rg: np.ndarray,
    dt_raw: float,
) -> dict:
    xq = np.load(scenario_dir / f"xq_traj_{suffix}", allow_pickle=True)
    xl = np.load(scenario_dir / f"xl_traj_{suffix}", allow_pickle=True)

    if xq.ndim != 3 or xq.shape[2] < 3:
        raise ValueError(f"Unexpected xq shape for suffix {suffix}: {xq.shape}")
    if xl.ndim != 2 or xl.shape[1] < 10:
        raise ValueError(f"Unexpected xl shape for suffix {suffix}: {xl.shape}")
    if xq.shape[1] != xl.shape[0]:
        raise ValueError(f"Length mismatch for suffix {suffix}: xq={xq.shape}, xl={xl.shape}")

    pos_enu = compute_drone_positions_enu(xq, xl, rl=rl, cl0=cl0, rg=rg)
    pos_ned = enu_to_ned(pos_enu)
    idx_drone = raw_indices(drone_csv_time, n_raw=pos_ned.shape[1], dt_raw=dt_raw)
    drone_err = pos_ned - drone_csv_pos[:, idx_drone, :]

    drone_rmse_per_drone = np.sqrt(np.mean(np.sum(drone_err * drone_err, axis=2), axis=1))
    result = {
        "suffix": suffix,
        "num_drones": int(pos_ned.shape[0]),
        "num_raw_points": int(pos_ned.shape[1]),
        "drone_rmse_per_drone": drone_rmse_per_drone,
        "drone_rmse_mean": float(np.mean(drone_rmse_per_drone)),
        "drone_max_abs_all": float(np.max(np.abs(drone_err))),
        "drone_mean_bias_xyz": np.mean(drone_err, axis=(0, 1)),
        "payload_rmse": None,
        "payload_max_abs": None,
        "cable_rmse": None,
        "cable_max_abs": None,
    }

    if payload_csv is not None:
        t_payload, payload_pos_csv = payload_csv
        idx_payload = raw_indices(t_payload, n_raw=xl.shape[0], dt_raw=dt_raw)
        payload_err = payload_pos_csv[idx_payload, :] - xl[:, 0:3]
        result["payload_rmse"] = rmse(payload_err)
        result["payload_max_abs"] = float(np.max(np.abs(payload_err)))

    if cable_csv is not None:
        t_cable, cable_dir_csv = cable_csv
        if cable_dir_csv.shape[0] == xq.shape[0]:
            idx_cable = raw_indices(t_cable, n_raw=xq.shape[1], dt_raw=dt_raw)
            cable_err = cable_dir_csv[:, idx_cable, :] - xq[:, :, 0:3]
            result["cable_rmse"] = rmse(cable_err)
            result["cable_max_abs"] = float(np.max(np.abs(cable_err)))

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose vertical conversion consistency")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config path")
    parser.add_argument("--scenario-dir", type=Path, default=None, help="Override scenario dir")
    parser.add_argument("--csv-dir", type=Path, default=None, help="Override output csv dir")
    parser.add_argument("--suffix", type=str, default=None, help="Override trajectory suffix")
    parser.add_argument("--dt-raw", type=float, default=None, help="Raw planner sample time")
    parser.add_argument("--rl", type=float, default=None, help="Payload radius [m]")
    parser.add_argument("--cl0", type=float, default=None, help="Cable length [m]")
    parser.add_argument("--m1", type=float, default=None, help="Payload base mass [kg]")
    parser.add_argument("--m2", type=float, default=None, help="Added mass [kg]")
    parser.add_argument("--rp", type=str, default=None, help="Added mass position x,y,z")
    args = parser.parse_args()

    cfg = read_config(resolve_path(args.config))

    scenario_cfg = cfg.get("base_dir")
    scenario_dir = resolve_path(args.scenario_dir) if args.scenario_dir else (
        resolve_path(Path(scenario_cfg)) if scenario_cfg else DEFAULT_SCENARIO
    )

    if args.csv_dir:
        csv_dir = resolve_path(args.csv_dir)
    else:
        base = Path("data/realflight_traj_vertical")
        csv_with_suffix = REPO_ROOT / base.parent / f"{base.name}{scenario_suffix(scenario_dir)}"
        csv_plain = REPO_ROOT / base
        csv_dir = csv_with_suffix if csv_with_suffix.exists() else csv_plain

    dt_raw = args.dt_raw if args.dt_raw is not None else (parse_float(cfg.get("dt")) or 0.05)
    rl = args.rl if args.rl is not None else (
        parse_float(cfg.get("payload_radius")) or parse_float(cfg.get("rl")) or 0.245
    )
    cl0 = args.cl0 if args.cl0 is not None else (parse_float(cfg.get("cl0")) or 1.0)
    m1 = args.m1 if args.m1 is not None else (parse_float(cfg.get("m1")) or 0.24)
    m2 = args.m2 if args.m2 is not None else (parse_float(cfg.get("m2")) or 0.30)
    rp_text = args.rp if args.rp is not None else cfg.get("rp")
    rp = parse_vec3(rp_text) if rp_text is not None else np.array([0.05, 0.0, 0.0], dtype=float)
    rg = (m2 / (m1 + m2)) * rp

    cfg_suffix = cfg.get("traj_suffix")
    chosen_suffix = args.suffix or cfg_suffix
    if chosen_suffix:
        suffixes = [normalize_suffix(chosen_suffix)]
    else:
        suffixes = list_suffixes(scenario_dir)
        if not suffixes:
            raise FileNotFoundError(f"No matched xq/xl suffixes found in {scenario_dir}")

    num_drones_cfg = parse_int(cfg.get("num_drones"))

    drone_csv_time, drone_csv_pos = load_drone_positions(csv_dir)
    payload_csv = load_payload_positions(csv_dir)
    cable_csv = load_cable_directions(csv_dir)

    print(f"config:       {resolve_path(args.config)}")
    print(f"scenario_dir: {scenario_dir}")
    print(f"csv_dir:      {csv_dir}")
    print(f"dt_raw={dt_raw}, rl={rl}, cl0={cl0}, m1={m1}, m2={m2}")
    print(f"rp={rp}, rg={rg}")
    print(f"num_drones(config)={num_drones_cfg}, drone_csv_files={drone_csv_pos.shape[0]}")
    if cfg_suffix:
        print(f"traj_suffix(config)={normalize_suffix(cfg_suffix)}")
    print(f"evaluating suffixes: {', '.join(suffixes)}")
    print("")

    results = []
    for suffix in suffixes:
        res = evaluate_suffix(
            scenario_dir=scenario_dir,
            suffix=suffix,
            drone_csv_time=drone_csv_time,
            drone_csv_pos=drone_csv_pos,
            payload_csv=payload_csv,
            cable_csv=cable_csv,
            rl=rl,
            cl0=cl0,
            rg=rg,
            dt_raw=dt_raw,
        )
        results.append(res)

    results.sort(key=lambda r: r["drone_rmse_mean"])
    if len(results) > 1:
        print(f"best_suffix={results[0]['suffix']} by drone RMSE")
        print("")

    for res in results:
        print(f"suffix: {res['suffix']}")
        print(f"  drone mean RMSE (m): {res['drone_rmse_mean']:.9f}")
        print(f"  drone max abs error (m): {res['drone_max_abs_all']:.9f}")
        print(
            "  drone RMSE per drone (m): "
            + " ".join(f"{v:.9f}" for v in res["drone_rmse_per_drone"])
        )
        print(
            "  drone mean bias xyz (m): "
            + " ".join(f"{v:.9e}" for v in res["drone_mean_bias_xyz"])
        )
        if res["payload_rmse"] is not None:
            print(f"  payload pos RMSE vs input xl (m): {res['payload_rmse']:.9f}")
            print(f"  payload pos max abs (m): {res['payload_max_abs']:.9f}")
        if res["cable_rmse"] is not None:
            print(f"  cable dir RMSE vs input xq (unitless): {res['cable_rmse']:.9f}")
            print(f"  cable dir max abs (unitless): {res['cable_max_abs']:.9f}")
        print("")


if __name__ == "__main__":
    main()

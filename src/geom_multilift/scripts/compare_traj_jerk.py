#!/usr/bin/env python3
"""
Compare jerk levels between 20 Hz data and 100 Hz preprocessed CSV output.
"""

import argparse
import csv
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
try:
    import matplotlib.pyplot as plt
except Exception as exc:
    plt = None
    _PLOT_IMPORT_ERROR = exc
else:
    _PLOT_IMPORT_ERROR = None


def parse_dt(setting_path: Path):
    if not setting_path.exists():
        return None
    dt = None
    with setting_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line.startswith("dt"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    try:
                        dt = float(parts[1].strip())
                    except ValueError:
                        dt = None
                break
    return dt


def resolve_payload_path(path: Path) -> Path:
    if path.is_dir():
        return path / "payload.csv"
    return path


def load_payload_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        for key in ("time", "x", "y", "z"):
            if key not in idx:
                raise ValueError(f"Missing '{key}' column in {path}")
        time = []
        pos = []
        for row in reader:
            if not row:
                continue
            time.append(float(row[idx["time"]]))
            pos.append([
                float(row[idx["x"]]),
                float(row[idx["y"]]),
                float(row[idx["z"]]),
            ])
    return np.asarray(time), np.asarray(pos)


def find_payload_npy(raw_dir: Path, override: Optional[Path]) -> Path:
    if override is not None:
        return override
    candidates = [
        raw_dir / "xl_traj.npy",
        raw_dir / "xl_traj_0_3_a_3.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No payload .npy found under {raw_dir}")


def load_payload_raw(
    raw_dir: Path,
    raw_dt: float,
    payload_override: Optional[Path],
) -> Tuple[np.ndarray, np.ndarray]:
    payload_path = find_payload_npy(raw_dir, payload_override)
    xl = np.load(payload_path, allow_pickle=True)
    pos = xl[:, 0:3]
    dt = parse_dt(raw_dir / "setting.txt") or raw_dt
    time = np.arange(pos.shape[0]) * dt
    return time, pos


def compute_jerk(time: np.ndarray, pos: np.ndarray) -> np.ndarray:
    vel = np.gradient(pos, time, axis=0)
    acc = np.gradient(vel, time, axis=0)
    jerk = np.gradient(acc, time, axis=0)
    return jerk


def jerk_stats(jerk: np.ndarray) -> dict:
    mag = np.linalg.norm(jerk, axis=1)
    return {
        "axis_rms": np.sqrt(np.mean(jerk ** 2, axis=0)),
        "axis_max": np.max(np.abs(jerk), axis=0),
        "mag_rms": float(np.sqrt(np.mean(mag ** 2))),
        "mag_mean": float(np.mean(mag)),
        "mag_max": float(np.max(mag)),
    }


def print_stats(label: str, stats: dict) -> None:
    axis_rms = stats["axis_rms"]
    axis_max = stats["axis_max"]
    print(f"{label} jerk RMS [x y z]: {axis_rms[0]:.6f} {axis_rms[1]:.6f} {axis_rms[2]:.6f}")
    print(f"{label} jerk max [x y z]: {axis_max[0]:.6f} {axis_max[1]:.6f} {axis_max[2]:.6f}")
    print(f"{label} jerk magnitude RMS: {stats['mag_rms']:.6f}")
    print(f"{label} jerk magnitude mean: {stats['mag_mean']:.6f}")
    print(f"{label} jerk magnitude max: {stats['mag_max']:.6f}")


def main() -> None:
    default_raw = Path(
        "raw_data/3quad_traj/Planning_plots_multiagent_meta_evaluation_COM_Dyn_V(m2=0.1,rg=[0.022,0.016])"
    )
    default_100hz = Path("data/preprocessed/rg_0022_0016_3quad_vertical_100hz/payload.csv")

    parser = argparse.ArgumentParser(description="Compare 20 Hz vs 100 Hz jerk levels")
    parser.add_argument("--raw-dir", type=Path, default=default_raw,
                        help="Raw scenario directory (20 Hz)")
    parser.add_argument("--raw-dt", type=float, default=0.05,
                        help="Raw dt used when setting.txt is missing")
    parser.add_argument("--raw-payload", type=Path, default=None,
                        help="Override payload .npy path")
    parser.add_argument("--csv-100hz", type=Path, default=default_100hz,
                        help="100 Hz payload CSV (file or directory)")
    parser.add_argument("--csv-20hz", type=Path, default=None,
                        help="Optional 20 Hz payload CSV (file or directory)")
    parser.add_argument("--plot", action="store_true",
                        help="Plot jerk magnitude comparison")
    args = parser.parse_args()

    if args.csv_20hz is not None:
        time_20, pos_20 = load_payload_csv(resolve_payload_path(args.csv_20hz))
    else:
        time_20, pos_20 = load_payload_raw(args.raw_dir, args.raw_dt, args.raw_payload)

    time_100, pos_100 = load_payload_csv(resolve_payload_path(args.csv_100hz))

    jerk_20 = compute_jerk(time_20, pos_20)
    jerk_100 = compute_jerk(time_100, pos_100)

    jerk_20_interp = np.column_stack([
        np.interp(time_100, time_20, jerk_20[:, 0]),
        np.interp(time_100, time_20, jerk_20[:, 1]),
        np.interp(time_100, time_20, jerk_20[:, 2]),
    ])
    jerk_diff = jerk_100 - jerk_20_interp

    print_stats("20 Hz", jerk_stats(jerk_20))
    print_stats("100 Hz", jerk_stats(jerk_100))
    print_stats("100 Hz - 20 Hz (interp)", jerk_stats(jerk_diff))

    if args.plot:
        if plt is None:
            print(f"Skipping plot; matplotlib not available: {_PLOT_IMPORT_ERROR}")
            return
        mag_20 = np.linalg.norm(jerk_20, axis=1)
        mag_100 = np.linalg.norm(jerk_100, axis=1)
        plt.figure(figsize=(10, 4))
        plt.plot(time_20, mag_20, label="20 Hz")
        plt.plot(time_100, mag_100, label="100 Hz", alpha=0.8)
        plt.xlabel("time [s]")
        plt.ylabel("jerk magnitude")
        plt.title("Payload jerk magnitude comparison")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

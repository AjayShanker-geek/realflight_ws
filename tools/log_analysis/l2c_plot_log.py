#!/usr/bin/env python3
"""
Quick plotting for L2C logs produced by log_path parameter.
Columns expected:
  - Payload DDP terms: NED setpoints/actuals/errors, ENU-transformed DDP errors,
    quaternion error (direct subtraction in wxyz), Euler angles (ENU roll/pitch/yaw),
    and the DDP feedback wrench.
  - Cable terms: q_id/q_i, e_qi=cross(q_id,q_i), omega_id/omega_i, e_omega_i.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_log(path: str):
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def quat_to_rpy_wxyz(qw, qx, qy, qz):
    """Return roll,pitch,yaw (rad) using the same formulas as the C++ logger."""
    roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = np.where(np.abs(sinp) >= 1.0, np.sign(sinp) * (np.pi / 2.0), np.arcsin(sinp))
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return roll, pitch, yaw


def wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def plot_errors(data):
    t = data["t"]

    fig1, axes1 = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    axes1[0].set_title("Payload Position (NED): setpoint vs actual")
    for k, lbl in zip(["x", "y", "z"], ["x", "y", "z"]):
        axes1[0].plot(t, data[f"x_d_{k}"], "--", label=f"{lbl}_d")
        axes1[0].plot(t, data[f"x_{k}"], "-", label=f"{lbl}")
    axes1[0].set_ylabel("pos [m]")
    axes1[0].legend(ncol=3)

    axes1[1].set_title("Payload Position Error (NED)")
    for k in ["x", "y", "z"]:
        axes1[1].plot(t, data[f"ex_ned_{k}"], label=f"ex_{k}")
    axes1[1].set_ylabel("err [m]")
    axes1[1].legend()

    axes1[2].set_title("Payload Position Error for DDP (ENU)  e_x_ENU")
    for k in ["x", "y", "z"]:
        axes1[2].plot(t, data[f"ex_enu_{k}"], label=f"ex_enu_{k}")
    axes1[2].set_ylabel("err [m]")
    axes1[2].set_xlabel("time [s]")
    axes1[2].legend()
    fig1.tight_layout()

    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    axes2[0].set_title("Payload Velocity (NED): setpoint vs actual")
    for k, lbl in zip(["x", "y", "z"], ["vx", "vy", "vz"]):
        axes2[0].plot(t, data[f"v_d_{k}"], "--", label=f"{lbl}_d")
        axes2[0].plot(t, data[f"v_{k}"], "-", label=f"{lbl}")
    axes2[0].set_ylabel("vel [m/s]")
    axes2[0].legend(ncol=3)

    axes2[1].set_title("Payload Velocity Error (NED)")
    for k in ["x", "y", "z"]:
        axes2[1].plot(t, data[f"ev_ned_{k}"], label=f"ev_{k}")
    axes2[1].set_ylabel("err [m/s]")
    axes2[1].legend()

    axes2[2].set_title("Payload Velocity Error for DDP (ENU)  e_v_ENU")
    for k in ["x", "y", "z"]:
        axes2[2].plot(t, data[f"ev_enu_{k}"], label=f"ev_enu_{k}")
    axes2[2].set_ylabel("err [m/s]")
    axes2[2].set_xlabel("time [s]")
    axes2[2].legend()
    fig2.tight_layout()

    roll, pitch, yaw = quat_to_rpy_wxyz(data["q_w"], data["q_x"], data["q_y"], data["q_z"])
    roll_d = np.zeros_like(roll) if "rpy_d_roll" not in data.dtype.names else data["rpy_d_roll"]
    pitch_d = np.zeros_like(pitch) if "rpy_d_pitch" not in data.dtype.names else data["rpy_d_pitch"]
    yaw_d = np.zeros_like(yaw) if "rpy_d_yaw" not in data.dtype.names else data["rpy_d_yaw"]
    roll_err = wrap_pi(roll - roll_d)
    pitch_err = wrap_pi(pitch - pitch_d)
    yaw_err = wrap_pi(yaw - yaw_d)

    fig3, axes3 = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    axes3[0].set_title("Payload Euler Angles (ENU): setpoint vs actual")
    axes3[0].plot(t, roll_d, "--", label="roll_d")
    axes3[0].plot(t, pitch_d, "--", label="pitch_d")
    axes3[0].plot(t, yaw_d, "--", label="yaw_d")
    axes3[0].plot(t, roll, "-", label="roll")
    axes3[0].plot(t, pitch, "-", label="pitch")
    axes3[0].plot(t, yaw, "-", label="yaw")
    axes3[0].set_ylabel("angle [rad]")
    axes3[0].legend(ncol=3)

    axes3[1].set_title("Payload Euler Error (ENU)")
    axes3[1].plot(t, roll_err, label="e_roll")
    axes3[1].plot(t, pitch_err, label="e_pitch")
    axes3[1].plot(t, yaw_err, label="e_yaw")
    axes3[1].set_ylabel("err [rad]")
    axes3[1].legend(ncol=3)

    axes3[2].set_title("DDP Quaternion Error (ENU, direct subtraction, wxyz)")
    for k in ["w", "x", "y", "z"]:
        axes3[2].plot(t, data[f"eq_{k}"], label=f"eq_{k}")
    axes3[2].set_ylabel("quat diff")
    axes3[2].set_xlabel("time [s]")
    axes3[2].legend(ncol=4)
    fig3.tight_layout()

    fig4, axes4 = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    axes4[0].set_title("Payload Body Rate (DDP version): omega_enu vs setpoint")
    for k in ["x", "y", "z"]:
        axes4[0].plot(t, data[f"omega_d_enu_{k}"], "--", label=f"omega_d_enu_{k}")
        axes4[0].plot(t, data[f"omega_enu_{k}"], "-", label=f"omega_enu_{k}")
    axes4[0].set_ylabel("rad/s")
    axes4[0].legend(ncol=3)

    axes4[1].set_title("Payload Body Rate Error for DDP (ENU)  e_Omega_ENU")
    for k in ["x", "y", "z"]:
        axes4[1].plot(t, data[f"eomega_enu_{k}"], label=f"eomega_enu_{k}")
    axes4[1].set_ylabel("rad/s")
    axes4[1].legend(ncol=3)

    axes4[2].set_title("Payload Body Rate (raw body frame)")
    for k in ["x", "y", "z"]:
        axes4[2].plot(t, data[f"omega_body_{k}"], label=f"omega_body_{k}")
    axes4[2].set_ylabel("rad/s")
    axes4[2].set_xlabel("time [s]")
    axes4[2].legend(ncol=3)
    fig4.tight_layout()

    fig5, axes5 = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    axes5[0].set_title("Cable Direction q: setpoint (q_id) vs actual (q_i)")
    for k in ["x", "y", "z"]:
        axes5[0].plot(t, data[f"q_id_{k}"], "--", label=f"q_id_{k}")
        axes5[0].plot(t, data[f"q_i_{k}"], "-", label=f"q_i_{k}")
    axes5[0].set_ylabel("unit")
    axes5[0].legend(ncol=3)

    axes5[1].set_title("Cable Direction Error used in controller  e_qi = cross(q_id, q_i)")
    for k in ["x", "y", "z"]:
        axes5[1].plot(t, data[f"eq_cable_{k}"], label=f"eq_cable_{k}")
    axes5[1].set_ylabel("unit")
    axes5[1].legend(ncol=3)

    axes5[2].set_title("Cable Omega: setpoint vs actual")
    for k in ["x", "y", "z"]:
        axes5[2].plot(t, data[f"omega_id_{k}"], "--", label=f"omega_id_{k}")
        axes5[2].plot(t, data[f"omega_i_{k}"], "-", label=f"omega_i_{k}")
    axes5[2].set_ylabel("rad/s")
    axes5[2].legend(ncol=3)

    axes5[3].set_title("Cable Omega Error used in controller  e_omega_i")
    for k in ["x", "y", "z"]:
        axes5[3].plot(t, data[f"ew_cable_{k}"], label=f"ew_cable_{k}")
    axes5[3].set_ylabel("rad/s")
    axes5[3].set_xlabel("time [s]")
    axes5[3].legend(ncol=3)
    fig5.tight_layout()

    fig6, axes6 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes6[0].set_title("DDP Feedback Force (FM_body F)")
    for k, name in zip(["Fx", "Fy", "Fz"], ["x", "y", "z"]):
        axes6[0].plot(t, data[f"FM_body_{k}"], label=f"F_{name}")
    axes6[0].set_ylabel("force")
    axes6[0].legend(ncol=3)

    axes6[1].set_title("DDP Feedback Moment (FM_body M)")
    for k, name in zip(["Mx", "My", "Mz"], ["x", "y", "z"]):
        axes6[1].plot(t, data[f"FM_body_{k}"], label=f"M_{name}")
    axes6[1].set_ylabel("moment")
    axes6[1].legend(ncol=3)

    axes6[2].set_title("delta_mu segment (this drone)")
    for k in ["x", "y", "z"]:
        axes6[2].plot(t, data[f"delta_mu_{k}"], label=f"delta_mu_{k}")
    axes6[2].set_ylabel("delta_mu")
    axes6[2].set_xlabel("time [s]")
    axes6[2].legend(ncol=3)
    fig6.tight_layout()

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot L2C log CSV")
    parser.add_argument("log_path", help="Path to CSV log from l2c log_path param")
    args = parser.parse_args()
    data = load_log(args.log_path)
    plot_errors(data)


if __name__ == "__main__":
    main()

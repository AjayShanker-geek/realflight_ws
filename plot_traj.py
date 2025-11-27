import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ========== 1. Read CSV ==========
# Data is in NED frame: x (North), y (East), z (Down)
df = pd.read_csv("./data/3drone_trajectories_001_-001/drone_2_traj.csv", sep=",")  # change sep if needed

time = df["time"].to_numpy()
x = df["x"].to_numpy()
y = df["y"].to_numpy()
z = df["z"].to_numpy()
vx = df["vx"].to_numpy()
vy = df["vy"].to_numpy()
vz = df["vz"].to_numpy()

# ========== 2. Velocity from position (finite difference) ==========
vx_from_pos = np.gradient(x, time)
vy_from_pos = np.gradient(y, time)
vz_from_pos = np.gradient(z, time)

# Error between CSV velocity and position-derived velocity
err_vx = vx - vx_from_pos
err_vy = vy - vy_from_pos
err_vz = vz - vz_from_pos

rmse_vx = np.sqrt(np.mean(err_vx**2))
rmse_vy = np.sqrt(np.mean(err_vy**2))
rmse_vz = np.sqrt(np.mean(err_vz**2))

print("RMSE between CSV velocity and position-derived velocity:")
print(f"  vx: {rmse_vx:.6e}")
print(f"  vy: {rmse_vy:.6e}")
print(f"  vz: {rmse_vz:.6e}")

# ========== 3. Acceleration and jerk from position ==========
ax_from_pos = np.gradient(vx_from_pos, time)
ay_from_pos = np.gradient(vy_from_pos, time)
az_from_pos = np.gradient(vz_from_pos, time)

jx_from_pos = np.gradient(ax_from_pos, time)
jy_from_pos = np.gradient(ay_from_pos, time)
jz_from_pos = np.gradient(az_from_pos, time)

# Stack for vector operations (NED)
a = np.column_stack((ax_from_pos, ay_from_pos, az_from_pos))  # acceleration in NED
j = np.column_stack((jx_from_pos, jy_from_pos, jz_from_pos))  # jerk in NED

# ========== 4. Attitude from acceleration (NED, z down) ==========
g = 9.81
e3 = np.array([0.0, 0.0, 1.0])  # NED down axis
m = 1.0  # mass, only scales thrust magnitude

# From NED dynamics:  m * a = m * g * e3 - T * R * e3  =>  T * z_b = m (g*e3 - a)
f_vec = g * e3 - a                   # proportional to T * z_b
f_norm = np.linalg.norm(f_vec, axis=1)
eps = 1e-6
f_norm = np.maximum(f_norm, eps)

# Body z-axis in NED (z_b points down)
z_b = f_vec / f_norm[:, None]        # (N,3)

# Choose desired yaw (heading) – still not observable from position alone
psi_des = 0.0  # rad, heading in NED frame (0 = facing North)
x_c = np.array([np.cos(psi_des), np.sin(psi_des), 0.0])
y_c = np.array([-np.sin(psi_des), np.cos(psi_des), 0.0])

# Build rotation matrices R (NED-from-body) for each time step
R_list = []
for i in range(len(time)):
    # x_b is perpendicular to both y_c and z_b
    x_b = np.cross(y_c, z_b[i])
    x_b_norm = np.linalg.norm(x_b)
    if x_b_norm < 1e-6:
        # Degenerate case; fallback
        x_b = np.array([1.0, 0.0, 0.0])
        x_b_norm = 1.0
    x_b /= x_b_norm

    # y_b = z_b × x_b
    y_b = np.cross(z_b[i], x_b)
    y_b /= np.linalg.norm(y_b)

    # Columns of R are body axes expressed in NED frame
    R = np.column_stack((x_b, y_b, z_b[i]))
    R_list.append(R)

R = np.stack(R_list)  # (N,3,3)

# Extract Euler angles (ZYX convention) relative to NED
roll  = np.arctan2(R[:, 2, 1], R[:, 2, 2])
pitch = np.arcsin(-R[:, 2, 0])
yaw   = np.arctan2(R[:, 1, 0], R[:, 0, 0])

# ========== 5. Angular velocity from jerk (NED version) ==========
# f = m (g e3 - a)  =>  f_dot = - m j
T = m * f_norm
f_dot = -m * j
T_dot = np.sum(f_dot * z_b, axis=1)   # since T = f · z_b

# z_b_dot = (f_dot - T_dot z_b) / T
z_b_dot = (f_dot - T_dot[:, None] * z_b) / T[:, None]

# Kinematics: z_b_dot = omega × z_b
# Cross both sides with z_b:
#   z_b × z_b_dot = z_b × (omega × z_b) = omega - (omega·z_b) z_b
# This gives the component of omega orthogonal to z_b (yaw rate unobservable)
omega_world_perp = np.cross(z_b, z_b_dot)

# Assume zero yaw rate about body z-axis => omega_world = omega_world_perp
omega_world = omega_world_perp

# Convert to body frame: omega_body = R^T * omega_world
omega_body = np.einsum('nij,nj->ni', np.transpose(R, (0, 2, 1)), omega_world)
p = omega_body[:, 0]  # roll rate
q = omega_body[:, 1]  # pitch rate
r = omega_body[:, 2]  # yaw rate (here mostly ~0 by construction)

# ========== 6. Plotting ==========

# Figure 1: 3D trajectory in NED
fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot(x, y, z)
ax1.set_title("3D Trajectory (NED)")
ax1.set_xlabel("North [m]")
ax1.set_ylabel("East [m]")
ax1.set_zlabel("Down [m]")
ax1.grid(True)

# Figure 2: position + velocity comparison
fig2, axs2 = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

axs2[0].plot(time, x, label="x (North)")
axs2[0].plot(time, y, label="y (East)")
axs2[0].plot(time, z, label="z (Down)")
axs2[0].set_ylabel("Position [m]")
axs2[0].set_title("Position (NED)")
axs2[0].legend()
axs2[0].grid(True)

axs2[1].plot(time, vx, label="vx (csv)")
axs2[1].plot(time, vx_from_pos, "--", label="vx (from pos)")
axs2[1].set_ylabel("vx [m/s]")
axs2[1].set_title("vx comparison")
axs2[1].legend()
axs2[1].grid(True)

axs2[2].plot(time, vy, label="vy (csv)")
axs2[2].plot(time, vy_from_pos, "--", label="vy (from pos)")
axs2[2].set_ylabel("vy [m/s]")
axs2[2].set_title("vy comparison")
axs2[2].legend()
axs2[2].grid(True)

axs2[3].plot(time, vz, label="vz (csv)")
axs2[3].plot(time, vz_from_pos, "--", label="vz (from pos)")
axs2[3].set_ylabel("vz [m/s]")
axs2[3].set_xlabel("time [s]")
axs2[3].set_title("vz comparison")
axs2[3].legend()
axs2[3].grid(True)

plt.tight_layout()

# Figure 3: acceleration and jerk
fig3, axs3 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axs3[0].plot(time, ax_from_pos, label="ax")
axs3[0].plot(time, ay_from_pos, label="ay")
axs3[0].plot(time, az_from_pos, label="az")
axs3[0].set_ylabel("Acceleration [m/s²]")
axs3[0].set_title("Acceleration (NED, from position)")
axs3[0].legend()
axs3[0].grid(True)

axs3[1].plot(time, jx_from_pos, label="jx")
axs3[1].plot(time, jy_from_pos, label="jy")
axs3[1].plot(time, jz_from_pos, label="jz")
axs3[1].set_ylabel("Jerk [m/s³]")
axs3[1].set_xlabel("time [s]")
axs3[1].set_title("Jerk (NED, from position)")
axs3[1].legend()
axs3[1].grid(True)

plt.tight_layout()

# Figure 4: attitude (roll, pitch, yaw) and body rates
fig4, axs4 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Attitude
axs4[0].plot(time, roll,  label="roll [rad]")
axs4[0].plot(time, pitch, label="pitch [rad]")
axs4[0].plot(time, yaw,   label="yaw [rad]")
axs4[0].set_ylabel("Angle [rad]")
axs4[0].set_title("Attitude (relative to NED)")
axs4[0].legend()
axs4[0].grid(True)

# Body rates
axs4[1].plot(time, p, label="p (roll rate)")
axs4[1].plot(time, q, label="q (pitch rate)")
axs4[1].plot(time, r, label="r (yaw rate)")
axs4[1].set_ylabel("Rate [rad/s]")
axs4[1].set_xlabel("time [s]")
axs4[1].set_title("Body rates (p, q, r)")
axs4[1].legend()
axs4[1].grid(True)

plt.tight_layout()
plt.show()

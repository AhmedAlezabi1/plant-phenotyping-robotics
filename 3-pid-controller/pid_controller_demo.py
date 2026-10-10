"""
controller.py (Task 10 – PID Controller Demo)

This script moves the OT-2 pipette to a target position using PID control.

What it does:
- Starts the OT-2 simulation
- Uses 3 PID controllers (X, Y, Z)
- Sends velocity commands (vx, vy, vz) to move toward the target
- Measures the Euclidean error (distance) to the target
- Stops when error < 0.001 m (1 mm)
- Saves a GIF (movement.gif) with a small on-screen overlay showing target, position, and error

Notes:
- This demo uses the same final tuned PID gains reported in README:
  Kp=3.5, Ki=0.0, Kd=1.2 for X/Y/Z, with output clamped to ±0.3.
- The control loop performs a "read step" (0 velocity) followed by a "command step"
  due to the sim API pattern used elsewhere in this project.
"""

from sim_class import Simulation
from pid import PID
import numpy as np
import time
import imageio
import cv2


# CONFIGURATION


TARGET_POS = [0.22, -0.15, 0.27]
MAX_STEPS = 2000
STOP_THRESHOLD = 0.001
DT = 0.01

GIF_NAME = "movement.gif"
GIF_FPS = 20

# Final tuned gains (match README + pid_tuning_test.py)
Kp, Ki, Kd = 3.5, 0.0, 1.2
VEL_CLAMP = 0.3



# Small overlay (bottom-center)


def overlay_text(frame, lines):
    img = np.array(frame)

    # RGBA -> RGB
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # Ensure uint8
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    img = img.copy()
    h, w = img.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.38
    thickness = 1
    dy = 12

    box_w = 320
    box_h = dy * (len(lines) + 1)

    x0 = max(0, (w - box_w) // 2)
    y1 = h - 3
    y0 = max(0, y1 - box_h)

    cv2.rectangle(img, (x0, y0), (x0 + box_w, y1), (0, 0, 0), thickness=-1)

    tx = x0 + 6
    ty = y0 + 14
    for i, t in enumerate(lines):
        y = ty + i * dy
        # shadow
        cv2.putText(
            img, t, (tx + 1, y + 1), font, scale, (0, 0, 0),
            thickness + 1, cv2.LINE_8
        )
        # text
        cv2.putText(
            img, t, (tx, y), font, scale, (255, 255, 255),
            thickness, cv2.LINE_8
        )

    return img


# SETUP SIMULATION


sim = Simulation(num_agents=1, render=True, rgb_array=True)
gif_frames = []

# CREATE 3 PID CONTROLLERS


pid_x = PID(Kp=Kp, Ki=Ki, Kd=Kd, output_limit=VEL_CLAMP)
pid_y = PID(Kp=Kp, Ki=Ki, Kd=Kd, output_limit=VEL_CLAMP)
pid_z = PID(Kp=Kp, Ki=Ki, Kd=Kd, output_limit=VEL_CLAMP)


# MAIN LOOP


final_error = None
final_step = None
final_pip = None

for step in range(MAX_STEPS):

    # Read current state (no movement)
    state = sim.run([[0, 0, 0, 0]])
    pip = state["robotId_1"]["pipette_position"]
    px, py, pz = pip

    # PID velocity commands from current position
    vx = pid_x.update(TARGET_POS[0], px)
    vy = pid_y.update(TARGET_POS[1], py)
    vz = pid_z.update(TARGET_POS[2], pz)

    # Apply velocity command (movement happens here)
    state = sim.run([[vx, vy, vz, 0]])

    # Read UPDATED position after movement
    pip = state["robotId_1"]["pipette_position"]

    # Euclidean error to target
    dist = float(np.linalg.norm(np.array(TARGET_POS) - np.array(pip)))

    print(f"Step {step} | Pipette: {pip} | Error(m): {dist:.6f}")

    # Save annotated frame for GIF
    if sim.current_frame is not None:
        lines = [
            f"target: [{TARGET_POS[0]:+.3f},{TARGET_POS[1]:+.3f},{TARGET_POS[2]:+.3f}]",
            f"pip:    [{pip[0]:+.3f},{pip[1]:+.3f},{pip[2]:+.3f}]",
            f"err(m):  {dist:.6f}",
        ]
        gif_frames.append(overlay_text(sim.current_frame, lines))

    # Stop condition
    if dist < STOP_THRESHOLD:
        print("\nTarget reached with high precision!")
        final_error = dist
        final_step = step
        final_pip = pip
        break

    time.sleep(DT)

# If not reached, still store final values
if final_error is None:
    final_error = dist
    final_step = MAX_STEPS
    final_pip = pip

print("\nFINAL RESULT")
print(f"Target: {TARGET_POS}")
print(f"Final pipette position: {final_pip}")
print(f"Final error (m): {final_error:.6f}")
print(f"Steps used: {final_step}")
# SAVE GIF
imageio.mimsave(GIF_NAME, gif_frames, fps=GIF_FPS)
print(f"\nGIF saved as {GIF_NAME}!")

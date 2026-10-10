"""
move_pid_to_tip.py

Uses the PID controller (from Task 10) to move the OT-2 pipette
to a robot coordinate and drop inoculum.

PID gains (from Task 10 tuning):
  Kp = 3.5, Ki = 0.0, Kd = 1.2
  Velocity clamp: ±0.3
  Tolerance: 0.001m (1mm)
"""

import sys
import os
import time
import numpy as np

# Add task10 folder to path so we can import pid.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "task10_pid_controller"))
from pid import PID


# PID Configuration (from Task 10) 

Kp = 3.5
Ki = 0.0
Kd = 1.2
VEL_CLAMP = 0.3
TOLERANCE = 0.001   # 1mm
MAX_STEPS = 2000
DT = 0.01


# Functions 
def move_to_target(sim, target, tolerance=TOLERANCE, max_steps=MAX_STEPS, dt=DT, gif_frames=None):
    """
    Move the pipette to target [x, y, z] using 3 PID controllers.

    Args:
        sim: Simulation instance
        target: list [x, y, z] in robot coordinates (meters)
        tolerance: stop when error < this (meters)
        max_steps: max control loop iterations
        dt: sleep between steps (seconds)
        gif_frames: optional list — appends rendered frames during movement

    Returns:
        final_error: float — Euclidean distance to target (meters)
        steps_used: int
        final_position: list [x, y, z]
    """
    # Create fresh PID controllers for each axis
    pid_x = PID(Kp=Kp, Ki=Ki, Kd=Kd, output_limit=VEL_CLAMP)
    pid_y = PID(Kp=Kp, Ki=Ki, Kd=Kd, output_limit=VEL_CLAMP)
    pid_z = PID(Kp=Kp, Ki=Ki, Kd=Kd, output_limit=VEL_CLAMP)

    target = np.array(target, dtype=float)
    pip = None
    dist = None

    for step in range(max_steps):
        # Read current position (zero-velocity step)
        state = sim.run([[0, 0, 0, 0]])

        # Find the correct robot key (could be robotId_1, robotId_2, etc.)
        robot_key = list(state.keys())[0]
        pip = state[robot_key]["pipette_position"]

        # Compute PID velocity commands
        vx = pid_x.update(target[0], pip[0])
        vy = pid_y.update(target[1], pip[1])
        vz = pid_z.update(target[2], pip[2])

        # Apply velocity command
        state = sim.run([[vx, vy, vz, 0]])
        pip = state[robot_key]["pipette_position"]

        # Compute Euclidean error
        dist = float(np.linalg.norm(target - np.array(pip)))

        # Print progress every 200 steps
        if step % 200 == 0:
            print(f"    step={step:04d}  "
                  f"pos=[{pip[0]:.4f}, {pip[1]:.4f}, {pip[2]:.4f}]  "
                  f"err={dist:.6f}m")

        # Capture GIF frame during movement (every 10th step to keep GIF small)
        if gif_frames is not None and step % 10 == 0 and hasattr(sim, 'current_frame') and sim.current_frame is not None:
            frame = np.array(sim.current_frame)
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]  # RGBA → RGB
            gif_frames.append(frame.copy())

        # Check if we reached the target
        if dist < tolerance:
            print(f"    Reached target in {step} steps (error: {dist:.6f}m)")
            return dist, step, list(pip)

        time.sleep(dt)

    print(f"    Max steps reached (error: {dist:.6f}m)")
    return dist, max_steps, list(pip)


def drop_inoculum(sim):
    """
    Drop inoculum at the current pipette position.
    The 4th action value = 1 triggers the drop.
    """
    sim.run([[0, 0, 0, 1]])
    print("    Inoculum dropped!")


def inoculate_tip(sim, robot_target, gif_frames=None):
    """
    Full inoculation sequence for one root tip:
      1. Move pipette to target using PID
      2. Drop inoculum if close enough
      3. Return result

    Args:
        sim: Simulation instance
        robot_target: list [x, y, z] in robot coordinates
        gif_frames: optional list — appends rendered frames during movement

    Returns:
        dict with 'error', 'steps', 'final_position', 'success'
    """
    print(f"  Moving to [{robot_target[0]:.4f}, {robot_target[1]:.4f}, {robot_target[2]:.4f}]")

    error, steps, final_pos = move_to_target(sim, robot_target, gif_frames=gif_frames)

    success = error < TOLERANCE
    if success:
        drop_inoculum(sim)
        # Capture a few frames after drop so the red dot is visible in GIF
        if gif_frames is not None:
            for _ in range(20):
                sim.run([[0, 0, 0, 0]])
                if hasattr(sim, 'current_frame') and sim.current_frame is not None:
                    frame = np.array(sim.current_frame)
                    if frame.ndim == 3 and frame.shape[2] == 4:
                        frame = frame[:, :, :3]
                    gif_frames.append(frame.copy())
    else:
        print(f"    WARNING: Could not reach target (error: {error:.6f}m)")

    return {
        "error": error,
        "steps": steps,
        "final_position": final_pos,
        "success": success,
    }


# Quick test 

if __name__ == "__main__":
    # This tests that the PID movement works standalone
    # Run: python move_pid_to_tip.py

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "task10_pid_controller"))
    from sim_class import Simulation

    sim = Simulation(num_agents=1, render=False, rgb_array=False)

    # Test target (safe position inside work envelope)
    test_target = [0.15, 0.00, 0.20]
    print(f"Testing PID movement to {test_target}\n")

    result = inoculate_tip(sim, test_target)

    print(f"\nResult:")
    print(f"  Success: {result['success']}")
    print(f"  Error:   {result['error']:.6f}m")
    print(f"  Steps:   {result['steps']}")
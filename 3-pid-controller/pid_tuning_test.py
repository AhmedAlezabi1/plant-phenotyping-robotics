"""
pid_tuning_test.py (Task 10 – PID Testing / Tuning)

This script tests the PID controller on multiple target positions.

What it does:
- Tests 5 fixed target points inside the OT-2 work envelope (repeatable results).
- For each target, it uses 3 PID controllers (X, Y, Z) to move the pipette to the target.
- Measures final positioning error (Euclidean distance) and steps taken.
- Prints a summary for each target and an overall summary (min / max / mean error).
- Saves clear graphs (PNG) for EACH target:
    1) X/Y/Z position vs step (with target lines AND target values in the title/legend)
    2) Euclidean error vs step (with 1 mm threshold AND target values in the title)

Why fixed targets:
- The OT-2 workspace is not a perfect box (Z-min depends on X/Y).
- Fixed targets make the test repeatable and avoid unreachable combinations.

Note:
- GIF evidence is produced in pid_controller_demo.py.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sim_class import Simulation
from pid import PID



# OT-2 workspace boundaries (from Task 9 measurements)

X_MIN, X_MAX = -0.187, 0.253
Y_MIN, Y_MAX = -0.1706, 0.2196
Z_MIN, Z_MAX = 0.1195, 0.2896


def make_pid():
    """Create a fresh PID controller for one axis (with velocity clamp)."""
    return PID(Kp=3.5, Ki=0.0, Kd=1.2, output_limit=0.3)


def save_target_plots(target_idx, target, steps_log, x_log, y_log, z_log, err_log, tolerance):
    """
    Save clear plots for one target:
    - Position response (X/Y/Z) vs step, with target values shown
    - Error vs step, with 1 mm threshold and target values shown
    """
    tx, ty, tz = target

    
    # Plot 1: Positions vs step
    
    plt.figure()
    plt.plot(steps_log, x_log, label="X (measured)")
    plt.plot(steps_log, y_log, label="Y (measured)")
    plt.plot(steps_log, z_log, label="Z (measured)")

    plt.axhline(tx, linestyle="--", label=f"X target = {tx:.3f}")
    plt.axhline(ty, linestyle="--", label=f"Y target = {ty:.3f}")
    plt.axhline(tz, linestyle="--", label=f"Z target = {tz:.3f}")

    # Mark final values (helps readability)
    plt.scatter(steps_log[-1], x_log[-1], marker="o")
    plt.scatter(steps_log[-1], y_log[-1], marker="o")
    plt.scatter(steps_log[-1], z_log[-1], marker="o")

    plt.xlabel("Step")
    plt.ylabel("Position (m)")
    plt.title(
        f"Target {target_idx}: Position Response\n"
        f"Target = [x={tx:.3f}, y={ty:.3f}, z={tz:.3f}]"
    )
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"target_{target_idx}_position_response.png", dpi=200)
    plt.close()

    
    # Plot 2: Error vs step
    
    plt.figure()
    plt.plot(steps_log, err_log, label="Euclidean error (m)")
    plt.axhline(tolerance, linestyle="--", label=f"Threshold = {tolerance:.3f} m (1 mm)")

    plt.scatter(steps_log[-1], err_log[-1], marker="o")

    plt.xlabel("Step")
    plt.ylabel("Error (m)")
    plt.title(
        f"Target {target_idx}: Error Response\n"
        f"Target = [x={tx:.3f}, y={ty:.3f}, z={tz:.3f}]"
    )
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"target_{target_idx}_error_response.png", dpi=200)
    plt.close()


def move_to_target(sim, target, max_steps=2000, tolerance=0.001, dt=0.01):
    """
    Move the pipette to the given target using 3 PID controllers.

    Returns:
        success (bool): True if final error < tolerance
        steps_used (int): number of steps used
        final_pip (list): final pipette position [x,y,z]
        final_error (float): Euclidean error (meters)
        logs (dict): time series logs for plotting
    """
    pid_x = make_pid()
    pid_y = make_pid()
    pid_z = make_pid()

    steps_log = []
    x_log, y_log, z_log = [], [], []
    err_log = []

    final_pip = None
    final_error = None

    for step in range(max_steps):
        # Read current pipette position
        state = sim.run([[0, 0, 0, 0]])
        pip = state["robotId_1"]["pipette_position"]

        # PID velocity commands (update(target, current))
        vx = pid_x.update(target[0], pip[0])
        vy = pid_y.update(target[1], pip[1])
        vz = pid_z.update(target[2], pip[2])

        # Apply command and read updated position
        state = sim.run([[vx, vy, vz, 0]])
        pip = state["robotId_1"]["pipette_position"]

        # Euclidean error to target
        dist = float(np.linalg.norm(np.array(target) - np.array(pip)))

        # Log data
        steps_log.append(step)
        x_log.append(pip[0])
        y_log.append(pip[1])
        z_log.append(pip[2])
        err_log.append(dist)

        # Print occasionally
        if step % 50 == 0:
            print(f"  step={step:04d} pip={pip} err(m)={dist:.6f}")

        # Stop if within tolerance
        if dist < tolerance:
            final_pip = pip
            final_error = dist
            logs = {"steps": steps_log, "x": x_log, "y": y_log, "z": z_log, "err": err_log}
            return True, step, final_pip, final_error, logs

        time.sleep(dt)

    # If not reached in time
    final_pip = pip
    final_error = dist
    logs = {"steps": steps_log, "x": x_log, "y": y_log, "z": z_log, "err": err_log}
    return False, max_steps, final_pip, final_error, logs


if __name__ == "__main__":
    sim = Simulation(num_agents=1, render=False, rgb_array=False)

   
    # Fixed targets (repeatable, safe inside envelope)
    
    
    targets = [
        [0.00, 0.00, 0.27],        
        [-0.15, -0.15, 0.20],      
        [0.22, -0.15, 0.27],       
        [-0.15, 0.20, 0.20],       
        [0.22, 0.20, 0.20],        
    ]

    print("\nFixed targets:")
    for i, t in enumerate(targets, 1):
        print(f"  {i}: {t}")

    results = []

    for i, target in enumerate(targets, 1):
        print(f"\n=== Target {i}/{len(targets)}: {target} ===")
        success, steps_used, final_pip, final_error, logs = move_to_target(sim, target)

        print("RESULT")
        print(f"  success: {success}")
        print(f"  steps_used: {steps_used}")
        print(f"  final_pip: {final_pip}")
        print(f"  final_error (m): {final_error:.6f}")

        # Save clearer plots for this target
        save_target_plots(
            target_idx=i,
            target=target,
            steps_log=logs["steps"],
            x_log=logs["x"],
            y_log=logs["y"],
            z_log=logs["z"],
            err_log=logs["err"],
            tolerance=0.001,
        )
        print(f"Saved plots for target {i}: "
              f"target_{i}_position_response.png, target_{i}_error_response.png")

        results.append((target, success, steps_used, final_pip, final_error))

    # Summary (quick error analysis)
    errors = [r[4] for r in results]
    successes = sum(1 for r in results if r[1])

    print("\nSUMMARY")
    print(f"Targets tested: {len(results)}")
    print(f"Successes (< 0.001 m): {successes}/{len(results)}")
    print(f"Min error (m): {min(errors):.6f}")
    print(f"Max error (m): {max(errors):.6f}")
    print(f"Mean error (m): {sum(errors)/len(errors):.6f}")

    print("\nFinished PID tuning test.")
    print("\nPlot files created:")
    for i in range(1, len(targets) + 1):
        print(f"  target_{i}_position_response.png, target_{i}_error_response.png")

"""
compare_pid_vs_rl.py (Task 11)

Creates simple comparison visuals between:
- PID controller (Task 10)
- RL controller (Task 11 Stage 4)

Outputs:
- comparison_bands.png
- comparison_avg_error.png

These plots are intended for the Task 11 README and slides.
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    
    # PID (Task 10) — from your README
    
    pid_targets_tested = 5
    pid_success_lt_1mm = 5  # 5/5 achieved < 0.001 m
    pid_mean_error = 0.000931  # m (from your README)
    pid_min_error = 0.000900
    pid_max_error = 0.000995

    # Convert PID to rubric bands for visualization (all <1mm)
    pid_bands = {
        "<1mm": pid_targets_tested,
        "1-5mm": 0,
        "5-10mm": 0,
        ">=10mm": 0,
    }

    
    # RL (Task 11) — from your evaluation outputs
    
    # Run A: threshold 0.005
    rl_a_episodes = 200
    rl_a_avg = 0.0049162026657722895
    rl_a_bands = {"<1mm": 0, "1-5mm": 199, "5-10mm": 1, ">=10mm": 0}

    # Run B: threshold 0.001
    rl_b_episodes = 200
    rl_b_avg = 0.0035388740902999416
    rl_b_bands = {"<1mm": 0, "1-5mm": 184, "5-10mm": 16, ">=10mm": 0}

    
    # Plot 1: Rubric band distributions (percent)
    
    bands = ["<1mm", "1-5mm", "5-10mm", ">=10mm"]

    pid_pct = np.array([pid_bands[b] for b in bands], dtype=float) / max(1, pid_targets_tested) * 100.0
    rl_a_pct = np.array([rl_a_bands[b] for b in bands], dtype=float) / rl_a_episodes * 100.0
    rl_b_pct = np.array([rl_b_bands[b] for b in bands], dtype=float) / rl_b_episodes * 100.0

    x = np.arange(len(bands))
    w = 0.25

    plt.figure()
    plt.bar(x - w, pid_pct, width=w, label="PID (5 targets)")
    plt.bar(x, rl_a_pct, width=w, label="RL Stage4 (thr=5mm, 200 eps)")
    plt.bar(x + w, rl_b_pct, width=w, label="RL Stage4 (thr=1mm, 200 eps)")
    plt.xticks(x, bands)
    plt.ylabel("Percent of runs (%)")
    plt.title("Rubric Band Distribution: PID vs RL")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("comparison_bands.png", dpi=200)
    plt.close()

    
    # Plot 2: Average final error (m)
    
    labels = ["PID (mean)", "RL (thr=5mm)", "RL (thr=1mm)"]
    vals = [pid_mean_error, rl_a_avg, rl_b_avg]

    plt.figure()
    plt.bar(labels, vals)
    plt.ylabel("Average final error (m)")
    plt.title("Average Final Error: PID vs RL")
    plt.tight_layout()
    plt.savefig("comparison_avg_error.png", dpi=200)
    plt.close()

    
    # Terminal summary table for README copy

    print("\n=== PID vs RL SUMMARY ===")
    print(f"PID: targets={pid_targets_tested}, mean={pid_mean_error:.6f} m, min={pid_min_error:.6f}, max={pid_max_error:.6f}")
    print(f"RL (thr=0.005): episodes={rl_a_episodes}, avg_final_distance={rl_a_avg:.6f} m, bands={rl_a_bands}")
    print(f"RL (thr=0.001): episodes={rl_b_episodes}, avg_final_distance={rl_b_avg:.6f} m, bands={rl_b_bands}")
    print("Saved: comparison_bands.png, comparison_avg_error.png\n")


if __name__ == "__main__":
    main()

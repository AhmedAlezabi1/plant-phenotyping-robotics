"""
comparison_per_target_error.py

Compares PID vs RL final positioning error on the SAME 5 fixed targets.
Outputs:
- comparison_per_target_error.png
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    targets = ["T1", "T2", "T3", "T4", "T5"]

    # PID per-target final errors (m) from Task 10 output
    pid_err = np.array([0.000995, 0.000949, 0.000900, 0.000906, 0.000900], dtype=float)

    # RL per-target final errors (m) from Task 11 GIF demo output
    rl_err = np.array([0.004993, 0.004986, 0.004966, 0.004966, 0.004920], dtype=float)

    x = np.arange(len(targets))
    w = 0.35

    plt.figure()
    plt.bar(x - w/2, pid_err, width=w, label="PID (Task 10)")
    plt.bar(x + w/2, rl_err, width=w, label="RL Stage 4 (Task 11)")

    plt.xticks(x, targets)
    plt.ylabel("Final error (m)")
    plt.title("Per-Target Final Error: PID vs RL (Same 5 Targets)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("comparison_per_target_error.png", dpi=200)
    plt.close()

    print("Saved: comparison_per_target_error.png")
    print("PID errors:", pid_err)
    print("RL errors :", rl_err)


if __name__ == "__main__":
    main()

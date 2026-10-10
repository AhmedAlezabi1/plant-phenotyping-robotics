"""
task13_benchmark.py

Task 13: Performance Benchmarking and System Evaluation

Runs the full Task 12 pipeline on multiple plates and measures:
  - Accuracy: positioning error per plant (meters)
  - Execution time: CV pipeline time, PID movement time, total time
  - Success rate: how many tips inoculated out of total detected
  - Per-plate and per-plant breakdown

Saves results to CSV and generates benchmark plots.

Usage:
    python task13_benchmark.py

Author: Ahmed Ali Alezabi
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add task10 folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "task10_pid_controller"))

from sim_class import Simulation
from tensorflow.keras.models import load_model

from cv_pipeline import run_pipeline
from pixel_to_robot_mm import compute_conversion_factor, convert_all_tips
from move_pid_to_tip import move_to_target, drop_inoculum, TOLERANCE



# CONFIGURATION


MODEL_PATH = "ahmed_245226_Residualunet_model_192px.h5"
PATCH_SIZE = 192
NUM_PLATES = 10  # benchmark on 10 plates
OUTPUT_DIR = "outputs/benchmark"



# BENCHMARK ONE PLATE


def benchmark_one_plate(sim, model, plate_num):
    """
    Run the full pipeline on one plate and measure everything.

    Returns:
        dict with timing, accuracy, and per-plant results
    """
    plate_result = {
        "plate": plate_num,
        "plants": [],
        "cv_time": 0,
        "convert_time": 0,
        "pid_total_time": 0,
        "total_time": 0,
        "tips_detected": 0,
        "tips_success": 0,
    }

    total_start = time.time()

    #  Step 1: Get plate image ---
    image_path = sim.get_plate_image()
    print(f"\n  Plate {plate_num}: {os.path.basename(image_path)}")

    #  Step 2: CV Pipeline (timed) ---
    cv_start = time.time()
    tips, plate_pixel_size = run_pipeline(image_path, model, patch_size=PATCH_SIZE)
    cv_end = time.time()
    cv_time = cv_end - cv_start

    plate_result["cv_time"] = cv_time
    plate_result["tips_detected"] = len(tips)

    if len(tips) == 0:
        plate_result["total_time"] = time.time() - total_start
        print(f"  No tips detected. CV time: {cv_time:.2f}s")
        return plate_result

    #  Step 3: Coordinate conversion (timed) ---
    convert_start = time.time()
    conversion_factor = compute_conversion_factor(plate_pixel_size)
    goal_positions = convert_all_tips(tips, conversion_factor)
    convert_end = time.time()
    convert_time = convert_end - convert_start

    plate_result["convert_time"] = convert_time

    #  Step 4: PID movement for each tip (timed per plant) ---
    pid_total_time = 0

    for goal in goal_positions:
        plant_id = goal["plant_id"]
        robot_target = goal["robot"]
        pixel = goal["pixel"]

        pid_start = time.time()
        error, steps, final_pos = move_to_target(sim, robot_target)
        pid_end = time.time()
        pid_time = pid_end - pid_start

        success = error < TOLERANCE
        if success:
            drop_inoculum(sim)

        pid_total_time += pid_time

        plant_data = {
            "plate": plate_num,
            "plant_id": plant_id,
            "pixel_x": pixel[0],
            "pixel_y": pixel[1],
            "robot_x": robot_target[0],
            "robot_y": robot_target[1],
            "robot_z": robot_target[2],
            "error_m": error,
            "error_mm": error * 1000,
            "steps": steps,
            "pid_time_s": pid_time,
            "success": success,
        }
        plate_result["plants"].append(plant_data)

        if success:
            plate_result["tips_success"] += 1

        status = "OK" if success else "MISS"
        print(f"    Plant {plant_id}: err={error:.6f}m ({error*1000:.2f}mm) "
              f"steps={steps} time={pid_time:.2f}s [{status}]")

    plate_result["pid_total_time"] = pid_total_time
    plate_result["total_time"] = time.time() - total_start

    return plate_result



# GENERATE PLOTS


def generate_plots(df_plants, df_plates, output_dir):
    """Generate benchmark visualization plots."""

    os.makedirs(output_dir, exist_ok=True)

    # --- Plot 1: Error per plant (bar chart) ---
    plt.figure(figsize=(12, 5))
    labels = [f"P{int(r['plate'])}-{int(r['plant_id'])}" for _, r in df_plants.iterrows()]
    colors = ['green' if r['success'] else 'red' for _, r in df_plants.iterrows()]
    plt.bar(range(len(df_plants)), df_plants["error_mm"], color=colors, alpha=0.7)
    plt.axhline(y=1.0, color='red', linestyle='--', label='1mm threshold')
    plt.xlabel("Plant (Plate-Plant)")
    plt.ylabel("Positioning Error (mm)")
    plt.title("Positioning Error per Plant")
    plt.xticks(range(len(df_plants)), labels, rotation=45, fontsize=7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_per_plant.png"), dpi=150)
    plt.close()
    print(f"  Saved: error_per_plant.png")

    # --- Plot 2: Error distribution (histogram) ---
    plt.figure(figsize=(8, 5))
    plt.hist(df_plants["error_mm"], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=1.0, color='red', linestyle='--', label='1mm threshold')
    plt.xlabel("Positioning Error (mm)")
    plt.ylabel("Count")
    plt.title("Distribution of Positioning Errors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_distribution.png"), dpi=150)
    plt.close()
    print(f"  Saved: error_distribution.png")

    # --- Plot 3: PID steps per plant ---
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(df_plants)), df_plants["steps"], color='orange', alpha=0.7)
    plt.xlabel("Plant (Plate-Plant)")
    plt.ylabel("PID Steps to Reach Target")
    plt.title("PID Steps per Plant")
    plt.xticks(range(len(df_plants)), labels, rotation=45, fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "steps_per_plant.png"), dpi=150)
    plt.close()
    print(f"  Saved: steps_per_plant.png")

    # --- Plot 4: Time breakdown per plate (stacked bar) ---
    plt.figure(figsize=(10, 5))
    x = range(len(df_plates))
    plt.bar(x, df_plates["cv_time"], label="CV Pipeline", color='steelblue')
    plt.bar(x, df_plates["pid_total_time"], bottom=df_plates["cv_time"],
            label="PID Movement", color='orange')
    plt.bar(x, df_plates["convert_time"],
            bottom=df_plates["cv_time"] + df_plates["pid_total_time"],
            label="Coord Conversion", color='green')
    plt.xlabel("Plate")
    plt.ylabel("Time (seconds)")
    plt.title("Time Breakdown per Plate")
    plt.xticks(x, [f"Plate {int(r['plate'])}" for _, r in df_plates.iterrows()])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_breakdown.png"), dpi=150)
    plt.close()
    print(f"  Saved: time_breakdown.png")

    # --- Plot 5: Total time per plate ---
    plt.figure(figsize=(10, 5))
    plt.bar(x, df_plates["total_time"], color='purple', alpha=0.7)
    plt.xlabel("Plate")
    plt.ylabel("Total Time (seconds)")
    plt.title("Total Pipeline Time per Plate")
    plt.xticks(x, [f"Plate {int(r['plate'])}" for _, r in df_plates.iterrows()])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "total_time_per_plate.png"), dpi=150)
    plt.close()
    print(f"  Saved: total_time_per_plate.png")

    # --- Plot 6: Success rate per plate ---
    plt.figure(figsize=(10, 5))
    success_rate = (df_plates["tips_success"] / df_plates["tips_detected"] * 100).fillna(0)
    plt.bar(x, success_rate, color='green', alpha=0.7)
    plt.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel("Plate")
    plt.ylabel("Success Rate (%)")
    plt.title("Inoculation Success Rate per Plate")
    plt.xticks(x, [f"Plate {int(r['plate'])}" for _, r in df_plates.iterrows()])
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "success_rate.png"), dpi=150)
    plt.close()
    print(f"  Saved: success_rate.png")


# MAIN


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("TASK 13: PERFORMANCE BENCHMARKING")
    print("=" * 60)

    # --- Load model ---
    print("\nLoading U-Net model...")
    model = load_model(MODEL_PATH, compile=False)
    print("Model loaded.")

    # --- Start simulation ---
    print("Starting simulation...")
    sim = Simulation(num_agents=1, render=False, rgb_array=False)
    print("Simulation started.")

    # --- Run benchmark on multiple plates ---
    print(f"\nBenchmarking on {NUM_PLATES} plates...\n")

    all_results = []
    all_plant_data = []

    for plate_num in range(1, NUM_PLATES + 1):
        if plate_num > 1:
            sim.reset(num_agents=1)

        result = benchmark_one_plate(sim, model, plate_num)
        all_results.append(result)
        all_plant_data.extend(result["plants"])

    # --- Build DataFrames ---
    df_plants = pd.DataFrame(all_plant_data)

    df_plates = pd.DataFrame([{
        "plate": r["plate"],
        "tips_detected": r["tips_detected"],
        "tips_success": r["tips_success"],
        "cv_time": r["cv_time"],
        "convert_time": r["convert_time"],
        "pid_total_time": r["pid_total_time"],
        "total_time": r["total_time"],
    } for r in all_results])

    # --- Save raw data ---
    df_plants.to_csv(os.path.join(OUTPUT_DIR, "benchmark_plants.csv"), index=False)
    df_plates.to_csv(os.path.join(OUTPUT_DIR, "benchmark_plates.csv"), index=False)
    print(f"\nSaved: benchmark_plants.csv, benchmark_plates.csv")

    # --- Print Summary ---
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    total_detected = df_plates["tips_detected"].sum()
    total_success = df_plates["tips_success"].sum()
    total_time_all = df_plates["total_time"].sum()

    print(f"\n  Plates tested:          {NUM_PLATES}")
    print(f"  Total tips detected:    {int(total_detected)}")
    print(f"  Total tips inoculated:  {int(total_success)}")
    print(f"  Overall success rate:   {total_success/total_detected*100:.1f}%" if total_detected > 0 else "  No tips detected")

    if len(df_plants) > 0:
        print(f"\n  ACCURACY:")
        print(f"    Mean error:           {df_plants['error_mm'].mean():.3f} mm")
        print(f"    Std error:            {df_plants['error_mm'].std():.3f} mm")
        print(f"    Min error:            {df_plants['error_mm'].min():.3f} mm")
        print(f"    Max error:            {df_plants['error_mm'].max():.3f} mm")
        print(f"    All within 1mm:       {'Yes' if (df_plants['error_mm'] < 1.0).all() else 'No'}")

    print(f"\n  TIMING:")
    print(f"    Mean CV time/plate:   {df_plates['cv_time'].mean():.2f} s")
    print(f"    Mean PID time/plate:  {df_plates['pid_total_time'].mean():.2f} s")
    print(f"    Mean total time/plate:{df_plates['total_time'].mean():.2f} s")
    print(f"    Total benchmark time: {total_time_all:.2f} s")

    if len(df_plants) > 0:
        print(f"\n  PID PERFORMANCE:")
        print(f"    Mean steps/plant:     {df_plants['steps'].mean():.0f}")
        print(f"    Mean PID time/plant:  {df_plants['pid_time_s'].mean():.2f} s")

    #  Generate Plots 
    print(f"\nGenerating plots...")
    generate_plots(df_plants, df_plates, OUTPUT_DIR)

    

if __name__ == "__main__":
    main()
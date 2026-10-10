"""
run_task12_all.py

Main script for Task 12: System Integration.

Ties everything together:
  1. Start simulation
  2. Get plate image from sim (sim.get_plate_image())
  3. Run CV pipeline → get root tip pixel coordinates
  4. Convert pixel → mm → robot coordinates
  5. Move PID to each tip and drop inoculum
  6. Repeat for multiple plates
  7. Save GIF of the whole process

Usage:
  python run_task12_all.py

Author: Ahmed Ali Alezabi
"""

import sys
import os
import time
import numpy as np
import imageio

# Add task10 folder to path for sim_class and pid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "task10_pid_controller"))

from sim_class import Simulation
from tensorflow.keras.models import load_model

# Our Task 12 modules
from cv_pipeline import run_pipeline
from pixel_to_robot_mm import compute_conversion_factor, convert_all_tips
from move_pid_to_tip import inoculate_tip



# CONFIGURATION


MODEL_PATH = "ahmed_245226_Residualunet_model_192px.h5"
PATCH_SIZE = 192

# How many plates to process
NUM_PLATES = 2  # adjust based on how many plates the sim provides

# GIF settings
SAVE_GIF = True
GIF_PATH = "task12_inoculation.gif"
GIF_FPS = 20



# PROCESS ONE PLATE


def process_one_plate(sim, model, plate_number, gif_frames=None):
    """
    Full pipeline for one plate:
      1. Get image
      2. Detect tips
      3. Convert to robot coords
      4. Inoculate each tip

    Args:
        sim: Simulation instance
        model: loaded U-Net model
        plate_number: int — plate index (for printing)
        gif_frames: optional list for GIF recording

    Returns:
        list of result dicts for each plant
    """
    print(f"\n{'='*60}")
    print(f"PLATE {plate_number}")
    print(f"{'='*60}")

    # Get plate image from simulation
    print("\n  Getting plate image...")
    image_path = sim.get_plate_image()
    print(f"  Image: {image_path}")

    # Run CV pipeline
    print("\n  Running CV pipeline...")
    tips, plate_pixel_size = run_pipeline(image_path, model, patch_size=PATCH_SIZE)

    if len(tips) == 0:
        print("  No root tips detected on this plate.")
        return []

    # Convert pixel → robot coordinates
    print(f"\n  Plate size: {plate_pixel_size}px")
    conversion_factor = compute_conversion_factor(plate_pixel_size)
    print(f"  Conversion factor: {conversion_factor:.6f} mm/pixel\n")

    goal_positions = convert_all_tips(tips, conversion_factor)

    # Inoculate each tip
    print(f"\n  Inoculating {len(goal_positions)} tips...\n")

    plate_results = []

    for i, goal in enumerate(goal_positions):
        plant_id = goal["plant_id"]
        robot_target = goal["robot"]

        print(f"  --- Plant {plant_id} ({i+1}/{len(goal_positions)}) ---")

        result = inoculate_tip(sim, robot_target, gif_frames=gif_frames)

        plate_results.append({
            "plate": plate_number,
            "plant_id": plant_id,
            "pixel": goal["pixel"],
            "robot_target": robot_target,
            "final_error": result["error"],
            "steps": result["steps"],
            "success": result["success"],
        })

        print()

    return plate_results



# MAIN


def main():
    print("=" * 60)
    print("TASK 12: SYSTEM INTEGRATION")
    print("=" * 60)

  
    # 1. Load model

    print("\n[1] Loading U-Net model...")
    model = load_model(MODEL_PATH, compile=False)
    print("    Model loaded.")


    # 2. Start simulation
 
    print("\n[2] Starting simulation...")
    sim = Simulation(num_agents=1, render=True, rgb_array=True)
    print("    Simulation started.")

    gif_frames = [] if SAVE_GIF else None

    
    # DEBUG: Print pipette starting position and plate image path
    
    state = sim.run([[0, 0, 0, 0]])
    robot_key = list(state.keys())[0]
    pip = state[robot_key]["pipette_position"]
    print(f"    Robot key: {robot_key}")
    print(f"    Pipette start position: {pip}")
    print(f"    Plate image path: {sim.plate_image_path}")


    # 3. Process multiple plates
    #    Reset simulation each time to load a new random plate
  
    all_results = []

    for plate_num in range(1, NUM_PLATES + 1):
        if plate_num > 1:
            # Reset simulation to get a new random plate image
            print(f"\n  Resetting simulation for next plate...")
            sim.reset(num_agents=1)

        plate_results = process_one_plate(sim, model, plate_num, gif_frames=gif_frames)
        all_results.extend(plate_results)


    # 4. Summary
  
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    if not all_results:
        print("  No tips were inoculated.")
        return

    errors = [r["final_error"] for r in all_results]
    successes = sum(1 for r in all_results if r["success"])
    plates_done = len(set(r["plate"] for r in all_results))

    print(f"  Plates processed:     {plates_done}")
    print(f"  Total tips found:     {len(all_results)}")
    print(f"  Successfully hit:     {successes}/{len(all_results)}")
    print(f"  Mean error:           {np.mean(errors):.6f} m")
    print(f"  Max error:            {np.max(errors):.6f} m")
    print(f"  All within 1mm:       {all(e < 0.001 for e in errors)}")

    print()
    for r in all_results:
        status = "OK" if r["success"] else "MISS"
        print(f"  Plate {r['plate']} Plant {r['plant_id']}: "
              f"pixel ({r['pixel'][0]:.0f}, {r['pixel'][1]:.0f}) → "
              f"robot [{r['robot_target'][0]:.4f}, {r['robot_target'][1]:.4f}, {r['robot_target'][2]:.4f}] "
              f"| err={r['final_error']:.6f}m [{status}]")

   
    # 5. Save GIF
  
    if SAVE_GIF and gif_frames:
        imageio.mimsave(GIF_PATH, gif_frames, fps=GIF_FPS)
        print(f"\n  GIF saved: {GIF_PATH} ({len(gif_frames)} frames)")

    print("\nTask 12 complete!")



# RUN


if __name__ == "__main__":
    main()
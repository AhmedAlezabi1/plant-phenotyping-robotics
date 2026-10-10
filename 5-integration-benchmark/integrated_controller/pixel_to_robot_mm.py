"""
pixel_to_robot_mm.py

Converts root tip pixel coordinates (from cv_pipeline.py) to robot coordinates
that the PID controller can use.

How it works:
  1. The plate is 150mm x 150mm in real life
  2. We measure how many pixels the plate is in the image
  3. conversion_factor = 150 / plate_pixels  (gives mm per pixel)
  4. Multiply pixel position by conversion_factor to get mm
  5. Convert mm to meters (divide by 1000)
  6. Add the plate's known position in robot space

Plate origin (top-left corner) in robot coordinates:
  x = 0.10775
  y = 0.088 - 0.026 = 0.062  (y adjusted by -0.026 per simulator bug fix)
  z = 0.1195  (lowest reachable pipette height)

Axis mapping:
  - Image x (horizontal, different plants) → Robot y
  - Image y (vertical, root depth)         → Robot x
"""

import numpy as np


#  Constants 

PLATE_SIZE_MM = 150  # plate is 150mm x 150mm

# Plate top-left corner in robot space (meters)
# y is adjusted by -0.026m per the simulator bug fix note
# z is set to 0.1195 (lowest reachable pipette z from Task 9 work envelope)
PLATE_ORIGIN = [0.10775, 0.088 - 0.026, 0.1195]


# ---- Functions ----

def compute_conversion_factor(plate_size_pixels):
    """
    Calculate how many mm one pixel represents.

    Args:
        plate_size_pixels: int — the plate size in pixels
                           (from the cropped image dimensions)

    Returns:
        float — mm per pixel

    Example:
        If the cropped plate is 2800 pixels wide:
        150 / 2800 = 0.0536 mm per pixel
    """
    return PLATE_SIZE_MM / plate_size_pixels


def pixel_to_robot(tip_pixel_x, tip_pixel_y, conversion_factor):
    """
    Convert one root tip from pixel coordinates to robot coordinates.

    IMPORTANT: The pixel coordinates must be relative to the cropped plate
    (not the full original image). cv_pipeline.py already handles this.

    Axis mapping:
      - pixel x (horizontal, varies between plants) → robot y
      - pixel y (vertical, root depth)              → robot x

    Args:
        tip_pixel_x: float — x in pixels (from left edge of plate)
        tip_pixel_y: float — y in pixels (from top edge of plate)
        conversion_factor: float — mm per pixel

    Returns:
        list [x, y, z] — robot coordinates in meters
    """
    # Step 1: pixel → mm
    tip_mm_x = tip_pixel_x * conversion_factor
    tip_mm_y = tip_pixel_y * conversion_factor

    # Step 2: mm → meters, add to plate origin
    # As per instructions: root_tip_robot = root_tip_mm + plate_position_robot
    # pixel x (horizontal) → robot x
    # pixel y (vertical)   → robot y
    robot_x = PLATE_ORIGIN[0] + (tip_mm_y / 1000.0)
    robot_y = PLATE_ORIGIN[1] + (tip_mm_x / 1000.0)
    robot_z = PLATE_ORIGIN[2]

    return [robot_x, robot_y, robot_z]


def convert_all_tips(tips, conversion_factor):
    """
    Convert a list of tips from cv_pipeline.py to robot coordinates.

    Args:
        tips: list of dicts from cv_pipeline.run_pipeline()
              each dict has 'plant_id', 'tip_x', 'tip_y'
        conversion_factor: float — mm per pixel

    Returns:
        list of dicts, each with:
            'plant_id': int
            'pixel': (tip_x, tip_y)
            'robot': [x, y, z]
    """
    results = []

    for tip in tips:
        robot_pos = pixel_to_robot(tip["tip_x"], tip["tip_y"], conversion_factor)

        results.append({
            "plant_id": tip["plant_id"],
            "pixel": (tip["tip_x"], tip["tip_y"]),
            "robot": robot_pos,
        })

        print(f"  Plant {tip['plant_id']}: "
              f"pixel ({tip['tip_x']:.0f}, {tip['tip_y']:.0f}) → "
              f"robot [{robot_pos[0]:.5f}, {robot_pos[1]:.5f}, {robot_pos[2]:.5f}]")

    return results


#  Quick test 

if __name__ == "__main__":
    # Example: plate is ~2800 pixels in the cropped image
    plate_px = 2800
    cf = compute_conversion_factor(plate_px)
    print(f"Plate: {plate_px}px → conversion factor: {cf:.6f} mm/pixel\n")

    # Test with a fake tip at center of plate
    robot = pixel_to_robot(1400, 1400, cf)
    print(f"Center pixel (1400, 1400) → robot {robot}")

    # Test with a tip near bottom of plate
    robot2 = pixel_to_robot(1400, 2500, cf)
    print(f"Bottom pixel (1400, 2500) → robot {robot2}")
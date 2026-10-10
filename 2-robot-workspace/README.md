# Task 09 – Robotics Environment Interaction
**Opentrons OT-2 Digital Twin – Workspace Exploration**

This task explores how to control a simulated **Opentrons OT-2** robot using velocity commands, observe its behavior, and determine the pipette’s **3D work envelope**.  
The robot is controlled through the provided `sim_class.py` PyBullet environment.

---

## **Objectives**

- Move the robot’s pipette around the workspace  
- Understand how the robot moves in x, y, and z  
- Find the *maximum* and *minimum* positions the pipette can reach  
- Move the pipette to all **8 corners** of the workspace  
- Save a GIF showing how the robot moved  
 
The robot is commanded through `sim_class.py`, and a custom script (`corners.py`) moves the pipette to all **eight extreme corners** of its reachable workspace while capturing frames and generating a GIF.

---

##  Project Structure

```
task09_robotics_environment/
│
├── meshes/                     # OT-2 3D model files
├── textures/                   # Plate and environment textures
│
├── sim_class.py                # Simulation engine (provided)
├── corners.py                  # Script that finds the 8 workspace corners
├── corners.gif                 # GIF of robot moving to all corners
├── ot_2_simulation_v6.urdf     # OT-2 model
├── custom.urdf                 # Specimen/plate URDF
└── README.md                   # (this file)
```

---

## Environment Setup

### Python Version
Python 3.10.x

### Operating System
- Tested on Windows 10 / Windows 11
- Expected to work on Linux (Ubuntu 20.04+) and macOS with Python 3.10

### Required Simulation Assets
The following provided files must be present in the project directory:

- sim_class.py – OT-2 PyBullet simulation interface (provided)
- ot_2_simulation_v6.urdf – OT-2 robot model
- custom.urdf – Plate/specimen model
- meshes/ – Robot mesh files
- textures/ – Environment and plate textures

No additional simulator downloads are required.

### Install Dependencies
Create or activate your Conda or virtual environment, then install:

pip install pybullet numpy imageio opencv-python

---

## How to Run the Simulation

To explore the robot’s work envelope and generate the GIF:

python corners.py

This script will:

- Start the PyBullet OT-2 simulation
- Apply velocity commands to the pipette
- Move the pipette in 8 diagonal directions toward workspace limits
- Detect when each physical limit is reached using motion stability
- Record pipette positions at each corner
- Capture rendered frames with on-screen observations
- Generate corners.gif
- Print the final WORK ENVELOPE CORNERS table to the terminal

---

## How Observations Are Computed

The simulation returns robot state information at each step via sim.run().

Pipette position:
- Read from state["robotId_1"]["pipette_position"]
- Represents the 3D Cartesian position (x, y, z) of the pipette tip in meters

Velocity (proxy):
- True velocity is not provided by the simulator
- Computed as the Euclidean distance between consecutive pipette positions:
  delta_per_step = ||pipette_position(t) − pipette_position(t−1)||
- Used to infer motion speed and detect workspace limits

Stability detection:
- A limit is considered reached when delta_per_step < 1e-4
- Condition must hold for 5 consecutive simulation steps

Observations are printed to the terminal and rendered as text overlays in the output GIF.

---

## Method for Determining Workspace Corners

The OT-2 pipette work envelope is determined using a velocity-based exploration approach:

- Diagonal velocity commands are applied in all 8 combinations of ±X, ±Y, and ±Z
- The pipette moves until joint or collision limits are reached
- Natural deceleration occurs near workspace boundaries
- Once motion stabilizes, the final pipette position is recorded as a workspace corner

Velocity control was chosen because:
- Direct XYZ targeting caused instability
- OT-2 kinematics couple multiple axes
- Velocity commands allow the robot to settle naturally at true physical limits

---

## Reproducibility

To reproduce the workspace measurements:

1. Ensure all required simulation assets are present
2. Activate the Python environment and install dependencies
3. Run: python corners.py
4. Copy the printed WORK ENVELOPE CORNERS output from the terminal
5. Use those values to populate the workspace table in this README

The generated corners.gif and printed coordinates come from the same simulation run.


##  GIF Output
The animation is generated using:

```
imageio.mimsave("corners.gif", gif_frames, fps=20)
```
The resulting GIF (`corners.gif`) shows the pipette moving smoothly to all 8 corners of its workspace.  
The GIF includes an on-screen text overlay showing the target corner, pipette position, per-step movement (velocity proxy), and stability state for each simulation step.

---
##  Determined Work Envelope (8 Corners)
Measured directly from the simulation:

| Corner | X | Y | Z |
|--------|--------|--------|--------|
| **x_min_y_min_z_min** | -0.187  | -0.1706 | 0.1195 |
| **x_min_y_min_z_max** | -0.187  | -0.1706 | 0.2895 |
| **x_min_y_max_z_min** | -0.187  | 0.2195  | 0.1695 |
| **x_min_y_max_z_max** | -0.187  | 0.2195  | 0.2895 |
| **x_max_y_min_z_min** | 0.2531 | -0.1705 | 0.1695 |
| **x_max_y_min_z_max** | 0.2530 | -0.1706 | 0.2896 |
| **x_max_y_max_z_min** | 0.2530 | 0.2196  | 0.1695 |
| **x_max_y_max_z_max** | 0.2530 | 0.2196  | 0.2896 |

These values form the OT-2 pipette’s **reachable 3D workspace**.


Method for Determining Corners

I used velocity-based diagonal movement rather than target-position commands.
Why diagonal velocity works better
The OT-2 arm geometry means moving in X can also slightly change Z
Velocity control avoids issues with inverse kinematics
The robot naturally decelerates near physical joint limits
The simulation becomes unstable if you try to force exact XYZ coordinates
A movement is considered “complete” when the pipette position stops changing for several frames.

---

## Key Findings

1. Minor variation in Z-minimum values across corners is due to geometric and collision constraints of the OT-2 model; the lowest reachable Z depends on the X/Y position rather than forming a perfectly flat plane

2. Joint limits cause natural deceleration

The pipette slows as it approaches its limits, allowing detection based on stability.

3. The workspace is not a perfect cube

Changing one axis influences others due to the robot’s kinematics.

4. Direct coordinate targeting is unreliable

Attempting to move directly to extreme values caused:
premature stopping
freezing
PyBullet disconnections
incorrect work envelope measurements

5. Velocity control is ideal for exploration

Diagonal velocity pushes the arm to its real physical boundaries.

6. Pipette tip is the origin of measurements

Values are returned for the pipette tip, not the robot base.














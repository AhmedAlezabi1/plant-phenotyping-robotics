
"""
pid.py (Task 10 – PID Controller)

This file contains a simple PID controller.

The PID controller is used to control one value, such as position on one axis.
It compares a target value with the current value and outputs a command
that helps move the system toward the target.

The controller uses:
- P (Proportional): reacts to how far we are from the target
- I (Integral): reacts to past error over time
- D (Derivative): reacts to how fast the error is changing

In Task 10, this PID controller is used three times:
one for X, one for Y, and one for Z.
The output of the controller is used as a velocity command for the robot.
"""

import time
class PID:
    def __init__(self, Kp=0.0, Ki=0.0, Kd=0.0, output_limit=None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

        # optional output clamp (very useful)
        self.output_limit = output_limit  

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def update(self, target, current):
        # --- Compute error ---
        error = target - current

        # --- Time step (dt) ---
        current_time = time.time()
        if self.prev_time is None:
            dt = 0.01  # default small dt for first run
        else:
            dt = current_time - self.prev_time

        self.prev_time = current_time

        #  Integral term
        self.integral += error * dt

        # Derivative term 
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error

        # PID formula 
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)

        #  Clamp output if needed 
        if self.output_limit is not None:
            output = max(min(output, self.output_limit), -self.output_limit)

        return output

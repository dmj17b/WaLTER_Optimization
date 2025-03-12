import yaml
import sys
import os
sys.path.append(os.path.dirname(__file__)+'/../lib')

import time
import mujoco
import mujoco.viewer
from pathlib import Path
import MotorModel as motor
import numpy as np
import GenWaLTER_2D
# Specs:
# Model parameters:
test_model_params = {
    'body': {
        'length': 0.5,
        'mass': 1.0,
    },
    'front_thigh': {
        'length': 0.25,
        'mass': 1.0,
    },
    'front_shin': {    
        'length': 0.3,
        'mass': 1.0,
    },
    'front_wheel': {
        'radius': 0.1,
        'mass': 1.0,
    },
    'rear_thigh': {
        'length': 0.25,
        'mass': 1.0,
    },
    'rear_shin': {    
        'length': 0.3,
        'mass': 1.0,
    },
    'rear_wheel': {
        'radius': 0.1,
        'mass': 1.0,
    },
    'general': {
        'width': 0.05,
    },
}

# Motor parameters:
test_motor_params = {
    'front_hip': {
        'kp': 1.0,
        'kd': 0.1,
        'stall_torque': 10.0,
        'no_load_speed': 10.0,
    },
    'front_knee': {
        'kp': 1.0,
        'kd': 0.1,
        'stall_torque': 10.0,
        'no_load_speed': 10.0,
    },
    'front_wheel': {
        'kp': 1.0,
        'kd': 0.1,
        'stall_torque': 10.0,
        'no_load_speed': 10.0,
    },
    'rear_hip': {
        'kp': 1.0,
        'kd': 0.1,
        'stall_torque': 10.0,
        'no_load_speed': 10.0,
    },
    'rear_knee': {
        'kp': 1.0,
        'kd': 0.1,
        'stall_torque': 10.0,
        'no_load_speed': 10.0,
    },
    'rear_wheel': {
        'kp': 1.0,
        'kd': 0.1,
        'stall_torque': 10.0,
        'no_load_speed': 10.0,
    }
}



# Generate the new robot spec:
walter = GenWaLTER_2D.WaLTER2D(test_model_params, test_motor_params)

# Generate the basic WaLTER scene (floor and lighting):
walter.gen_scene()
walter.add_stairs()
# Randomize ledge height and model pos:
rng = np.random.default_rng(seed=69)

# Compile the model:
m = walter.spec.compile()
d = mujoco.MjData(m)







with mujoco.viewer.launch_passive(m,d) as viewer:
    viewer.cam.distance = 5
    viewer.cam.azimuth = 45

    start = time.time()
    while viewer.is_running():
        step_start = time.time()
        viewer

        # Sim step:
        mujoco.mj_step(m, d)

        # Sync changes in the viewer
        viewer.sync()
        
        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
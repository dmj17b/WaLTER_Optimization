from typing import Any
from absl import app
import os
from pathlib import Path
import yaml
import numpy as np
import mujoco
import scipy


# 2D Model will appear in the X-Z plane

# Model parameters:
test_model_params = {
    'body_params': {
        'length': 0.5,
        'mass': 1.0,
    },
    'thigh_params': {
        'length': 0.5,
        'mass': 1.0,
    },
    'shin_params': {    
        'length': 0.5,
        'mass': 1.0,
    },
    'wheel_params': {
        'radius': 0.1,
        'mass': 1.0,
    },
    'general_params': {
        'width': 0.1,
    },
}

# Motor parameters:
test_motor_params = {
    'hip_params': {
        'kp': 1.0,
        'kd': 0.1,
        'stall_torque': 10.0,
        'no_load_speed': 10.0,
    },
    'knee_params': {
        'kp': 1.0,
        'kd': 0.1,
        'stall_torque': 10.0,
        'no_load_speed': 10.0,
    },
    'wheel_params': {
        'kp': 1.0,
        'kd': 0.1,
        'stall_torque': 10.0,
        'no_load_speed': 10.0,
    }
}



class GenWaLTER2D():

    def __init__(self, model_params: dict, motor_params: dict):
        self.model_params = model_params
        self.motor_params = motor_params
        self.walter = None
        self.m = None
        self.d = None
        self.spec = mujoco.MjSpec()

        # Generating the basic 2D WaLTER model:
        torso_body = self.spec.worldbody.add_body(
            name = 'torso',
            pose = [0, 0, 0],
            quat = [1, 0, 0, 0],

        )
        torso_body.add_geom(
            type = mujoco.mjtGeom.mjGEOM_CAPSULE,
            size = [self.model_params['body_params'], self.model_params['general_width']],
        )
        

def main():
    walter = GenWaLTER2D(test_model_params, test_model_params)

if __name__ == '__main__':
    main()
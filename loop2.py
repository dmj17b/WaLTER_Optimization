import yaml
import sys
import time
import mujoco
import mujoco.viewer
from pathlib import Path
import lib.MotorModel as motor
import numpy as np
import AutoSim
import lib.RandomControl as rc
from scipy.spatial.transform import Rotation

# ----------------------------------------------------------------------
# 1. Load config and do one initial model compilation
# ----------------------------------------------------------------------
model_config_path = 'model_config.yaml'
motor_config_path = 'motor_config.yaml'
motor_config = yaml.safe_load(Path(motor_config_path).read_text())

# Generate your initial robot spec (Walter) once
walter = AutoSim.GenerateModel(
    model_config_path=model_config_path, 
    motor_config_path=motor_config_path
)
walter.gen_scene()

# Compile the model once
m = walter.spec.compile()
d = mujoco.MjData(m)

# (Optionally) set initial random seed
rng_seed = 420

# ----------------------------------------------------------------------
# 2. Create/open the viewer once (outside the loop!)
# ----------------------------------------------------------------------
with mujoco.viewer.launch_passive(m, d) as viewer:
    # Customize camera, etc.
    viewer.cam.distance = 8
    viewer.cam.azimuth = 45

    # Performance counters
    num_tests = 50
    num_successes = 0
    num_failures = 0
    
    # ------------------------------------------------------------------
    # 3. Inside the loop, randomize state and run each simulation
    # ------------------------------------------------------------------
    for i in range(num_tests):
        rng = np.random.default_rng(seed=i + rng_seed)

        # a) Optionally re-randomize your scene geometry 
        #    (careful: re-generating geometry requires recompile 
        #     unless you keep the same geometry but move it).
        #    If you absolutely must re-compile the model each time,
        #    see the notes below for how to handle that.
        walter.randomize_test_scene(rng)
        
        # b) Reset or randomize the poses in MjData
        d = walter.randomize_pose(rng, m, d)
        
        # c) Initialize motors, controllers, etc.
        # (Same code you had for MotorModel creation or simply do it once.)
        # ...
        
        # d) Run your simulation loop
        start = time.time()
        while viewer.is_running():
            # --- your usual step code ---
            # ctrl.control()
            mujoco.mj_step(m, d)
            
            # check for success/failure
            # ...
            # break if success/failure

            # sync viewer
            viewer.sync()

            # Optional timing
            time.sleep(m.opt.timestep * 0.5)

    # Once we break out of the for-loop, the viewer stays open.
    # Exiting 'with' block closes it.
    print(f"Final results - Successes: {num_successes}, Failures: {num_failures}")
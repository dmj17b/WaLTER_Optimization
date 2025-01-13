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

# Create a new simulation model with AutoSim
model_config_path = 'model_config.yaml'
motor_config_path = 'motor_config.yaml'
motor_config = yaml.safe_load(Path(motor_config_path).read_text())

num_tests = 50
rng_seed = 420

success_dist = 3
torso_head_angle_tolerance = 25

num_successes = 0
num_failures = 0

for i in range(num_tests):

    # Generate the new robot spec with random ICs:
    walter = AutoSim.GenerateModel(model_config_path=model_config_path, motor_config_path=motor_config_path)
    walter.gen_scene()
    rng = np.random.default_rng(seed=i+rng_seed)
    walter.randomize_test_scene(rng)
    # Compile the model:
    m = walter.spec.compile()
    d = mujoco.MjData(m)


    # Randomize initial robot pose:
    d = walter.randomize_pose(rng,m,d)
    # Initializing motor models (ignore this part)
    fr_hip = motor.MotorModel(m, d, 'head_right_thigh_joint', motor_config['hip_params'], 12)
    fl_hip = motor.MotorModel(m, d,'head_left_thigh_joint',  motor_config['hip_params'], 8)
    br_hip = motor.MotorModel(m, d,'torso_right_thigh_joint',  motor_config['hip_params'], 4)
    bl_hip = motor.MotorModel(m, d,'torso_left_thigh_joint',  motor_config['hip_params'], 0)

    fr_knee = motor.MotorModel(m, d, 'head_right_thigh_shin_joint', motor_config['knee_params'], 13)
    fl_knee = motor.MotorModel(m, d, 'head_left_thigh_shin_joint', motor_config['knee_params'], 9)
    br_knee = motor.MotorModel(m, d, 'torso_right_thigh_shin_joint', motor_config['knee_params'], 5)
    bl_knee = motor.MotorModel(m, d, 'torso_left_thigh_shin_joint', motor_config['knee_params'], 1)

    fr_wheel1_joint = motor.MotorModel(m, d, 'head_right_shin_front_wheel_joint', motor_config['wheel_params'], 14)
    fr_wheel2_joint = motor.MotorModel(m, d, 'head_right_shin_rear_wheel_joint', motor_config['wheel_params'], 15)
    fl_wheel1_joint = motor.MotorModel(m, d, 'head_left_shin_front_wheel_joint', motor_config['wheel_params'], 10)
    fl_wheel2_joint = motor.MotorModel(m, d, 'head_left_shin_rear_wheel_joint', motor_config['wheel_params'], 11)
    br_wheel1_joint = motor.MotorModel(m, d, 'torso_right_shin_front_wheel_joint', motor_config['wheel_params'], 6)
    br_wheel2_joint = motor.MotorModel(m, d, 'torso_right_shin_rear_wheel_joint', motor_config['wheel_params'], 7)
    bl_wheel1_joint = motor.MotorModel(m, d, 'torso_left_shin_front_wheel_joint', motor_config['wheel_params'], 2)
    bl_wheel2_joint = motor.MotorModel(m, d, 'torso_left_shin_rear_wheel_joint', motor_config['wheel_params'], 3)

    motors = [fr_hip, fl_hip, br_hip, bl_hip, 
            fr_knee, fl_knee, br_knee, bl_knee, 
            fr_wheel1_joint, fr_wheel2_joint, fl_wheel1_joint, fl_wheel2_joint, br_wheel1_joint, br_wheel2_joint, bl_wheel1_joint, bl_wheel2_joint]



    # Initialize new controller with random inputs:
    ctrl = rc.RandomController(m,d,motors,rng)
    ctrl.randomize_control()


    with mujoco.viewer.launch_passive(m,d) as viewer:
        viewer.cam.distance = 8
        viewer.cam.azimuth = 45

        start = time.time()
        while viewer.is_running():
            step_start = time.time()
            viewer

            # Call controller:
            ctrl.control()

            # Sim step:
            mujoco.mj_step(m, d)

            # If WaLTER moves off the ledge, break the loop
            off_ledge = (abs(d.body('torso').xpos[0]) > success_dist or abs(d.body('torso').xpos[1]) > success_dist)

            # Calculate angle between torso and head. If this angle is too large, a failure may be considered
            head_angle = Rotation.from_quat(d.body('head').xquat).as_euler('xyz', degrees=True)[2]
            torso_angle = Rotation.from_quat(d.body('torso').xquat).as_euler('xyz', degrees=True)[2]
            torso_head_angle_offset = abs(torso_angle-head_angle)

            print(d.body('torso').xpos)

            # Check for success:
            if (abs(d.jnt('torso_joint').qpos[0]) > success_dist or abs(d.jnt('torso_joint').qpos[1]) > success_dist) and torso_head_angle_offset<torso_head_angle_tolerance:
                num_successes += 1
                print('Successes: ' + str(num_successes) + " Failures: " + str(num_failures))
                break

            # Check for failure:
            rot = Rotation.from_quat(d.body('torso').xquat)
            angles = rot.as_euler('xyz', degrees=True)
            if(abs(angles[2])>110 or time.time()-start>10):
                num_failures += 1
                print('Successes: ' + str(num_successes) + " Failures: " + str(num_failures))
                break

        
            # Sync changes in the viewer
            viewer.sync()
            
            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step*0.005)

print(f"Successes: {num_successes}")
print(f"Failures: {num_failures}")
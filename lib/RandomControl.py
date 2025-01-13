import time
import mujoco
import mujoco.viewer
import numpy as np
import lib.MotorModel as motor

class RandomController:
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, motors: motor.MotorModel,rng: np.random.Generator):
        
        # Adding mjmodel, data, and motors to class:
        self.m = m
        self.d = d
        self.motors = motors
        self.rng = rng

        # Maximum values for control randomization:
        self.min_knee_vel = 0.001
        self.max_knee_vel = 0.008
        self.min_wheel_vel = 0.05
        self.max_wheel_vel = 0.3

        # Initializing random velocity variables:
        self.left_knee_des_vel = 0
        self.right_knee_des_vel = 0
        self.left_wheel_des_vel = 0
        self.right_wheel_des_vel = 0
        
        # Get current pose and set as initial desired knee/hip positions:
        self.get_joint_state()
        self.set_ICs()


    def get_joint_state(self):
        # Get current joint angles
        self.fr_hip_pos = self.d.jnt('head_right_thigh_joint').qpos[0]
        self.fl_hip_pos = self.d.jnt('head_left_thigh_joint').qpos[0]
        self.br_hip_pos = self.d.jnt('torso_right_thigh_joint').qpos[0]
        self.bl_hip_pos = self.d.jnt('torso_right_thigh_joint').qpos[0]

        self.fr_knee_pos = self.d.jnt('head_right_thigh_shin_joint').qpos[0]
        self.fl_knee_pos = self.d.jnt('head_left_thigh_shin_joint').qpos[0]
        self.br_knee_pos = self.d.jnt('torso_right_thigh_shin_joint').qpos[0]
        self.bl_knee_pos = self.d.jnt('torso_left_thigh_shin_joint').qpos[0]

    # Set desired position to be the same as current position
    def set_ICs(self):
        self.fr_hip_des_pos = self.fr_hip_pos
        self.fl_hip_des_pos = self.fl_hip_pos
        self.br_hip_des_pos = self.br_hip_pos
        self.bl_hip_des_pos = self.bl_hip_pos

        self.fr_knee_des_pos = self.fr_knee_pos
        self.fl_knee_des_pos = self.fl_knee_pos
        self.br_knee_des_pos = self.br_knee_pos
        self.bl_knee_des_pos = self.bl_knee_pos

    def randomize_control(self):
        self.left_knee_des_vel = self.rng.uniform(self.min_knee_vel, self.max_knee_vel)
        self.right_knee_des_vel = self.rng.uniform(self.min_knee_vel, self.max_knee_vel)
        self.left_wheel_des_vel = self.rng.uniform(self.min_wheel_vel, self.max_wheel_vel)
        self.right_wheel_des_vel = self.rng.uniform(self.min_wheel_vel, self.max_wheel_vel)

    def integrate_pos(self):
        self.fr_knee_des_pos += self.right_knee_des_vel
        self.fl_knee_des_pos += self.left_knee_des_vel
        self.br_knee_des_pos += self.right_knee_des_vel
        self.bl_knee_des_pos += self.left_knee_des_vel


    # Function that sends the commands to the motors
    def send_commands(self):
        self.motors[0].pos_control(self.fr_hip_des_pos)
        self.motors[1].pos_control(self.fl_hip_des_pos)
        self.motors[2].pos_control(self.br_hip_des_pos)
        self.motors[3].pos_control(self.bl_hip_des_pos)

        self.motors[4].pos_control(self.fr_knee_des_pos)
        self.motors[5].pos_control(self.fl_knee_des_pos)
        self.motors[6].pos_control(self.br_knee_des_pos)
        self.motors[7].pos_control(self.bl_knee_des_pos)

        self.motors[9].vel_control(self.right_wheel_des_vel)
        self.motors[8].vel_control(self.right_wheel_des_vel)
        self.motors[10].vel_control(self.left_wheel_des_vel)
        self.motors[11].vel_control(self.left_wheel_des_vel)
        self.motors[12].vel_control(self.right_wheel_des_vel)
        self.motors[13].vel_control(self.right_wheel_des_vel)
        self.motors[14].vel_control(self.left_wheel_des_vel)
        self.motors[15].vel_control(self.left_wheel_des_vel)

    # Final function to send control commands to the motors
    def control(self):
        self.integrate_pos()
        self.send_commands()
from typing import Any
from absl import app
import os
import sys
from pathlib import Path
import yaml
import numpy as np
import mujoco
import scipy
sys.path.append(os.path.dirname(__file__))
import MotorModel as motor

# 2D Model will appear in the X-Z plane

# Model parameters:
test_model_params = {
    'body': {
        'length': 0.5,
        'mass': 1.0,
    },
    'front_thigh': {
        'length': 0.5,
        'mass': 1.0,
    },
    'front_shin': {    
        'length': 0.5,
        'mass': 1.0,
    },
    'front_wheel': {
        'radius': 0.1,
        'mass': 1.0,
    },
    'rear_thigh': {
        'length': 0.5,
        'mass': 1.0,
    },
    'rear_shin': {    
        'length': 0.5,
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
        'Kp': 100.0,
        'Kd': 0.1,
        'stall_torque': 100.0,
        'no_load_speed': 10.0,
        'gear_ratio': 1.0,
        'rotor_inertia': 0.001,
    },
    'front_knee': {
        'Kp': 100.0,
        'Kd': 0.1,
        'stall_torque': 100.0,
        'no_load_speed': 10.0,
        'gear_ratio': 1.0,
        'rotor_inertia': 0.001,
    },
    'front_wheel': {
        'Kp': 10.0,
        'Kd': 0.1,
        'stall_torque': 10.0,
        'no_load_speed': 10.0,
        'rotor_inertia': 0.001,
        'gear_ratio': 1.0,

    },
    'rear_hip': {
        'Kp': 100.0,
        'Kd': 0.1,
        'stall_torque': 100.0,
        'no_load_speed': 10.0,
        'rotor_inertia': 0.001,
        'gear_ratio': 1.0,
    },
    'rear_knee': {
        'Kp': 100.0,
        'Kd': 0.1,
        'stall_torque': 100.0,
        'no_load_speed': 10.0,
        'rotor_inertia': 0.001,
        'gear_ratio': 1.0,
    },
    'rear_wheel': {
        'Kp': 10.0,
        'Kd': 0.1,
        'stall_torque': 10.0,
        'no_load_speed': 10.0,
        'rotor_inertia': 0.001,
        'gear_ratio': 1.0,
    }
}

class WaLTER2D():

    def __init__(self, model_params: dict, motor_params: dict):
        self.model_params = model_params
        self.motor_params = motor_params
        self.walter = None
        self.m = None
        self.d = None
        self.spec = mujoco.MjSpec()

        body_contype = 2
        body_conaffinity = 1
        thigh_contype = 2
        thigh_conaffinity = 1
        shin_contype = 2
        shin_conaffinity = 1
        wheel_contype = 4
        wheel_conaffinity = 5
        world_contype = 1
        world_conaffinity = 1

        # Calculate joint armatures:
        f_hip_armature = self.motor_params['front_hip']['rotor_inertia']*self.motor_params['front_hip']['gear_ratio']**2
        f_knee_armature = self.motor_params['front_knee']['rotor_inertia']*self.motor_params['front_knee']['gear_ratio']**2
        f_wheel_armature = self.motor_params['front_wheel']['rotor_inertia']*self.motor_params['front_wheel']['gear_ratio']**2
        r_hip_armature = self.motor_params['rear_hip']['rotor_inertia']*self.motor_params['rear_hip']['gear_ratio']**2
        r_knee_armature = self.motor_params['rear_knee']['rotor_inertia']*self.motor_params['rear_knee']['gear_ratio']**2
        r_wheel_armature = self.motor_params['rear_wheel']['rotor_inertia']*self.motor_params['rear_wheel']['gear_ratio']**2

        # Generating the basic 2D WaLTER model:m
        torso_body = self.spec.worldbody.add_body(
            name = 'torso',
            pos = [0, 0, 1],
            quat = [1, 0, 0, 0],

        )
        torso_body.add_geom(
            type = mujoco.mjtGeom.mjGEOM_CAPSULE,
            size = [self.model_params['general']['width']/2, self.model_params['body']['length']/2, 0],
            quat = [1, 0, 1, 0],
            mass = self.model_params['body']['mass'],
            contype = body_contype,
            conaffinity = body_conaffinity,
        )
        torso_body.add_joint(
            type = mujoco.mjtJoint.mjJNT_SLIDE,
            name = 'z_slide'
        )
        torso_body.add_joint(
            type = mujoco.mjtJoint.mjJNT_SLIDE,
            axis = [1, 0, 0],
            name = 'x_slide'
        )
        torso_body.add_joint(
            type = mujoco.mjtJoint.mjJNT_HINGE,
            axis = [0, 1, 0],
            name = 'y_rot'
        )


        # Creating the front thigh:
        front_thigh_pos = [self.model_params['body']['length']/2, 0, -self.model_params['front_thigh']['length']/2]
        front_thigh = torso_body.add_body(
            name = 'front_thigh',
            quat = [1, 0, 0, 0],
            pos = front_thigh_pos,
        )
        front_thigh.add_geom(
            type = mujoco.mjtGeom.mjGEOM_CAPSULE,
            size = [ self.model_params['general']['width']/2, self.model_params['front_thigh']['length']/2,0],
            mass = self.model_params['front_thigh']['mass'],
            contype = thigh_contype,
            conaffinity = thigh_conaffinity,
        )
        front_thigh.add_joint(
            type = mujoco.mjtJoint.mjJNT_HINGE,
            axis = [0, 1, 0],
            pos = [0, 0, self.model_params['front_thigh']['length']/2],
            name = 'front_hip',
            armature = f_hip_armature
        )

        # Creating the front shin:
        front_shin_pos = [0, 0, -self.model_params['front_shin']['length']/2]
        front_shin = front_thigh.add_body(
            name = 'front_shin',
            quat = [1, 0, 1, 0],
            pos = front_shin_pos,
        )
        front_shin.add_geom(
            type = mujoco.mjtGeom.mjGEOM_CAPSULE,
            size = [ self.model_params['general']['width']/2, self.model_params['front_shin']['length']/2,0],
            mass = self.model_params['front_shin']['mass'],
            contype = shin_contype,
            conaffinity = shin_conaffinity,
        )
        front_shin.add_joint(
            type = mujoco.mjtJoint.mjJNT_HINGE,
            axis = [0, 1, 0],
            pos = [0, 0, 0],
            name = 'front_knee',
            armature = f_knee_armature,
        )

        # Creating the front wheels:
        front_wheel1_pos = [0, 0, self.model_params['front_shin']['length']/2]
        front_wheel1 = front_shin.add_body(
            name = 'front_wheel1',
            quat = [1, 0, 0, 0],
            pos = front_wheel1_pos,
        )
        front_wheel1.add_geom(
            type = mujoco.mjtGeom.mjGEOM_SPHERE,
            size = [self.model_params['front_wheel']['radius'], 0, 0],
            mass = self.model_params['front_wheel']['mass'],
            contype = wheel_contype,
            conaffinity = wheel_conaffinity,
        )
        front_wheel1.add_joint(
            type = mujoco.mjtJoint.mjJNT_HINGE,
            axis = [0, 1, 0],
            name = 'front_wheel1_joint',
            armature = f_wheel_armature,
        )
        front_wheel2_pos = [0, 0, -self.model_params['front_shin']['length']/2]
        front_wheel2 = front_shin.add_body(
            name = 'front_wheel2',
            quat = [1, 0, 0, 0],
            pos = front_wheel2_pos,
        )
        front_wheel2.add_geom(
            type = mujoco.mjtGeom.mjGEOM_SPHERE,
            size = [self.model_params['front_wheel']['radius'], 0, 0],
            mass = self.model_params['front_wheel']['mass'],
            contype = wheel_contype,
            conaffinity = wheel_conaffinity,
        )
        front_wheel2.add_joint(
            type = mujoco.mjtJoint.mjJNT_HINGE,
            axis = [0, 1, 0],
            name = 'front_wheel2_joint',
            armature = f_wheel_armature,
        )

        # Creating the rear thigh:
        rear_thigh_pos = [-self.model_params['body']['length']/2, 0, -self.model_params['rear_thigh']['length']/2]
        rear_thigh = torso_body.add_body(
            name = 'rear_thigh',
            quat = [1, 0, 0, 0],
            pos = rear_thigh_pos,
        )
        rear_thigh.add_geom(
            type = mujoco.mjtGeom.mjGEOM_CAPSULE,
            size = [ self.model_params['general']['width']/2, self.model_params['rear_thigh']['length']/2,0],
            mass = self.model_params['rear_thigh']['mass'],
            contype = thigh_contype,
            conaffinity = thigh_conaffinity,
        )
        rear_thigh.add_joint(
            type = mujoco.mjtJoint.mjJNT_HINGE,
            axis = [0, 1, 0],
            pos = [0, 0, self.model_params['rear_thigh']['length']/2],
            name = 'rear_hip',
            armature = r_hip_armature,
        )

        # Creating the rear shin:
        rear_shin_pos = [0, 0, -self.model_params['rear_shin']['length']/2]
        rear_shin = rear_thigh.add_body(
            name = 'rear_shin',
            quat = [1, 0, 1, 0],
            pos = rear_shin_pos,
        )
        rear_shin.add_geom(
            type = mujoco.mjtGeom.mjGEOM_CAPSULE,
            size = [ self.model_params['general']['width']/2, self.model_params['rear_shin']['length']/2,0],
            mass = self.model_params['rear_shin']['mass'],
            contype = shin_contype,
            conaffinity = shin_conaffinity,
        )
        rear_shin.add_joint(
            type = mujoco.mjtJoint.mjJNT_HINGE,
            axis = [0, 1, 0],
            pos = [0, 0, 0],
            name = 'rear_knee',
            armature = r_knee_armature,
        )

        # Creating the rear wheels:
        rear_wheel1_pos = [0, 0, self.model_params['rear_shin']['length']/2]
        rear_wheel1 = rear_shin.add_body(
            name = 'rear_wheel1',
            quat = [1, 0, 0, 0],
            pos = rear_wheel1_pos,
        )
        rear_wheel1.add_geom(
            type = mujoco.mjtGeom.mjGEOM_SPHERE,
            size = [self.model_params['rear_wheel']['radius'], 0, 0],
            mass = self.model_params['rear_wheel']['mass'],
            contype = wheel_contype,
            conaffinity = wheel_conaffinity,
        )
        rear_wheel1.add_joint(
            type = mujoco.mjtJoint.mjJNT_HINGE,
            axis = [0, 1, 0],
            name = 'rear_wheel1_joint',
            armature = r_wheel_armature,
        )
        rear_wheel2_pos = [0, 0, -self.model_params['rear_shin']['length']/2]
        rear_wheel2 = rear_shin.add_body(
            name = 'rear_wheel2',
            quat = [1, 0, 0, 0],
            pos = rear_wheel2_pos,
        )
        rear_wheel2.add_geom(
            type = mujoco.mjtGeom.mjGEOM_SPHERE,
            size = [self.model_params['rear_wheel']['radius'], 0, 0],
            mass = self.model_params['rear_wheel']['mass'],
            contype = wheel_contype,
            conaffinity = wheel_conaffinity,
        )
        rear_wheel2.add_joint(
            type = mujoco.mjtJoint.mjJNT_HINGE,
            axis = [0, 1, 0],
            name = 'rear_wheel2_joint',
            armature = r_wheel_armature,
        )

        # Assigning actuators
        self.spec.add_actuator(
            name = 'f_hip',
            target = 'front_hip',
            trntype = mujoco.mjtTrn.mjTRN_JOINT,
        )
        self.spec.add_actuator(
            name = 'f_knee',
            target = 'front_knee',
            trntype = mujoco.mjtTrn.mjTRN_JOINT,
        )
        self.spec.add_actuator(
            name = 'f_wheel1',
            target = 'front_wheel1_joint',
            trntype = mujoco.mjtTrn.mjTRN_JOINT,
        )
        self.spec.add_actuator(
            name = 'f_wheel2',
            target = 'front_wheel2_joint',
            trntype = mujoco.mjtTrn.mjTRN_JOINT,
        )
        self.spec.add_actuator(
            name = 'r_hip',
            target = 'rear_hip',
            trntype = mujoco.mjtTrn.mjTRN_JOINT,
        )
        self.spec.add_actuator(
            name = 'r_knee',
            target = 'rear_knee',
            trntype = mujoco.mjtTrn.mjTRN_JOINT,
        )
        self.spec.add_actuator(
            name = 'r_wheel1',
            target = 'rear_wheel1_joint',
            trntype = mujoco.mjtTrn.mjTRN_JOINT,
        )
        self.spec.add_actuator(
            name = 'r_wheel2',
            target = 'rear_wheel2_joint',
            trntype = mujoco.mjtTrn.mjTRN_JOINT,
        )
    
    def add_motors(self,m:mujoco.MjModel,d:mujoco.MjData,motor_params: dict):
        self.f_hip_motor = motor.MotorModel(m, d, 'front_hip', motor_params['front_hip'], 0)
        self.f_knee_motor = motor.MotorModel(m, d, 'front_knee', motor_params['front_knee'], 1)
        self.f_wheel1_motor = motor.MotorModel(m, d, 'front_wheel1_joint', motor_params['front_wheel'], 2)
        self.f_wheel2_motor = motor.MotorModel(m, d, 'front_wheel2_joint', motor_params['front_wheel'], 3)
        self.r_hip_motor = motor.MotorModel(m, d, 'rear_hip', motor_params['rear_hip'], 4)
        self.r_knee_motor = motor.MotorModel(m, d, 'rear_knee', motor_params['rear_knee'], 5)
        self.r_wheel1_motor = motor.MotorModel(m, d, 'rear_wheel1_joint', motor_params['rear_wheel'], 6)
        self.r_wheel2_motor = motor.MotorModel(m, d, 'rear_wheel2_joint', motor_params['rear_wheel'], 7)
        self.motors = [self.f_hip_motor, self.f_knee_motor, self.f_wheel1_motor, self.f_wheel2_motor, self.r_hip_motor, self.r_knee_motor, self.r_wheel1_motor, self.r_wheel2_motor]
        
    def tumble_drive(self):
        self.f_hip_motor.pos_control(-np.pi/4)
        self.f_knee_motor.vel_control(1)
        self.f_wheel1_motor.vel_control(1)
        self.f_wheel2_motor.vel_control(1)
        self.r_hip_motor.pos_control(np.pi/4)
        self.r_knee_motor.vel_control(1)
        self.r_wheel1_motor.vel_control(1)
        self.r_wheel2_motor.vel_control(1)

    def gen_scene(self):
        # Create ground plane texture/material
        ground = self.spec.add_texture(type = mujoco.mjtTexture.mjTEXTURE_2D,
                              name="ground_texture",
                              builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, 
                              width=200, 
                              height=200, 
                              rgb1=[0.5, 0.8, 0.9], 
                              rgb2=[0.5, 0.9, 0.8],
                              markrgb=[0.8, 0.8, 0.8])
        
        self.spec.add_material(name="groundplane",
                              texrepeat=[2, 2],
                              reflectance=0., 
                              ).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'ground_texture'
        
        self.spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 0.05],
            material="groundplane",
        )


        # Create skybox so background isn't just black
        self.spec.add_texture(type = mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                              builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                                width = 300,
                                height = 300,
                                name="skybox")
        # Add an array of lights to the scene:
        for i in range(5):
            for j in range(5):
                self.spec.worldbody.add_light(
                    pos=[2*i, 2*j, 15],
                    dir=[0, 0, -1],
                    diffuse=[0.1, 0.1, 0.1],
                    specular=[0., 0., 0.],
                    directional=True,
                )

    def add_box(self, pos:list, size:list):
        self.spec.worldbody.add_body(pos=pos).add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=size)

    def add_stairs(self, pos: list = [2,0,0], rise: float = 0.1, run: float = 0.1, width: float=1.2, num_steps: int=5):
        for i in range(num_steps):
            self.add_box(
                pos=[pos[0]+i*run, pos[1], pos[2] + i*rise],
                size=[run, width, rise],
            )

    def compile_to_XML(self):
        """
        Compiles current model to XML file
        """
        self.spec.compile()
        xml_path = os.path.join(os.path.dirname(__file__), '2D_WaLTER.xml')
        with open(xml_path, 'w') as f:
            f.write(self.spec.to_xml())
        

def main():
    walter = WaLTER2D(test_model_params, test_model_params)
    walter.gen_scene()
    walter.compile_to_XML()

if __name__ == '__main__':
    main()
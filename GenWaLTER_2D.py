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



class GenWaLTER2D():

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
            name = 'front_hip'
        )

        # Creating the front shin:
        front_shin_pos = [0, 0, -self.model_params['front_shin']['length']/2]
        front_shin = front_thigh.add_body(
            name = 'front_shin',
            quat = [1, 0, 0, 0],
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
            name = 'front_knee'
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
            name = 'front_wheel1_joint'
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
            name = 'front_wheel2_joint'
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
            name = 'rear_hip'
        )

        # Creating the rear shin:
        rear_shin_pos = [0, 0, -self.model_params['rear_shin']['length']/2]
        rear_shin = rear_thigh.add_body(
            name = 'rear_shin',
            quat = [1, 0, 0, 0],
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
            name = 'rear_knee'
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
            name = 'rear_wheel1_joint'
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
            name = 'rear_wheel2_joint'
        )

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

    def compile_to_XML(self):
        self.spec.compile()
        xml_path = os.path.join(os.path.dirname(__file__), '2D_WaLTER.xml')
        with open(xml_path, 'w') as f:
            f.write(self.spec.to_xml())
        

def main():
    walter = GenWaLTER2D(test_model_params, test_model_params)
    walter.gen_scene()
    walter.compile_to_XML()

if __name__ == '__main__':
    main()
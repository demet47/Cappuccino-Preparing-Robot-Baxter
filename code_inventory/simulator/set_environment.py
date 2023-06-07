import pybullet as p
import pybullet_data
from manipulator import Manipulator
import pandas as pd
import torch
import os
import pdb
import numpy as np
import sys

'''
This file provides functions to create pybullet environment.
**create_baxter** function creates a baxter corresponding to the one in lab in scale
**add_custom_object** function places the objects specified by urdf file name, custom locations and orientations and custom scale.
For .urdf files, pybullet looks at the directory we get when we run pybullet_data.getDataPath() function.
'''


def create_baxter():
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId=p.loadURDF("plane.urdf")
    p.setGravity(0,0,-9.81)
    p.setRealTimeSimulation(0)
    baxter = Manipulator(p, "C:/Users/dmtya/Cappuccino-Preparing-Robot-Baxter/baxter_common/baxter_description/urdf/toms_baxter.urdf", position=(0,0,0.9), ik_idx=20)
    baxter.add_debug_text()
    joint_names = [joint_name.decode() for joint_name in baxter.names]
    return baxter, p



def add_custom_objects():
    tableId = p.loadURDF("table.urdf", [1,0,0.12], [0,0,1,1], globalScaling = 1.2)
    cubeId = p.loadURDF("cube.urdf", [1,0,0.97], [0,0,0,1], globalScaling = 0.2)
    




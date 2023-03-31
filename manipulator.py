import time

import numpy as np


class Manipulator:
    def __init__(self, p, path, position=(0, 0, 0), orientation=(0, 0, 0, 1), ik_idx=-1):
        self._p = p
        self._timestep = self._p.getPhysicsEngineParameters()["fixedTimeStep"]
        self._freq = int(1. / self._timestep)
        self.id = self._p.loadURDF(
            fileName=path,
            basePosition=position,
            baseOrientation=orientation,
            useFixedBase=True)
        self.ik_idx = ik_idx
        self.joints = []
        self.names = []
        self.forces = []
        self.fixed_joints = []
        self.fixed_names = []
        for i in range(self._p.getNumJoints(self.id)):
            info = self._p.getJointInfo(self.id, i)
            if info[2] != self._p.JOINT_FIXED:
                self.joints.append(i)
                self.names.append(info[1])
                self.forces.append(info[10])
            else:
                self.fixed_joints.append(i)
                self.fixed_names.append(info[1])
        self.joints = tuple(self.joints)
        self.names = tuple(self.names)
        self.num_joints = len(self.joints)
        self.debug_params = []
        self.child = None
        self.constraints = []
        for j in self.joints:
            self._p.enableJointForceTorqueSensor(self.id, j, 1)
        self.forces = [5000 for _ in self.joints]

    def get_end_effector_pose(self): #!!!!
        p, _, _, _, _,_ = self._p.getLinkState(self.id, 26, computeForwardKinematics=True)
        return p
    
    def add_debug_text(self, text='gripper_base', color=[200,50,50]):
        self._p.addUserDebugText(text, [0, 0, 0.1],
                   textColorRGB=color,
                   textSize=1.5,
                   parentObjectUniqueId=self.id,
                   parentLinkIndex=self.ik_idx)
    
    def add_line(self, start, end, color=[1., .5, .5], lineWidth=5):
        return self._p.addUserDebugLine(start, end, lineColorRGB=color, lineWidth=lineWidth)

    def set_cartesian_position(self, position, orientation=None, t=None, sleep=False, traj=False): #the position here is in cartesian coordinate system
        target_joints = self._p.calculateInverseKinematics(
            bodyUniqueId=self.id,
            endEffectorLinkIndex=self.ik_idx,
            targetPosition=position,
            targetOrientation=orientation)
        self.set_joint_position(target_joints, t=t, sleep=sleep, traj=traj)

    def move_in_cartesian(self, position, orientation=None, t=1.0, sleep=False, ignore_force=False):
        N = int(t * 240)
        current_position = self.get_end_effector_pose()
        position_traj = np.linspace(current_position, position, N+1)[1:]
        for p_i in position_traj:
            target_joints = self._p.calculateInverseKinematics(
                bodyUniqueId=self.id,
                endEffectorLinkIndex=self.ik_idx,
                targetPosition=p_i,
                targetOrientation=orientation)
            self.set_joint_position(target_joints, t=1/240, sleep=sleep)
            
    def set_joint_position(self, position, velocity=None, t=None, sleep=False, traj=False):
        assert len(self.joints) > 0
        if traj:
            assert (t is not None)
            N = int(t * 240)
            current_position = self.get_joint_position()
            trajectory = np.linspace(current_position, position, N)
            for t_i in trajectory:
                self._p.setJointMotorControlArray(
                    bodyUniqueId=self.id,
                    jointIndices=self.joints,
                    controlMode=self._p.POSITION_CONTROL,
                    targetPositions=t_i,
                    forces=self.forces)
                self._p.stepSimulation()
                if sleep:
                    self._waitsleep(1/240, sleep)
        else:
            if velocity is not None:
                self._p.setJointMotorControlArray(
                    bodyUniqueId=self.id,
                    jointIndices=self.joints,
                    controlMode=self._p.POSITION_CONTROL,
                    targetPositions=position,
                    targetVelocities=velocity,
                    forces=self.forces)
            else:
                self._p.setJointMotorControlArray(
                    bodyUniqueId=self.id,
                    jointIndices=self.joints,
                    controlMode=self._p.POSITION_CONTROL,
                    targetPositions=position,
                    forces=self.forces)
            self._waitsleep(t, sleep)

    def set_joint_velocity(self, velocity, t=None, sleep=False):
        assert len(self.joints) > 0
        self._p.setJointMotorControlArray(
            bodyUniqueId=self.id,
            jointIndices=self.joints,
            controlMode=self._p.VELOCITY_CONTROL,
            targetVelocities=velocity,
            forces=self.forces)
        self._waitsleep(t, sleep)

    def set_joint_torque(self, torque, t=None, sleep=False):
        assert len(self.joints) > 0
        self._p.setJointMotorControlArray(
            bodyUniqueId=self.id,
            jointIndices=self.joints,
            controlMode=self._p.TORQUE_CONTROL,
            forces=torque)
        self._waitsleep(t, sleep)

    # TODO: make this only joint position, joint velocity etc.
    def get_joint_states(self):
        return self._p.getJointStates(self.id, self.joints)

    def get_joint_position(self): #!!!!
        joint_states = self.get_joint_states()
        return [joint[0] for joint in joint_states]

    def get_joint_forces(self): 
        joint_states = self.get_joint_states()
        return [joint[2][2] for joint in joint_states]

    def add_debug_param(self):
        current_angle = [j[0] for j in self.get_joint_states()]
        for i in range(self.num_joints):
            joint_info = self._p.getJointInfo(self.id, self.joints[i])
            low, high = joint_info[8:10]
            self.debug_params.append(self._p.addUserDebugParameter(self.names[i].decode("utf-8"),
                                     low, high, current_angle[i]))

    def update_debug(self):
        target_angles = []
        for param in self.debug_params:
            try:
                angle = self._p.readUserDebugParameter(param)
                target_angles.append(angle)
            except Exception:
                break
        if len(target_angles) == len(self.joints):
            self.set_joint_position(target_angles)

    def _waitsleep(self, t, sleep=False):
        if t is not None:
            iters = int(t*self._freq)
            for _ in range(iters):
                self._p.stepSimulation()
                if sleep:
                    time.sleep(self._timestep)

    def convert_to_joint_data(self, position_xyz, orientation):
        return self._p.calculateInverseKinematics(
                bodyUniqueId=self.id,
                endEffectorLinkIndex=self.ik_idx,
                targetPosition=position_xyz,
                targetOrientation=orientation)
        
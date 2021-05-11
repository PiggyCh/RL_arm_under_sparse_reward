
'''
2021.4.13
PiggyCh
夹爪动作连续版本
'''
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as p
import numpy as np
import copy
import math
import pybullet_data
import bmirobot_env.bmirobot_inverse_kinematics as inverse
import  time

class bmirobotv0:

  def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01):
    self.urdfRootPath = urdfRootPath
    self.timeStep = timeStep
    self.maxVelocity = .35
    self.maxForce = 200.
    self.fingerAForce = 2
    self.fingerBForce = 2.5
    self.fingerTipForce = 2
    self.useInverseKinematics = 1
    self.useSimulation = 1
    self.useNullSpace = 21
    self.useOrientation = 1
    self.bmirobot_righthand= 11
    #lower limits for null space
    self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
    #upper limits for null space
    self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
    #joint ranges for null space
    self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
    #restposes for null space
    self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
    #joint damping coefficents
    self.jd = [
        0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
        0.00001, 0.00001, 0.00001, 0.00001
    ]
    self.limit_x=[-1,1]
    self.limit_y=[-1,1]
    self.limit_z=[0,1]
    self.reset()

  def reset(self):
    #load model
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf")
    objects = p.loadURDF('URDF_model/bmirobot_description/urdf/robotarm_description.urdf',flags=9)
    #get bmirobot id
    self.bmirobotid = objects
    #for i in range (p.getNumJoints(self.kukaUid)):
    #  print(p.getJointInfo(self.kukaUid,i))
    #reset original position
    p.resetBasePositionAndOrientation(self.bmirobotid, [-0.100000, 0.000000, 0.070000],
                                      [0.000000, 0.000000, 0.000000, 1.000000])
    #reset joint positions
    self.jointPositions = [0]*24
    self.numJoints = p.getNumJoints(self.bmirobotid)
    for jointIndex in range(self.numJoints):
      p.resetJointState(self.bmirobotid, jointIndex, self.jointPositions[jointIndex])
      p.setJointMotorControl2(self.bmirobotid,
                              jointIndex,
                              p.POSITION_CONTROL,
                              targetPosition=self.jointPositions[jointIndex],
                              force=self.maxForce)
    #这里的table的初始位姿需要进行调整
    self.tableUid = p.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"), basePosition=[0,0.3,-0.45],globalScaling=1)
    #这里的末端位置是由 原模型给出，可能需要再考虑? 而且如果是两个臂,应该分别写数据，这里只用到右臂
    self.endEffectorPos = [0]*3
    self.endEffectorAngle = 0

    self.motorNames = []
    self.motorIndices = []

    for i in range(self.numJoints):
      jointInfo = p.getJointInfo(self.bmirobotid, i)
      qIndex = jointInfo[3]
      #这里的q还不是很清楚意思
      if qIndex > -1:
        #print("motorname")
        #print(jointInfo[1])
        self.motorNames.append(str(jointInfo[1]))
        self.motorIndices.append(i)

  def getActionDimension(self):
    if (self.useInverseKinematics):
      return len(self.motorIndices)
    return 6  #position x,y,z and roll/pitch/yaw euler angles of end effector

  def getObservationDimension(self):
    return len(self.getObservation())

  def getObservation(self):
    observation = []
    state = p.getLinkState(self.bmirobotid, self.bmirobot_righthand)
    pos = state[4]
    pos=list(pos)
    #做为一个补偿，因为末端实际上在手中间。 但这里还是应该再仔细考虑一下，因为这个补偿量应该取决于末端的姿态角度
    orn = state[5]
    euler = p.getEulerFromQuaternion(orn)

    observation.extend(list(pos))
    observation.extend(list(euler))

    return observation
  def get_to_place(self,position):
      orn=[]
      jointPoses = inverse.getinversePoisition(self.bmirobotid, position, orn)
      for i in range(3,12):
            p.setJointMotorControl2(bodyIndex=self.bmirobotid,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i-3],
                                    targetVelocity=0,
                                    force=500,
                                    positionGain=0.03,
                                    velocityGain=1)
      return
  def applyAction(self, motorCommands): #4 actiions
      limit_x=[-1,1]
      limit_y=[-1,1]
      limit_z=[0,1]
      def clip_val(val,limit):
          if val<limit[0]:
              return limit[0]
          if val>limit[1]:
              return limit[1]
          return val
      if (self.useInverseKinematics):
          dx = motorCommands[0]
          dy = motorCommands[1]
          dz = motorCommands[2]
          fingerAngle = motorCommands[3]  # fingerangle应该是末端开口大小
          state = p.getLinkState(self.bmirobotid, self.bmirobot_righthand)  # 似乎并非如此，此处是getlinkstate
          actualEndEffectorPos = state[4]
          self.endEffectorPos[0] = clip_val(actualEndEffectorPos[0] + dx,limit_x)
          self.endEffectorPos[1] = clip_val(actualEndEffectorPos[1] + dy,limit_y)
          self.endEffectorPos[2] = clip_val(actualEndEffectorPos[2] + dz,limit_z)
          orn = [0, 0, 0]
          pos = self.endEffectorPos
          jointPoses = inverse.getinversePoisition(self.bmirobotid, pos)
          if (self.useSimulation):
              for i in range(3, 10):
                  p.setJointMotorControl2(bodyIndex=self.bmirobotid,
                                          jointIndex=i,
                                          controlMode=p.POSITION_CONTROL,
                                          targetPosition=jointPoses[i - 3],
                                          targetVelocity=0,
                                          force=500,
                                          positionGain=0.03,
                                          velocityGain=1)
          self.sent_hand_moving(fingerAngle)
  def sent_hand_moving(self,moterCommand):
      right_hand_joint = 10
      right_hand_joint2 = 11
      limit = [-1.57079632679, 1.57079632679]
      right_hand_joint_now = p.getJointState(self.bmirobotid, right_hand_joint)[0]
      right_hand_joint2_now = p.getJointState(self.bmirobotid, right_hand_joint2)[0]
      moterCommand_1 = right_hand_joint_now + moterCommand
      moterCommand_2 = right_hand_joint2_now - moterCommand
      #print("目前： ", str(right_hand_joint_now), " 到： ", str(moterCommand_1))
      # if moterCommand > 0:
      #     moterCommand = limit[1]
      # if moterCommand <= 0:
      #     moterCommand = limit[0]
      p.setJointMotorControl2(bodyIndex=self.bmirobotid,
                              jointIndex=right_hand_joint,
                              controlMode=p.POSITION_CONTROL,
                              targetPosition=moterCommand_1,
                              targetVelocity=0,
                              force=500,
                              positionGain=0.03,
                              velocityGain=1)
      p.setJointMotorControl2(bodyIndex=self.bmirobotid,
                              jointIndex=right_hand_joint2,
                              controlMode=p.POSITION_CONTROL,
                              targetPosition=moterCommand_2,
                              targetVelocity=0,
                              force=500,
                              positionGain=0.03,
                              velocityGain=1)

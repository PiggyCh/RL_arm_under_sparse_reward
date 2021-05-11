import pybullet as p
import time
import math
from datetime import datetime
import pybullet_data
#position_desired[]
def getinversePoisition(bmirobot_id,position_desired,orientation_desired=[]):
    joints_info = []
    joint_damping = []
    joint_ll = []
    joint_ul = []
    useOrientation=len(orientation_desired)
    for i in range(24):
        joints_info.append(p.getJointInfo(bmirobot_id, i))
    bmirobotEndEffectorIndex = 11 #也有可能是9 或10
    numJoints = p.getNumJoints(bmirobot_id)
    useNullSpace = 1
    ikSolver = 1
    trailDuration = 30
    pos = [position_desired[0], position_desired[1], position_desired[2]]
    #end effector points down, not up (in case useOrientation==1)
    if useOrientation:
        orn = p.getQuaternionFromEuler([orientation_desired[0],orientation_desired[1] , orientation_desired[2]])
    if (useNullSpace == 1):
      if (useOrientation == 1):
        jointPoses = p.calculateInverseKinematics(bmirobot_id, bmirobotEndEffectorIndex, pos, orn)
      else:
        jointPoses = p.calculateInverseKinematics(bmirobot_id,
                                                  bmirobotEndEffectorIndex,
                                                  pos,
                                                  lowerLimits=joint_ll,
                                                  upperLimits=joint_ul,
                                               )
        # print(jointPoses)
    else:
      if (useOrientation == 1):
        jointPoses = p.calculateInverseKinematics(bmirobot_id,
                                                  bmirobotEndEffectorIndex,
                                                  pos,
                                                  orn,
                                                  solver=ikSolver,
                                                  maxNumIterations=100,
                                                  residualThreshold=.01)
        # print(jointPoses)
      else:
        jointPoses = p.calculateInverseKinematics(bmirobot_id,
                                                  bmirobotEndEffectorIndex,
                                                  pos,
                                                  solver=ikSolver)
    return jointPoses

    # if (useSimulation):
    #   for i in range(3,12):
    #     p.setJointMotorControl2(bodyIndex=bmirobot_id,
    #                             jointIndex=i,
    #                             controlMode=p.POSITION_CONTROL,
    #                             targetPosition=jointPoses[i-3],
    #                             targetVelocity=0,
    #                             force=500,
    #                             positionGain=0.03,
    #                             velocityGain=1)
#     else:
#       #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
#       for i in range(numJoints):
#         p.resetJointState(bmirobot_id, i, jointPoses[i])
# #
#   ls = p.getLinkState(bmirobot_id, bmirobotEndEffectorIndex)
#   if (hasPrevPose):
#     p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 1, trailDuration)
#     p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, trailDuration)
#   prevPose = pos
#   prevPose1 = ls[4]
#   hasPrevPose = 1
# p.disconnect()
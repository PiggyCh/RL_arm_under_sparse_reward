import sys
import pybullet as p
import numpy as np
import math
import pybullet_data
import time
import random
import os
import inspect
from arguments import Args
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # os.path.abspath() 获得文件的绝对路径
parentdir = os.path.dirname(os.path.dirname(currentdir))  # os.path.dirname()  去掉文件名，返回目录
sys.path.append('/')
os.sys.path.insert(0, parentdir)

class Arm_Robot:
    def __init__ (self, robot_params, urdfRootPath=pybullet_data.getDataPath()):
        self.urdfRootPath = urdfRootPath
        for key, val in robot_params.items():
            exec(f'self.{key} = "{val}"') if type(val) == str else exec(f'self.{key} = {val}' )
        self.reset()

    def reset(self):
        physics_id = p.connect(p.GUI) if self.IS_GUI else p.connect(p.DIRECT)# GUI图形
        p.resetSimulation()  # 重置仿真环境
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = p.loadURDF(self.plane_path)  # 地面
        print(self.robot_URDF_path)
        self.ArmRobotid  = p.loadURDF(self.robot_URDF_path, flags=9)
        tableUid = p.loadURDF(self.table_path, basePosition=self.table_base_pos, globalScaling=1)
        # robot位置
        p.resetBasePositionAndOrientation(self.ArmRobotid, self.robot_base_pos, self.robot_base_orientation)

        self.numJoints = p.getNumJoints(self.ArmRobotid)
        self.jointPositions = [0] * self.numJoints  # 初始化24个关节位置pos信息
        # 重置关节位置
        for jointIndex in range(self.numJoints):
            p.resetJointState(self.ArmRobotid, jointIndex, self.jointPositions[jointIndex])
            self._control_joint(jointIndex, self.jointPositions[jointIndex])

    def getObservation(self):
        '''
        :return: 12个参数(0-2为右臂 xyz，3-5为右臂欧拉角，6-8为左臂 xyz，9-11为左臂欧拉角)
        '''
        observation = []
        # 右臂
        endStateRt = p.getLinkState(self.ArmRobotid, self.ArmRobot_endRt)  # 使用getLinkState查询每个链接的质心的笛卡尔世界位置和方向
        endPosRt = np.array(endStateRt[4])  # 世界坐标系中的位置Position（三维的 x y z）
        endOrnRt = endStateRt[5]  # 世界坐标系中的姿态 Orientation
        endEulerRt = p.getEulerFromQuaternion(endOrnRt)  # 将四元姿态转换为欧拉角  返回一个由3个浮点值组成的vec3列表。旋转顺序是首先绕X滚动，然后绕Y俯仰，最后绕Z偏航
        observation.extend(list(endPosRt))
        observation.extend(list(endEulerRt))

        # 左臂
        endStateLt = p.getLinkState(self.ArmRobotid, self.ArmRobot_endLt)
        endPosLt = np.array(endStateLt[4])
        endOrnLt = endStateLt[5]
        endEulerLt = p.getEulerFromQuaternion(endOrnLt)
        observation.extend(list(endPosLt))
        observation.extend(list(endEulerLt))

        return observation


    def applyAction(self, actions):  # actiions 8 维
        limit_x, limit_y, limit_z = self.limit_x, self.limit_y, self.limit_z
        # clip_val限制动作幅度不会过大
        if (self.useInverseKinematics):  # if IK ,则使用电机控制
            # unpack
            assert len(actions) == 8
            dxRt, dyRt, dzRt, fingerAngleRt, dxLt, dyLt, dzLt, fingerAngleLt = actions
            endStateRt = p.getLinkState(self.ArmRobotid, self.ArmRobot_endRt)
            cur_Rx, cur_Ry, cur_Rz = endStateRt[4] # 末端执行器的世界坐标系中的pos
            endPosRt = [self._clip_val(cur_Rx + dxRt, limit_x),
                        self._clip_val(cur_Ry + dyRt, limit_y),
                        self._clip_val(cur_Rz + dzRt, limit_z)]
            # 通过逆运动学反算出关节位置
            jointPosesRt = p.calculateInverseKinematics(self.ArmRobotid,
                                                        self.ArmRobot_endRt,
                                                        endPosRt,
                                                        lowerLimits=self.joint_ll,
                                                        upperLimits=self.joint_ul,
                                                        )
             # 左臂
            endStateLt = p.getLinkState(self.ArmRobotid, self.ArmRobot_endLt)
            cur_Lx, cur_Ly, cur_Lz = endStateLt[4]  # 末端执行器的世界坐标系中的pos
            endPosLt = [self._clip_val(cur_Lx + dxLt, limit_x),
                        self._clip_val(cur_Ly + dyLt, limit_y),
                        self._clip_val(cur_Lz + dzLt, limit_z)]
            # 通过逆运动学反算出关节位置
            jointPosesLt = p.calculateInverseKinematics(self.ArmRobotid,
                                                        self.ArmRobot_endLt,
                                                        endPosLt,
                                                        lowerLimits=self.joint_ll,
                                                        upperLimits=self.joint_ul,
                                                        )
            for i in self.r_arm_joint:  # 3——10 对应关节值
                self._control_joint(i, jointPosesRt[i - 3])
            self._hand_moving(fingerAngleRt, 0)
            for i in self.l_arm_joint:  # 14——20
                self._control_joint(i, jointPosesLt[i - 5])
            self._hand_moving(-fingerAngleLt, 1)  # 左手需要取负

    def _control_joint(self, jointIndex, target):
        p.setJointMotorControl2(bodyIndex=self.ArmRobotid,
                                jointIndex=jointIndex,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=target,  # 9——15
                                targetVelocity=0,
                                force=self.maxForce,
                                positionGain=0.03,
                                velocityGain=1)


    def _clip_val(self, val, limit):
            if val < limit[0]:
                return limit[0]
            if val > limit[1]:
                return limit[1]
            return val

    def _hand_moving(self, fingerAngle, arm_id):
        '''
        :param fingerAngle: 使手臂末端开口加紧抓取物块
        :param arm_id:  0右臂 1左臂
        '''
        hand_joint1, hand_joint2 = (self.r_hand_joint if arm_id == 0 else self.l_hand_joint)
        hand_joint1_now = p.getJointState(self.ArmRobotid, hand_joint1)[0]
        hand_joint2_now = p.getJointState(self.ArmRobotid, hand_joint2)[0]
        moterCommand_1 = hand_joint1_now + fingerAngle
        moterCommand_2 = hand_joint2_now - fingerAngle

        self._control_joint(hand_joint1, moterCommand_1)
        self._control_joint(hand_joint2, moterCommand_2)



if __name__ == '__main__':
    pybullet_data.getDataPath()

    ArmRobot = Arm_Robot(Args.robot_params)
    p.setGravity(0, 0, -10)  # 设置重力
    count = 0
    while 1:
        random_posRt = [] * 3
        random_posLt = [] * 3
        for _ in range(100):  # random.random()用于生成一个0到1的随机符点数: 0 ≤ n < 1.0
            xpos_targetRt = -0.3 * random.random() + 0.1  # 0.35
            ypos_targetRt = (random.random() * 0.3) - 0.2  # 0.10 0.50
            zpos_targetRt = 0.3 + 0.2 * random.random()
            random_posRt = [xpos_targetRt, ypos_targetRt, zpos_targetRt]  # 随机位置

            # target Position：
            xpos_targetLt = -0.3 * random.random() + 0.1  # 0.35
            ypos_targetLt = (random.random() * 0.3) + 0.2  # 0.10 0.50
            zpos_targetLt = 0.3 + 0.2 * random.random()
            random_posLt = [xpos_targetLt, ypos_targetLt, zpos_targetLt]  # 随机位置

            # 计算两者之间的距离
            dis_between_target_block = math.sqrt(
                (xpos_targetRt - xpos_targetLt) ** 2 + (ypos_targetRt - ypos_targetLt) ** 2 + (zpos_targetRt - zpos_targetLt) ** 2)
            # 保证生成的物块初始位置和目标位置距离不会过小
            if dis_between_target_block >= 0.15:
                break
        cubeRt = p.loadURDF("../URDF_model/cube_small_target_pick.urdf", [random_posRt[0], random_posRt[1], random_posRt[2]], useFixedBase=1)
        cubeLt = p.loadURDF("../URDF_model/cube_small_target_push.urdf", [random_posLt[0], random_posLt[1], random_posLt[2]], useFixedBase=1)
        while count < 25:
            applyActions = [0, 0, 0, 0, 0, 0, 0, 0]  # 8维，第四、八个是末端开口大小fingerangle
            # # 右臂
            # stateRt = p.getLinkState(ArmRobot.ArmRobotid, ArmRobot.ArmRobot_endRt)
            # actualEndEffectorPosRt = stateRt[4]
            # applyActions[0] = -actualEndEffectorPosRt[0] + random_posRt[0]
            # applyActions[1] = -actualEndEffectorPosRt[1] + random_posRt[1]
            # applyActions[2] = -actualEndEffectorPosRt[2] + random_posRt[2]
            # # 左臂
            # stateLt = p.getLinkState(ArmRobot.ArmRobotid, ArmRobot.ArmRobot_endLt)
            # actualEndEffectorPosLt = stateLt[4]
            # applyActions[4] = -actualEndEffectorPosLt[0] + random_posLt[0]
            # applyActions[5] = -actualEndEffectorPosLt[1] + random_posLt[1]
            # applyActions[6] = -actualEndEffectorPosLt[2] + random_posLt[2]

            ArmRobot.applyAction(applyActions)
            for _ in range(20):
              p.stepSimulation()  # 使用正向动力学进行步进模拟
            count += 1
            # time.sleep(0.1)
        count = 0
        p.removeBody(cubeRt)
        p.removeBody(cubeLt)



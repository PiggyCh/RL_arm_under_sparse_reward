import sys
sys.path.append('../')
import gym
import pybullet as p
from Env.ArmRobot import Arm_Robot
import pybullet_data
import random
import  time
from gym import spaces
from arguments import Args
import numpy as np
import math

class ArmRobotGymEnv(gym.Env):
    def __init__(self, task_params, env_pid = 0):
        for key, val in task_params.items():
            exec(f'self.{key} = "{val}"') if type(val) == str else exec(f'self.{key} = {val}' )
        self._urdfRoot = pybullet_data.getDataPath()
        self.env_pid = env_pid
        self.blockUid = -1
        # initialize env and random seed 
        self.Arm_Robot = Arm_Robot(Args.robot_params)
        self.action_space = spaces.Box(-np.array(self.action_high), np.array(self.action_high))
        self.reset()

    def reset(self):
        # 选择约束求解器迭代的最大次数。如果达到了solverResidualThreshold，求解器可能会在numsolver迭代之前终止
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)  # 对于每一个动作，我们执行n个子步骤，即我们需要规定动作频率
        p.setGravity(0, 0, -10)
        for i in range(self.joint_num):
            p.resetJointState(self.Arm_Robot.ArmRobotid, i, 0, 0)
        for _ in range(self.max_gen_time):
            # generate cube Position
            xpos = random.uniform(self.x_gen_range[0], self.x_gen_range[1])
            ypos = random.uniform(self.y_gen_range[0], self.y_gen_range[1])
            zpos = self.z_gen_range[0]
            ang = 3.14 * 0.5 + 3.1415925438 * random.random()
            orn = p.getQuaternionFromEuler([0, 0, ang])
            block_pos = [xpos, ypos, zpos]
            # generate target Position
            xpos_target = random.uniform(self.x_gen_range[0], self.x_gen_range[1])
            ypos_target = random.uniform(self.y_gen_range[0], self.y_gen_range[1])
            zpos_target = random.uniform(self.z_gen_range[0], self.z_gen_range[1]) if self.task_type == 'pick' else self.z_gen_range[0]
            ang_target = 3.14 * 0.5 + 3.1415925438 * random.random()
            orn_target = p.getQuaternionFromEuler([0, 0, ang_target])
            target_pos = [xpos_target, ypos_target, zpos_target]
            # 保证生成的物块初始位置和目标位置距离不会过小
            if self._goal_distance(np.array(block_pos), np.array(target_pos)) >= 0.15:
                break
        if self.blockUid != -1:
            p.removeBody(self.blockUid)
            p.removeBody(self.targetUid)
        self.blockUid = p.loadURDF(self.cube_pick, block_pos, orn)
        self.targetUid = p.loadURDF(self.cube_target, target_pos, orn_target, useFixedBase=1)  # useFixedBase强制加载对象的底座为静态
        p.setCollisionFilterPair(self.targetUid, self.blockUid, -1, -1, 0)  # 碰撞检测
        obs = self._get_obs()
        return obs

    def step(self, action):
        if self.task_type  == 'push':
            action[3] = -1
            action[7] = -1
        self.Arm_Robot.applyAction(action)
        # 一个动作执行20个仿真步
        for _ in range(self.n_substeps):
            p.stepSimulation()
        obs_all = self._get_obs()
        done = self._is_success(self.achieved_goal, self.goal)
        info = {'is_success': done}
        reward = [self.compute_reward(self.achieved_goal, self.goal)]
        return obs_all, reward, done, info

    def _get_obs(self):
        # 关于机械臂的状态观察
        # 末端位置、夹持器状态位置、物体位置、物体姿态、  物体相对末端位置、物体线速度、物体角速度、末端速度、物体相对末端线速度

        # 右臂Rt
        endStateRt = p.getLinkState(self.Arm_Robot.ArmRobotid, self.Arm_Robot.ArmRobot_endRt, computeLinkVelocity=1)  # 参数值为1则也会得到速度
        endPosRt = np.array(endStateRt[4])  # 位置
        endOrnRt = np.array(endStateRt[5])  # 姿态
        endOrnRt = np.array(p.getEulerFromQuaternion(endOrnRt))  # 把四元数转换成欧拉角，使数据都是三维的
        end_linearVelocityRt = np.array(endStateRt[6])  # linear_Velocity 线速度
        end_angularVelocityRt = np.array(endStateRt[7])  # 角速度

        # 左臂Lt
        endStateLt = p.getLinkState(self.Arm_Robot.ArmRobotid, self.Arm_Robot.ArmRobot_endLt, computeLinkVelocity=1)
        endPosLt = np.array(endStateLt[4])
        endOrnLt = np.array(endStateLt[5])
        endOrnLt = np.array(p.getEulerFromQuaternion(endOrnLt))
        end_linearVelocityLt = np.array(endStateLt[6])
        end_angularVelocityLt = np.array(endStateLt[7])

        # 物块block
        blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
        blockPos = np.array(blockPos)
        blockOrn = np.array(p.getEulerFromQuaternion(blockOrn))

        # 目标target
        targetPos, targetOrn = p.getBasePositionAndOrientation(self.targetUid)
        targetPos = np.array(targetPos)
        targetOrn = np.array(p.getEulerFromQuaternion(targetOrn))

        # 臂与物块相对信息 vec *3
        block_relative_posRt = blockPos - endPosRt
        block_relative_ornRt = blockOrn - endOrnRt
        block_relative_posLt = blockPos - endPosLt
        block_relative_ornLt = blockOrn - endOrnLt

        # 右臂目标相对信息
        target_relative_posRt = targetPos - endPosRt
        target_relative_OrnRt = targetOrn - endOrnRt
        target_relative_posLt = targetPos - endPosLt
        target_relative_OrnLt = targetOrn - endOrnLt

        if not self.pick_has_block:
            self.achieved_goal = np.array([endPosRt,endPosLt])
        else:
            self.achieved_goal = blockPos.flatten().copy()

        self.goal = targetPos.flatten().copy()

        push_obsRt = [
            endPosRt.flatten(),  # 右臂末端位置
            endOrnRt.flatten(),  # 右臂末端姿态
            end_linearVelocityRt.flatten(),  # 右臂末端线速度
            end_angularVelocityRt.flatten(),  # 右臂末端角速度
            block_relative_posRt.flatten(),  # 右臂末端与物块的相对位置
            block_relative_ornRt.flatten(),  # 右臂末端与物块的相对姿态
            target_relative_posRt.flatten(),  # 右臂末端与目标的相对位置
            target_relative_OrnRt.flatten()   # 右臂末端与目标的相对姿态
        ]
        push_obsRt = np.array(push_obsRt).reshape(-1)

        push_obsLt=[
            endPosLt.flatten(),
            endOrnLt.flatten(),
            end_linearVelocityLt.flatten(),
            end_angularVelocityLt.flatten(),
            block_relative_posLt.flatten(),
            block_relative_ornLt.flatten(),
            target_relative_posLt.flatten(),
            target_relative_OrnLt.flatten()
        ]

        push_obsLt = np.array(push_obsLt).reshape(-1)
        push_obs = np.array([push_obsRt, push_obsLt])
        return {
            'observation': push_obs,
            'achieved_goal': self.achieved_goal,
            'desired_goal': self.goal
        }

    def compute_reward(self, achieved_goal, goal, sample=False):
        # Compute distance between goal and the achieved goal.
        d = self._goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':  # 稀疏奖励
            return -(d > self.distance_threshold).astype(np.float32)  # 如果达到目标，返回0，没达到目标，返回-1
        else:
            return -d

    def random_action(self):
        action = [0.1*random.random() for _ in range(8)]
        return action
        
    def _goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape 
        # np.linalg.norm指求范数，默认是l2范数  l2-范数表示向量（或矩阵）的元素平方和开根号
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def _is_success(self, achieved_goal, desired_goal):
        d = self._goal_distance(achieved_goal, desired_goal)
        return True if d < self.distance_threshold else False

if __name__ == '__main__':
    pybullet_data.getDataPath()

    args = Args()
    PickRobot = ArmRobotGymEnv(args.task_params)

    obs_all = PickRobot.reset()
    obs = obs_all['observation']
    ag = obs_all['achieved_goal']
    g = obs_all['desired_goal']
    block_pos = ag
    while 1:
        for t in range(100):
            action = [0, 0, 0, 0, 0, 0, 0, 0]
            observation_new, reward, done, info = PickRobot.step(action)
            obs_new = observation_new['observation']
            ag_new = observation_new['achieved_goal']
            obs = obs_new
            ag = ag_new










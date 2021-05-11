import numpy as np
# from gym.envs.robotics import rotations, robot_env, utils
import gym
import os
import math
import pybullet as p
from bmirobot_env.bmirobot import bmirobotv0
from gym.utils import seeding
import pybullet_data
import random
import  time
from gym import spaces
from arguments import get_args, Args
'''
Modified by Hao Cheng
original code from fetch_env
start from 2021 4.5 
'''

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    # np.linalg.norm指求范数，默认是l2范数
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class bmirobotGymEnv(gym.Env):
    """Superclass for all bmirobot environments.
    """
    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): XML文件的路径，这里可以写URDF，在bmirobot里用的是Pybullet环境
            n_substeps (int): 目前推测n-substep是 每次step用的步数。比如一个动作发出后，后续25个时间步骤就继续执行动作
            gripper_extra_height (float): 当定位夹持器时，额外的高度高于桌子
            block_gripper (boolean): 抓手是否被阻塞(即不能移动)
            has_object (boolean):环境中是否有对象
            target_in_the_air (boolean):目标物是否应该在桌子上方的空中或桌面上
            target_offset (float or array with 3 elements): 目标偏移量
            obj_range (float): 初始目标位置采样的均匀分布范围
            target_range (float):采样目标的均匀分布范围
            distance_threshold (float): 目标达到之后的临界值
            initial_qpos (dict):定义初始配置的联合名称和值的字典
            reward_type ('sparse' or 'dense'):奖励类型，如稀疏或密集
        """
        IS_USEGUI = Args().Use_GUI
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.model_path=model_path
        self.n_substeps=n_substeps
        self.n_actions=4
        self.blockUid = -1
        self.initial_qpos=initial_qpos
        self._urdfRoot = pybullet_data.getDataPath()
        self.seed()
        #是否进行渲染，GUI是图形界面，direct是不渲染
        if IS_USEGUI:
            self.physics = p.connect(p.GUI)
        else:
            self.physics = p.connect(p.DIRECT)
        #加载机器人模型
        self._bmirobot = bmirobotv0()
        self._timeStep= 1. / 240.
        action_dim = 4
        self._action_bound = 0.5
        # 这里的action和obs space 的low and high 可能需要再次考虑
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        #重置环境
        self.reset()
        # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def step(self, action):
        action = np.clip(action,-0.5,0.5)
        if p.getClosestPoints(self._bmirobot.bmirobotid, self.blockUid, 0.0001): #如果臂和块足够靠近，可以锁死手爪
            action[3]=-1
        self._set_action(action)
        # print(action[3])
        #一个动作执行20个仿真步
        for _ in range(self.n_substeps):
            p.stepSimulation()
        obs = self._get_obs()
        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def reset(self):
        p.setPhysicsEngineParameter(numSolverIterations=150)
        # 选择约束求解器迭代的最大次数。如果达到了solverResidualThreshold，求解器可能会在numsolver迭代之前终止
        for i in range(24):
            p.resetJointState(self._bmirobot.bmirobotid, i, 0, 0)
        p.setTimeStep(self._timeStep)
        # Cube Pos
        for _ in range(100):
            xpos = 0.15 +0.2 * random.random()  # 0.35
            ypos = (random.random() * 0.3) + 0.2  # 0.10 0.50
            zpos = 0.2
            ang = 3.14 * 0.5 + 3.1415925438 * random.random()
            orn = p.getQuaternionFromEuler([0, 0, ang])
            # target Position：
            xpos_target = 0.35 * random.random()  # 0.35
            ypos_target = (random.random() * 0.25) + 0.3  # 0.10 0.50
            zpos_target = 0.3+ 0.2 * random.random()
            ang_target = 3.14 * 0.5 + 3.1415925438 * random.random()
            orn_target = p.getQuaternionFromEuler([0, 0, ang_target])
            self.dis_between_target_block = math.sqrt(
                (xpos - xpos_target) ** 2 + (ypos - ypos_target) ** 2 + (zpos - zpos_target) ** 2)
            if self.dis_between_target_block >= 0.15:
                break
        if self.blockUid == -1:
            self.blockUid = p.loadURDF("URDF_model/cube_small_pick.urdf", xpos, ypos, zpos,
                                       orn[0], orn[1], orn[2], orn[3])
            self.targetUid = p.loadURDF("URDF_model/cube_small_target_pick.urdf",
                                        [xpos_target, ypos_target, zpos_target],
                                        orn_target, useFixedBase=1)
        else:
            p.removeBody(self.blockUid)
            p.removeBody(self.targetUid)
            self.blockUid = p.loadURDF("URDF_model/cube_small_pick.urdf", xpos, ypos, zpos,
                                       orn[0], orn[1], orn[2], orn[3])
            self.targetUid = p.loadURDF("URDF_model/cube_small_target_pick.urdf",
                                        [xpos_target, ypos_target, zpos_target],
                                        orn_target, useFixedBase=1)
        #print([xpos_target,ypos_target,zpos_target])
        p.setCollisionFilterPair(self.targetUid, self.blockUid, -1, -1, 0)
        self.goal=np.array([xpos_target,ypos_target,zpos_target])
        p.setGravity(0, 0, -10)
        self._envStepCounter = 0
        obs = self._get_obs()
        self._observation = obs
        return self._observation

    def _set_action(self, action):
        self._bmirobot.applyAction(action)

    def _get_obs(self):
        # 关于机械臂的状态观察，可以从以下几个维度进行考虑
        # 末端位置、夹持器状态位置、物体位置、物体姿态、  物体相对末端位置、物体线速度、物体角速度、末端速度、物体相对末端线速度
        # 末端位置 3vec 及速度
        end_pos = np.array(self._bmirobot.getObservation())
        end_pos = end_pos[:3]
        # 夹持器位置、姿姿 vec3 *2  ，可能需要重新考虑一下末端位置和grip的关系
        gripperState = p.getLinkState(self._bmirobot.bmirobotid, self._bmirobot.bmirobot_righthand,
                                      computeLinkVelocity=1)
        gripperPos = np.array(gripperState[4])
        gripperOrn_temp = np.array(gripperState[5])
        gripper_linear_Velocity = np.array(gripperState[6])
        gripper_angular_Velocity = np.array(gripperState[7])
        # 把四元数转换成欧拉角，使数据都是三维的
        gripperOrn = p.getEulerFromQuaternion(gripperOrn_temp)
        gripperOrn = np.array(gripperOrn)
        # 物体位置、姿态
        blockPos, blockOrn_temp = p.getBasePositionAndOrientation(self.blockUid)
        blockPos = np.array(blockPos)
        blockOrn = p.getEulerFromQuaternion(gripperOrn_temp)
        blockOrn = np.array(blockOrn)
        # 物体相对位置 vec *3
        relative_pos = blockPos - gripperPos
        #relative_orn = blockOrn - gripperOrn
        # 物体的线速度和角速度
        block_Velocity = p.getBaseVelocity(self.blockUid)
        block_linear_velocity = np.array(block_Velocity[0])
        target_pos = np.array(p.getBasePositionAndOrientation(self.targetUid)[0])
        block_angular_velocity = np.array(block_Velocity[1])
        # 物体相对末端线速度
        #block_relative_linear_velocity = block_linear_velocity - gripper_linear_Velocity
        # 问题：是否把相对速度、相对位置想得过于理所当然了？ 用不用进行四元数的转换、需不需要考虑位姿下的相对位置，直接写可行吗？
        obs = [
            end_pos.flatten(),
            #gripperPos.flatten(),
            gripperOrn.flatten(),
            gripper_linear_Velocity.flatten(),
            gripper_angular_Velocity.flatten(),
            blockPos.flatten(),
            blockOrn.flatten(),
            relative_pos.flatten(),
            #relative_orn.flatten(),
            #target_pos.flatten(),
            #target_relative_pos.flatten()
            block_linear_velocity.flatten(),
            block_angular_velocity.flatten(),
            # block_relative_linear_velocity.flatten()
        ]
        #print(blockPos)
        if not self.has_object:
            achieved_goal = end_pos.copy()
        else:
            achieved_goal = blockPos.copy()
        for i in range(1, len(obs)):
            end_pos = np.append(end_pos, obs[i])
        obs = end_pos.reshape(-1)
        self._observation = obs
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': target_pos.flatten(),
        }

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

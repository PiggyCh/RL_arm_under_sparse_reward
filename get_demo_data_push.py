#! /usr/bin/env python
import numpy as np
import math
from arguments import get_args, Args
from mpi4py import MPI
#mpi4py是分布式计算的库   MPI:message passing interface 即消息传递接口
import random
from bmirobot_env.bmirobot_push_F import bmirobotGympushEnv as bmenv
"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)
"""
#设置获取demo的个数
demo_num=1000
def get_env_params(env):
    obs = env.reset()
    params = {'obs': obs['observation'].shape[0],
              'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0],
              'action_max': env.action_space.high[0],
              }
    params['max_timesteps'] = 100
    return params

def get_push_demo(env,env_params):
    savetime = 0
    obs_total, ag_total, g_total, actions_total, info_total = [], [], [], [], []
    for epoch in range(10000):
        if savetime >= demo_num:
            break
        mb_obs, mb_ag, mb_g, mb_actions, mb_info = [], [], [], [], []
        # reset the rollouts
        ep_obs, ep_ag, ep_g, ep_actions, ep_info = [], [], [], [], []
        # reset the environment
        observation = env.reset()
        obs = observation['observation']
        ag = observation['achieved_goal']
        g = observation['desired_goal']
        step_time = 0
        for t in range(int(env_params['max_timesteps'])):
            step_time += 1
            grip_pos = obs[:3]
            blocK_pos = obs[12:15]
            if step_time <= 10:
                action = [0, -0.1, 0.1, 0]
            elif step_time <= 20:
                action = [(g[0] - blocK_pos[0]) * (-0.5) + blocK_pos[0] - grip_pos[0],
                          (g[1] - blocK_pos[1]) * (-0.5) + blocK_pos[1] - grip_pos[1],
                          blocK_pos[2] + (g[2] - blocK_pos[2]) * (-0.5) - grip_pos[2], 0]
            elif step_time <= 40:
                action = [g[0] - blocK_pos[0], g[1] - blocK_pos[1], g[2] - blocK_pos[2], 0]
            elif step_time <= 60:
                action = [0.241 - grip_pos[0], 0.3265 - grip_pos[1], 0.294 - grip_pos[2], 0]
            elif step_time <= 80:
                action = [(g[0] - blocK_pos[0]) * (-0.5) + blocK_pos[0] - grip_pos[0],
                          (g[1] - blocK_pos[1]) * (-0.5) + blocK_pos[1] - grip_pos[1],
                          blocK_pos[2] + (g[2] - blocK_pos[2]) * (-0.5) - grip_pos[2], 0]
            else:
                action = [g[0] - blocK_pos[0], g[1] - blocK_pos[1], g[2] - blocK_pos[2], 0]
            if math.sqrt((blocK_pos[0] - g[0]) ** 2 + (blocK_pos[1] - g[1]) ** 2 + (
                    blocK_pos[2] - g[2]) ** 2) < 0.05:
                action = [0, 0, 0, 0]
            action = list(action)
            observation_new, _, _, info = env.step(action)
            obs_new = observation_new['observation']
            ag_new = observation_new['achieved_goal']
            # append rollouts
            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())
            ep_g.append(g.copy())
            ep_actions.append(action.copy())
            ep_info.append(info.copy())
            # re-assign the observation
            obs = obs_new
            ag = ag_new
        if info['is_success'] == 1.0:
            savetime += 1
            print("This is " +str(savetime) +" savetime " )
            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())
            mb_obs.append(ep_obs)
            mb_ag.append(ep_ag)
            mb_g.append(ep_g)
            mb_actions.append(ep_actions)
            mb_info.append(ep_info)
            # convert them into arrays
            obs_total.append(mb_obs)
            actions_total.append(mb_actions)
            g_total.append(mb_g)
            ag_total.append(mb_ag)
            info_total.append(mb_info)
    file = "bmirobot_"+str(savetime)+"_push_demo.npz"
    np.savez_compressed(file, acs=np.array(actions_total).squeeze(), obs=np.array(obs_total).squeeze(),
                        info=np.array(info_total).squeeze(), g=np.array(g_total).squeeze(),
                        ag=np.array(ag_total).squeeze())
def launch(args):
    # create the ddpg_agent
    #创建环境，从参数文件里找
    #env = gym.make(args.env_name)
    env = bmenv()
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    env_params = get_env_params(env)
    # create the ddpg agent to interact with the environment
    get_push_demo(env,env_params)

if __name__ == '__main__':
    # get the params
    args = Args()
    launch(args)

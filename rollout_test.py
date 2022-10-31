#! /usr/bin/env python
import os
import torch as th
from Algorithm.model import actor
from Env.ArmRobot_gym import ArmRobotGymEnv as bmenv
from Algorithm.util import select_action, process_inputs
from arguments import Args
import sys
sys.path.append('Env')
sys.path.append('Algorithm')

#加载训练好的模型 数据
model_path = "saved_models_gail/armrobot_push_ seed123_919/123_True29_model.pt"

def rollout_test(args):
    o_mean, o_std, g_mean, g_std, act1, act2, cr1, cr2, disc = th.load(model_path, map_location=lambda storage, loc: storage)
    # create the environment+
    env_params = args.env_params
    # get agents
    actors_network = [actor(env_params) for _ in range(env_params['n_agents'])]
    actors_network[0].load_state_dict(act1)
    actors_network[1].load_state_dict(act2)
    actors_network[0].eval()
    actors_network[1].eval()
    env = bmenv(args.task_params)
    is_successes = 0
    for i in range(args.demo_length):
        observation = env.reset()
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        for t in range(env_params['max_timesteps']):
            obs_norm, g_norm = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            action = select_action(actors_network, obs_norm, g_norm, env_params)
            # put actions into the environment
            observation_new, reward, _, info = env.step(action)
            obs = observation_new['observation']
            g = observation_new['desired_goal']
            if info['is_success'] == 1:
                is_successes += 1
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))
    print('demo test success rate:', is_successes/args.demo_length)

if __name__ == '__main__':
    args = Args()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    rollout_test(args)


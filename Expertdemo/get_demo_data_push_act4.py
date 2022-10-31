#! /usr/bin/env python
import numpy as np
from arguments import Args

import math
from Env.ArmRobot_gym import ArmRobotGymEnv as pickenv
"""

"""

demo_num = 4000
def get_env_params():
    params ={}
    params['n_agents'] = 2
    params['dim_observation'] = 24  # 每个agent的obs维度
    params['dim_action'] = 3
    params['dim_hand'] = 3
    params['dim_achieved_goal'] = 3
    params['dim_goal'] = 3
    params['max_timesteps'] = 100
    params['action_max'] = 0.5
    return params

def launch(args):
    # create the ddpg_agent 创建环境，从参数文件里找

    env = pickenv(args.task_params)
    env_params = get_env_params()
    # create the ddpg agent to interact with the environment
    get_pick_demo(env, env_params)


def get_pick_demo(env, env_params):
    savetime = 0
    obs_total, ag_total, g_total, actions_total, acts_total, hands_total, info_total, next_obs_total, next_ag_total, r_total = [], [], [], [], [], [], [], [], [], []
    for epoch in range(100000):
        if savetime >= demo_num:
            break
        mb_obs, mb_ag, mb_g, mb_actions, mb_acts, mb_hands, mb_info, mb_next_obs, mb_next_ag, mb_r = [], [], [], [], [], [], [], [], [], []
        # reset the rollouts
        ep_obs, ep_ag, ep_g, ep_actions, ep_acts, ep_hands, ep_info, ep_next_obs, ep_next_ag, ep_r = [], [], [], [], [], [], [], [], [], []
        # reset the environment
        obs_all = env.reset()
        obs = obs_all['observation']
        ag = obs_all['achieved_goal']
        g = obs_all['desired_goal']
        block_pos = ag
        step_time = 0

        if block_pos[0] > 0.025 and g[0] > 0.025:
            # 物块和目标都在右侧位置，只有右臂pick
            for t in range(int(env_params['max_timesteps'])):
                step_time += 1
                end_posRt = obs[0, 0:3]
                block_pos = ag
                # action是8维  action_store是11维
                if step_time <= 5:
                    action = [0, -0.1, 0.1, -1, 0, 0, 0, -1]
                    acts = [0, -0.1, 0.1, 0, 0, 0]
                    hands = [1, 0, 0, 1, 0, 0]
                elif step_time <= 15:
                    action = [block_pos[0] - end_posRt[0], block_pos[1] - end_posRt[1] - 0.1,
                              block_pos[2] - end_posRt[2] + 0.2, -1, 0, 0, 0, -1]
                    acts = [block_pos[0] - end_posRt[0], block_pos[1] - end_posRt[1] - 0.1,
                            block_pos[2] - end_posRt[2] + 0.2, 0, 0, 0]
                    hands = [1, 0, 0, 1, 0, 0]

                elif step_time <= 35:
                    action = [block_pos[0] - end_posRt[0], block_pos[1] - end_posRt[1] - 0.05,
                              block_pos[2] - end_posRt[2] - 0.1, -1, 0, 0, 0, -1]
                    acts = [block_pos[0] - end_posRt[0], block_pos[1] - end_posRt[1] - 0.05,
                            block_pos[2] - end_posRt[2] - 0.1, 0, 0, 0]
                    hands = [1, 0, 0, 1, 0, 0]

                else:
                    action = [g[0] - block_pos[0], g[1] - block_pos[1], g[2] - block_pos[2], -1, 0, 0, 0, -1]
                    acts = [g[0] - block_pos[0], g[1] - block_pos[1], g[2] - block_pos[2], 0, 0, 0]
                    hands = [1, 0, 0, 1, 0, 0]

                if math.sqrt((block_pos[0] - g[0]) ** 2 + (block_pos[1] - g[1]) ** 2 + (
                        block_pos[2] - g[2]) ** 2) < 0.05:
                    action = [0, 0, 0, -1, 0, 0, 0, -1]
                    acts = [0, 0, 0, 0, 0, 0]
                    hands = [1, 0, 0, 1, 0, 0]

                action = list(action)
                acts = list(acts)
                hands = list(hands)
                observation_new, reward, done, info = env.step(action)
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                # append rollouts
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                ep_g.append(g.copy())
                ep_actions.append(action.copy())
                ep_acts.append(acts.copy())
                ep_hands.append(hands.copy())
                ep_next_obs.append(obs_new.copy())
                ep_next_ag.append(ag_new.copy())
                ep_r.append(reward.copy())
                ep_info.append(info.copy())

                # re-assign the observation
                obs = obs_new
                ag = ag_new

        elif block_pos[0] < 0.025 and g[0] < 0.025:
            # 物块和目标都在左侧位置，只有左臂pick
            for t in range(int(env_params['max_timesteps'])):
                step_time += 1
                end_posLt = obs[1, 0:3]
                block_pos = ag
                if step_time <= 5:
                    action = [0, 0, 0, -1, 0, -0.1, 0.1, -1]
                    acts = [0, 0, 0, 0, -0.1, 0.1]
                    hands = [1, 0, 0, 1, 0, 0]

                elif step_time <= 15:
                    action = [0, 0, 0, -1, block_pos[0] - end_posLt[0], block_pos[1] - end_posLt[1] - 0.1,
                              block_pos[2] - end_posLt[2] + 0.2, -1]
                    acts = [0, 0, 0, block_pos[0] - end_posLt[0], block_pos[1] - end_posLt[1] - 0.1,
                            block_pos[2] - end_posLt[2] + 0.2]
                    hands = [1, 0, 0, 1, 0, 0]

                elif step_time <= 35:
                    action = [0, 0, 0, -1, block_pos[0] - end_posLt[0], block_pos[1] - end_posLt[1] - 0.05,
                              block_pos[2] - end_posLt[2] - 0.1, -1]
                    acts = [0, 0, 0, block_pos[0] - end_posLt[0], block_pos[1] - end_posLt[1] - 0.05,
                            block_pos[2] - end_posLt[2] - 0.1]
                    hands = [1, 0, 0, 1, 0, 0]

                else:
                    action = [0, 0, 0, -1, g[0] - block_pos[0], g[1] - block_pos[1], g[2] - block_pos[2], -1]
                    acts = [0, 0, 0, g[0] - block_pos[0], g[1] - block_pos[1], g[2] - block_pos[2]]
                    hands = [1, 0, 0, 1, 0, 0]

                if math.sqrt((block_pos[0] - g[0]) ** 2 + (block_pos[1] - g[1]) ** 2 + (
                        block_pos[2] - g[2]) ** 2) < 0.05:
                    action = [0, 0, 0, -1, 0, 0, 0, -1]
                    acts = [0, 0, 0, 0, 0, 0]
                    hands = [1, 0, 0, 1, 0, 0]

                action = list(action)
                acts = list(acts)
                hands = list(hands)
                observation_new, reward, done, info = env.step(action)
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                # append rollouts
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                ep_g.append(g.copy())
                ep_actions.append(action.copy())
                ep_acts.append(acts.copy())
                ep_hands.append(hands.copy())
                ep_next_obs.append(obs_new.copy())
                ep_next_ag.append(ag_new.copy())
                ep_r.append(reward.copy())
                ep_info.append(info.copy())

                # re-assign the observation
                obs = obs_new
                ag = ag_new

        elif block_pos[0] < 0.025 and g[0] > 0.025:
            # 物块在左侧,目标在右侧，左臂先抓到中间，右臂再抓
            for t in range(int(env_params['max_timesteps'])):
                step_time += 1
                end_posRt = obs[0, 0:3]
                end_posLt = obs[1, 0:3]
                block_pos = ag

                if step_time <= 5:
                    action = [0, 0, 0, -1, 0, -0.1, 0.1, -1]
                    acts = [0, 0, 0, 0, -0.1, 0.1]
                    hands = [1, 0, 0, 1, 0, 0]

                elif step_time <= 15:
                    action = [0, 0, 0, -1, block_pos[0] - end_posLt[0], block_pos[1] - end_posLt[1] - 0.1,
                              block_pos[2] - end_posLt[2] + 0.2, 0]
                    acts = [0, 0, 0, block_pos[0] - end_posLt[0], block_pos[1] - end_posLt[1] - 0.1,
                                    block_pos[2] - end_posLt[2] + 0.2]
                    hands = [1, 0, 0, 1, 0, 0]

                elif step_time <= 35:
                    action = [0, 0, 0, -1, block_pos[0] - end_posLt[0], block_pos[1] - end_posLt[1] - 0.05,
                              block_pos[2] - end_posLt[2] - 0.1, 0]
                    acts = [0, 0, 0, block_pos[0] - end_posLt[0], block_pos[1] - end_posLt[1] - 0.05,
                            block_pos[2] - end_posLt[2] - 0.1]
                    hands = [1, 0, 0, 1, 0, 0]

                elif step_time <= 50:
                    action = [0, 0, 0, -1, g[0] - block_pos[0], g[1] - block_pos[1], g[2] - block_pos[2], -1]
                    acts = [0, 0, 0, g[0] - block_pos[0], g[1] - block_pos[1], g[2] - block_pos[2]]
                    hands = [1, 0, 0, 1, 0, 0]

                elif step_time <= 55:
                    action = [0, -0.1, 0.1, 0, 0, 0, 0.5, -1]
                    acts = [0, -0.1, 0.1, 0, 0, 0.5]
                    hands = [1, 0, 0, 1, 0, 0]

                elif step_time <= 65:
                    action = [block_pos[0] - end_posRt[0] + 0.1, block_pos[1] - end_posRt[1] - 0.1,
                              block_pos[2] - end_posRt[2] + 0.2, -1,
                             -0.191 - end_posLt[0], 0.3265 - end_posLt[1], 0.394 - end_posLt[2], -1]
                    acts = [block_pos[0] - end_posRt[0] + 0.1, block_pos[1] - end_posRt[1] - 0.1,
                                    block_pos[2] - end_posRt[2] + 0.2,
                            -0.191 - end_posLt[0], 0.3265 - end_posLt[1], 0.394 - end_posLt[2]]
                    hands = [1, 0, 0, 1, 0, 0]

                elif step_time <= 85:
                    action = [block_pos[0] - end_posRt[0], block_pos[1] - end_posRt[1] - 0.05,
                              block_pos[2] - end_posRt[2] - 0.1, -1, 0, 0, 0, -1]
                    acts = [block_pos[0] - end_posRt[0], block_pos[1] - end_posRt[1] - 0.05,
                            block_pos[2] - end_posRt[2] - 0.1, 0, 0, 0]
                    hands = [1, 0, 0, 1, 0, 0]

                else:
                    action = [g[0] - block_pos[0], g[1] - block_pos[1], g[2] - block_pos[2], -1, 0, 0, 0, -1]
                    acts = [g[0] - block_pos[0], g[1] - block_pos[1], g[2] - block_pos[2], 0, 0, 0]
                    hands = [1, 0, 0, 1, 0, 0]

                if math.sqrt((block_pos[0] - g[0]) ** 2 + (block_pos[1] - g[1]) ** 2 + (
                        block_pos[2] - g[2]) ** 2) < 0.05:
                    action = [0, 0, 0, -1, 0, 0, 0, -1]
                    acts = [0, 0, 0, 0, 0, 0]
                    hands = [1, 0, 0, 1, 0, 0]

                action = list(action)
                acts = list(acts)
                hands = list(hands)
                observation_new, reward, done, info = env.step(action)
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                # append rollouts
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                ep_g.append(g.copy())
                ep_actions.append(action.copy())
                ep_acts.append(acts.copy())
                ep_hands.append(hands.copy())
                ep_next_obs.append(obs_new.copy())
                ep_next_ag.append(ag_new.copy())
                ep_r.append(reward.copy())
                ep_info.append(info.copy())

                # re-assign the observation
                obs = obs_new
                ag = ag_new

        elif block_pos[0] > 0.025 and g[0] < 0.025:
            # 物块在右侧,目标在左侧，右臂先抓到中间，左臂再抓
            for t in range(int(env_params['max_timesteps'])):
                step_time += 1
                end_posRt = obs[0, 0:3]
                end_posLt = obs[1, 0:3]
                block_pos = ag

                if step_time <= 5:
                    action = [0,  -0.1, 0.1, -1, 0, 0, 0, -1]
                    acts = [0, -0.1, 0.1, 0, 0, 0]
                    hands = [1, 0, 0, 1, 0, 0]

                elif step_time <= 15:
                    action = [block_pos[0] - end_posRt[0] + 0.1, block_pos[1] - end_posRt[1] - 0.1,
                              block_pos[2] - end_posRt[2] + 0.2, -1,
                              0, 0, 0, -1]
                    acts = [block_pos[0] - end_posRt[0] + 0.1, block_pos[1] - end_posRt[1] - 0.1,
                            block_pos[2] - end_posRt[2] + 0.2, 0, 0, 0]
                    hands = [1, 0, 0, 1, 0, 0]

                elif step_time <= 35:
                    action = [block_pos[0] - end_posRt[0], block_pos[1] - end_posRt[1] - 0.05,
                              block_pos[2] - end_posRt[2], -1, 0, 0, 0, -1]
                    acts = [block_pos[0] - end_posRt[0], block_pos[1] - end_posRt[1] - 0.05,
                                    block_pos[2] - end_posRt[2], 0, 0, 0,]
                    hands = [1, 0, 0, 1, 0, 0]

                elif step_time <= 50:
                    action = [g[0] - block_pos[0], g[1] - block_pos[1], g[2] - block_pos[2], -1, 0, 0, 0, -1]
                    acts = [g[0] - block_pos[0], g[1] - block_pos[1], g[2] - block_pos[2], 0, 0, 0]
                    hands = [1, 0, 0, 1, 0, 0]

                elif step_time <= 55:
                    action = [0, 0, 0.5, -1, 0, -0.1, 0.1, 0]
                    acts = [0, 0, 0.5, 0, -0.1, 0.1]
                    hands = [1, 0, 0, 1, 0, 0]

                elif step_time <= 65:
                    action = [0.191 - end_posRt[0], 0.3265 - end_posRt[1], 0.394 - end_posRt[2], -1,
                              block_pos[0] - end_posLt[0], block_pos[1] - end_posLt[1] - 0.1,
                              block_pos[2] - end_posLt[2] + 0.2, -1]
                    acts = [0.191 - end_posRt[0], 0.3265 - end_posRt[1], 0.394 - end_posRt[2],
                            block_pos[0] - end_posLt[0], block_pos[1] - end_posLt[1] - 0.1,
                            block_pos[2] - end_posLt[2] + 0.2]
                    hands = [1, 0, 0, 1, 0, 0]

                elif step_time <= 85:
                    action = [0, 0, 0, -1, block_pos[0] - end_posLt[0], block_pos[1] - end_posLt[1] - 0.05,
                              block_pos[2] - end_posLt[2] - 0.1, -1]
                    acts = [0, 0, 0, block_pos[0] - end_posLt[0], block_pos[1] - end_posLt[1] - 0.05,
                                    block_pos[2] - end_posLt[2] - 0.1]
                    hands = [1, 0, 0, 1, 0, 0]

                else:
                    action = [0, 0, 0, -1, g[0] - block_pos[0], g[1] - block_pos[1], g[2] - block_pos[2], -1]
                    acts = [0, 0, 0, g[0] - block_pos[0], g[1] - block_pos[1], g[2] - block_pos[2]]
                    hands = [1, 0, 0, 1, 0, 0]

                if math.sqrt((block_pos[0] - g[0]) ** 2 + (block_pos[1] - g[1]) ** 2 + (
                        block_pos[2] - g[2]) ** 2) < 0.01:
                    action = [0, 0, 0, -1, 0, 0, 0, -1]
                    acts = [0, 0, 0, 0, 0, 0]
                    hands = [1, 0, 0, 1, 0, 0]

                action = list(action)
                acts = list(acts)
                hands = list(hands)
                observation_new, reward, done, info = env.step(action)
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                # append rollouts
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                ep_g.append(g.copy())
                ep_actions.append(action.copy())
                ep_acts.append(acts.copy())
                ep_hands.append(hands.copy())
                ep_next_obs.append(obs_new.copy())
                ep_next_ag.append(ag_new.copy())
                ep_r.append(reward.copy())
                ep_info.append(info.copy())

                # re-assign the observation
                obs = obs_new
                ag = ag_new

        if info['is_success'] == 1.0:
            savetime += 1
            print("This is " +str(savetime) +" savetime " )

            mb_obs.append(ep_obs)
            mb_ag.append(ep_ag)
            mb_g.append(ep_g)
            mb_actions.append(ep_actions)
            mb_acts.append(ep_acts)
            mb_hands.append(ep_hands)
            mb_next_obs.append(ep_next_obs)
            mb_next_ag.append(ep_next_ag)
            mb_r.append(ep_r)
            mb_info.append(ep_info)

            # convert them into arrays
            obs_total.append(mb_obs)
            actions_total.append(mb_actions)
            acts_total.append(mb_acts)
            hands_total.append(mb_hands)
            g_total.append(mb_g)
            ag_total.append(mb_ag)
            next_obs_total.append(mb_next_obs)
            next_ag_total.append(mb_next_ag)
            r_total.append(mb_r)
            info_total.append(mb_info)

    file = "armrobot_"+str(savetime)+"_push_demo.npz"
    actions = np.array(actions_total).squeeze()
    acts = np.array(acts_total).squeeze()
    hands = np.array(hands_total).squeeze()
    obs = np.array(obs_total).squeeze()
    info = np.array(info_total).squeeze()
    g = np.array(g_total).squeeze()
    ag = np.array(ag_total).squeeze()
    next_obs = np.array(next_obs_total).squeeze()
    next_ag = np.array(next_ag_total).squeeze()
    reward = np.array(r_total).squeeze()
    reward = np.expand_dims(reward, axis=2)
    np.savez_compressed(file, actions=actions, acts=acts, hands=hands, obs=obs, info=info, g=g,
                        ag=ag, next_obs=next_obs,
                        next_ag=next_ag, reward=reward)


if __name__ == '__main__':
    # get the params
    args = Args()
    launch(args)

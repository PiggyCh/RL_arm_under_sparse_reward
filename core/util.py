import torch as th
import numpy as np
import sys
from arguments import Args
import time
import random
sys.path.append('Env')
sys.path.append('Algorithm')
from pathlib import Path
import torch.multiprocessing as mp
import sys
sys.path.append(Path(__file__).parent.parent.resolve().as_posix())
# process the inputs

def process_inputs(o, g, o_mean, o_std, g_mean, g_std):
    clip_obs = Args.clip_obs
    clip_range = Args.clip_range
    size = 48
    o = o.reshape(-1, size)
    o_clip = np.clip(o, -clip_obs, clip_obs)
    g_clip = np.clip(g, -clip_obs, clip_obs)
    o_norm_ = np.clip((o_clip - o_mean) / (o_std), -clip_range, clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -clip_range, clip_range)
    o_norm = o_norm_.reshape(-1, 24)
    return o_norm, g_norm


def select_action(actors_network, obs_norm, g_norm, env_params):
    n_agents = env_params['n_agents']
    dim_action = env_params['dim_action']
    actions = np.ones([n_agents, dim_action + 1])  # 2*4
    hand_states = [-1, 0, 0.5]
    for i in range(n_agents):
        sb_norm = obs_norm[i, :]
        inputs = np.concatenate([sb_norm, g_norm])
        inputs_tensor = th.tensor(inputs, dtype=th.float32)
        act, hand_logits = actors_network[i](inputs_tensor)  # actor网络得到初步的act
        act = act.detach().numpy().squeeze()  # 转化为数组形式 .aqueeze()去除一维
        act = np.clip(act, -env_params['action_max'], env_params['action_max'])
        hand_id = np.argmax(hand_logits.detach().cpu().numpy(), axis=-1)
        hand_act = [hand_states[hand_id]]  # 获取抽到末端状态位置对应的值
        hand_act_numpy = np.array(hand_act)  # 转为numpy，方便与act结合
        actions_ = np.concatenate((act, hand_act_numpy), axis=-1)
        actions[i, :] = actions_
    actions = actions.reshape(-1)
    return actions

def num_to_tensor(inputs, device = 'cpu'):
    # inputs_tensor = th.tensor(inputs, dtype=th.float32).unsqueeze(0)  # 会在第0维增加一个维度
    inputs_tensor = th.tensor(inputs, dtype=th.float32).to(device)
    return inputs_tensor

def compute_reward(achieved_goal, goal, sample=False):
    # Compute distance between goal and the achieved goal.
    d = np.linalg.norm(achieved_goal - goal, axis=-1)
    if Args.task_params.reward_type == 'sparse':  # 稀疏奖励
        return -(d > Args.task_params.distance_threshold).astype(np.float32)  # 如果达到目标，返回0，没达到目标，返回-1
    else:
        return -d
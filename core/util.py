import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.resolve().as_posix())

import numpy as np
import sys
import torch
from torch.distributions.categorical import Categorical

from arguments import Args
# process the inputs

env_params = Args.env_params
train_params = Args.train_params
task_params = Args.task_params

dim_observation = env_params.dim_observation
n_agents = env_params.n_agents
dim_action = env_params.dim_action
dim_hand = env_params.dim_hand
noise_eps = train_params.noise_eps
action_max = env_params.action_max
random_eps = train_params.random_eps
hand_states =  train_params.hand_states
clip_obs = Args.clip_obs
clip_range = Args.clip_range

def process_inputs(o, g, norm):
    o = o.reshape(-1, o.shape[-1] * o.shape[-2])
    o_clip = np.clip(o, -clip_obs, clip_obs)
    g_clip = np.clip(g, -clip_obs, clip_obs)
    o_norm_ = np.clip((o_clip - norm['o_mean']) / (norm['o_std']), -clip_range, clip_range)
    g_norm = np.clip((g_clip - norm['g_mean']) / (norm['g_std']), -clip_range, clip_range)
    o_norm = o_norm_.reshape(-1, dim_observation)
    return o_norm, g_norm

@torch.no_grad()
def select_action(actors, obs, g, normalizer, explore):
    acts = np.ones([n_agents, dim_action])
    hands = np.ones([n_agents, dim_hand])
    actions = np.ones([n_agents, dim_action + 1])
    obs_norm, g_norm = process_inputs(obs, g, normalizer)
    for i in range(n_agents):
        sb_norm = obs_norm[i, :]
        inputs = np.concatenate([sb_norm, g_norm])
        act, hand_logits = actors[i](num_to_tensor(inputs)) 
        act = act.cpu().numpy().squeeze()
        # add the gaussian
        if explore:
            act += noise_eps * action_max * np.random.randn(*act.shape)
            act = np.clip(act, -action_max, action_max)
            # random actions...
            random_actions = np.random.uniform(low=-action_max, high=action_max,
                                            size=dim_action)
            # choose if use the random actions
            act += np.random.binomial(1, random_eps, 1)[0] * (random_actions - act)
            hand_id = Categorical(hand_logits).sample().detach().cpu().numpy() 
        else:
            hand_id = np.argmax(hand_logits.detach().cpu().numpy(), axis=-1)
        acts[i, :] = act
        hands[i, :] = hand_logits
        # cat actions
        hand_act = [hand_states[hand_id]] 
        actions_ = np.concatenate((act, np.array(hand_act)), axis = -1)
        actions[i, :] = actions_
    return actions.reshape(-1), acts.reshape(-1), hands.reshape(-1)

def num_to_tensor(inputs, device = 'cpu'):
    # inputs_tensor = th.tensor(inputs, dtype=th.float32).unsqueeze(0)  # 会在第0维增加一个维度
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)
    return inputs_tensor

def compute_reward(achieved_goal, goal, sample=False):
    # Compute distance between goal and the achieved goal.
    d = np.linalg.norm(achieved_goal - goal, axis=-1)
    if Args.task_params.reward_type == 'sparse':  # 稀疏奖励
        return -(d > Args.task_params.distance_threshold).astype(np.float32)  # 如果达到目标，返回0，没达到目标，返回-1
    else:
        return -d
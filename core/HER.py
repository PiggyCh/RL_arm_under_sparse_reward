import numpy as np
import torch
from core.util import compute_reward

class her_sampler:
    def __init__(self, replay_strategy, replay_k):
        self.replay_strategy = replay_strategy 
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + self.replay_k))
        else:
            self.future_p = 0
        self.reward_func = compute_reward # recompute_reward after replacing goal

    def sample_her_transitions(self, buffer, batch_size):
        T =  buffer['acts'][0].shape[0]
        episode_size = len(buffer['acts'])
        batch_size = batch_size
        episode_idxs = np.random.randint(0, episode_size, batch_size)# sample episode 
        t_samples = np.random.randint(T, size=batch_size)  # sample t_step from the episodes
        # gather training data
        transitions = {}
        for key in buffer.keys():
            transitions[key] = torch.stack([buffer[key][idx][t_samples[t_idx]] for t_idx,idx in enumerate(episode_idxs)])
        # sample new goal from achieved goals
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + future_offset)[her_indexes]
        # replace goal with achieved goal
        for her_idx, idx, t_idx in zip(her_indexes[0], episode_idxs[her_indexes], future_t):
            transitions['g'][her_idx] = buffer['ag'][episode_idxs[her_idx]][t_idx]
        # re-compute reward
        transitions['reward'] = np.expand_dims(self.reward_func(transitions['next_ag'].detach().cpu(), transitions['g'].detach().cpu(), sample = True), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}
        return transitions
    


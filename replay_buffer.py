import threading
import numpy as np

"""
the replay buffer here is basically from the openai baselines code

"""


class replay_buffer:
    def __init__(self, env_params, buffer_size, sample_func):
        self.env_params = env_params
        #这里的T的目的是什么：可能是为了下面的buffer存储
        self.T = env_params['max_timesteps']
        #size是总共能存episode的大小
        self.size = int(buffer_size // self.T)
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        print("Buffer_size:", self.size, "max_timesteps:", self.T, "env_params:", self.env_params)
        self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                        'g': np.empty([self.size, self.T, self.env_params['goal']]),
                        'actions': np.empty([self.size, self.T, self.env_params['action']]),
                        }
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        # print(mb_obs.shape, mb_ag.shape, mb_g.shape, mb_actions.shape)
        batch_size = mb_obs.shape[0] #batch_size 是有多少组
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['obs'][idxs] = mb_obs
            self.buffers['ag'][idxs] = mb_ag
            self.buffers['g'][idxs] = mb_g
            self.buffers['actions'][idxs] = mb_actions
            self.n_transitions_stored += self.T * batch_size

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1 #if  inc =0, inc=inc or 1  result:inc =1
        if self.current_size+inc <= self.size:  #如果buffer没有满，就直接存入，满了就替换
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx

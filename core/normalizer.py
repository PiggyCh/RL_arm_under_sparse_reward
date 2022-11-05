from copy import deepcopy
import numpy as np
import torch
import torch.multiprocessing as mp

class normalizer:
    def __init__(self, n_agent, size, device, eps=1e-2, default_clip_range=np.inf):
        self.n_agent = n_agent
        self.size = size
        self.eps = eps
        self.device = device
        self.default_clip_range = default_clip_range
        self.Manager = mp.Manager()
        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)
        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.zeros(1, np.float32)
        self.total = self.Manager.dict({
                'mean' : np.zeros(self.size, np.float32),
                'std': np.ones(self.size, np.float32)
                })
        self.torch_mean = torch.zeros(self.size, dtype = torch.float32).to(self.device)
        self.torch_std = torch.ones(self.size, dtype = torch.float32).to(self.device)
        self.np_mean = np.zeros(self.size, dtype = np.float32)
        self.np_std = np.ones(self.size, dtype = np.float32)

    # update the parameters of the normalizer
    def update(self, v):
        v = v.reshape(-1, self.size)
        # do the computing
        self.local_sum += v.sum(axis=0) 
        self.local_sumsq += (np.square(v)).sum(axis=0) 
        self.local_count[0] += v.shape[0]

    def recompute_stats(self):
        # with self.lock:
        local_count = self.local_count.copy()
        local_sum = self.local_sum.copy()
        local_sumsq = self.local_sumsq.copy()
        # reset
        self.local_count[...] = 0
        self.local_sum[...] = 0
        self.local_sumsq[...] = 0
        # sync the stats  同步统计数据
        # recompute new value
        self.total_sum += local_sum
        self.total_sumsq += local_sumsq
        self.total_count += local_count
        self.np_mean = self.total_sum / self.total_count
        self.np_std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(
            self.total_sum / self.total_count)))
        self.torch_mean = torch.tensor(self.np_mean).to(self.device)
        self.torch_std = torch.tensor(self.np_std).to(self.device)
        # return data to manager
        self.total['mean'] = deepcopy(self.np_mean)
        self.total['std'] = deepcopy(self.np_std)

    # normalize the observation
    def normalize_obs(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range  # 5
        v = v.reshape(-1, self.size)  # 展开成所需维度
        if type(v) == type(torch.zeros([0])):
            norm = torch.clamp((v - self.torch_mean) / self.torch_std, -clip_range, clip_range)
        else:
            norm = np.clip((v - self.np_mean) / self.np_std, -clip_range, clip_range)
        norm = norm.reshape(-1, self.n_agent, int(self.size / 2))
        return norm

    def normalize_g(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range  
        if type(v) == type(torch.zeros([0])):
            return torch.clamp((v - self.torch_mean) / self.torch_std, -clip_range, clip_range)
        else:
            return np.clip((v - self.np_mean) / self.np_std, -clip_range, clip_range)
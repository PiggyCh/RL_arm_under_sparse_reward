import numpy as np

class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]         #时间步数
        rollout_batch_size = episode_batch['actions'].shape[0]    #batch 有多少rollout  也就是episode
        batch_size = batch_size_in_transitions
        ''' select which rollouts and which timesteps to be used
        #  np.random.randint(low, high=None, size=None, dtype='l') low:生成元素的最小值
        # high:生成元素的值一定小于high值
        # size:输出的大小，可以是整数也可以是元组
        # dtype:生成元素的数据类型
        # 注意：high不为None，生成元素的值在[low,high)区间中；如果high=None，生成的区间为[0,low)区间
        '''
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        # her idx
        #np.where: 按条件筛选元素
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        #np.random.uniform(size=batch_size) 150个0-1
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace goal with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions

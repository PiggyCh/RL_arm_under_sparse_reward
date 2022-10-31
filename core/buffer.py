import torch
import numpy as np
import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.resolve().as_posix())
from core.logger import Logger
from core.HER import her_sampler 
from core.normalizer import normalizer

class replay_buffer:
    def __init__(self, env_params, train_params, logger = Logger(logger="module_test")):
        self.logger = logger
        self.env_params = env_params
        self.n_agents = env_params.n_agents
        self.T = env_params.max_timesteps
        self.clip_obs = train_params.clip_obs
        self.buffer_size = train_params.buffer_size
        self.size = int(self.buffer_size // self.T)  # episode size 
        self.her_module = her_sampler(train_params.replay_strategy, train_params.replay_k)
        self.sample_fun = self.her_module.sample_her_transitions
        self.device = train_params.device
        self.specs = dict(
            obs = dict(size=(self.T, self.n_agents, self.env_params.dim_observation), dtype=torch.float32),
            ag = dict(size=(self.T, self.env_params.dim_achieved_goal), dtype=torch.float32),
            next_ag = dict(size=(self.T, self.env_params.dim_achieved_goal ), dtype=torch.float32),
            g = dict(size=(self.T, self.env_params.dim_goal ), dtype=torch.float32),
            acts = dict(size=(self.T, self.n_agents * self.env_params.dim_action ), dtype=torch.float32),
            hands = dict(size=(self.T, self.n_agents * self.env_params.dim_hand), dtype=torch.float32),
            next_obs = dict(size=(self.T, self.n_agents, self.env_params.dim_observation), dtype=torch.float32),
            reward =  dict(size=(self.T, 1), dtype=torch.float32),
        )
        self.o_norm = normalizer(
            n_agent = env_params.n_agents, 
            size = env_params.n_agents * env_params.dim_observation, 
            default_clip_range = env_params.clip_obs, 
            )
        self.g_norm = normalizer(
            n_agent = env_params.n_agents, 
            size = env_params.dim_goal, 
            default_clip_range = env_params.clip_obs, 
            )
        self.buffer_tmp = {key: [] for key in self.specs}
        logger.info(f'creating buffer, episode length : {self.T}, episode size: {self.size}')
        for _ in range(self.size):
            try:
                for key in self.specs:
                    self.buffer_tmp[key].append(torch.zeros(**self.specs[key]))
            except:
                logger.error(f'memory exceed! Turn down buffer size, now buffer size {self.size}, while creating maxsize {_}')
                break
        self.buffers = dict({key : self.buffer_tmp[key] for key in self.buffer_tmp})
        self.current_size = 0
        logger.info(f'creating buffer success: {self.specs.keys()}')

    def push(self, episode_batch):
        keys = ['obs', 'ag', 'g', 'acts', 'hands', 'next_obs', 'next_ag', 'reward']
        buffer_temp =  {key : item for key, item in zip(keys, episode_batch)}
        batch_size = buffer_temp['obs'].shape[0]
        idxs = self._get_storage_idx(inc=batch_size)
        for key in keys:
            for i, idx in enumerate(idxs):
                self.buffers[key][idx] = torch.tensor(buffer_temp[key][i], dtype= self.specs[key]['dtype'])
        self._update_normalizer(episode_batch = buffer_temp)

    def sample(self, batch_size):
        if batch_size > self.current_size:
            self.logger.warning(f"buffer current size {self.current_size} smaller than batch size {batch_size}")
        temp_buffers = {}
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][:self.current_size]
        data_tmp = self.sample_fun(temp_buffers, batch_size)
        # reward is recomputed
        data_tmp['reward'] = torch.tensor(data_tmp['reward'], dtype= self.specs['reward']['dtype'])
        transitions = {}
        for key, val in data_tmp.items():
            transitions[key] = val.clone()
        return transitions
    
    def get_norm(self):
        return {
            'o_norm' : self.o_norm,
            'g_norm' : self.g_norm
        }

    def check_real_cur_size(self):
        # check truely the buffer have data:
        for i in range(self.size):
            try:
                if torch.equal(torch.zeros(self.buffers['obs'][i].shape), self.buffers['obs'][i]):
                    return i
            except:
                self.logger.critical(f"buffer length: {len(self.buffers['obs'])} ")

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:  # buffer capacity sufficient
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        return idx

    # update the normalizer
    def _update_normalizer(self, episode_batch, init=False):
        update_data = {
            'obs' : episode_batch['obs'],
            'g' : episode_batch['g']
        }
        o = np.clip(update_data['obs'], -self.clip_obs, self.clip_obs)
        g = np.clip(update_data['g'], -self.clip_obs, self.clip_obs)
        # update
        self.o_norm.update(o)
        self.g_norm.update(g)
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

if __name__ == '__main__':
    '''
    buffer code local test...
    '''
    from pathlib import Path
    import torch.multiprocessing as mp
    import sys
    sys.path.append(Path(__file__).parent.parent.resolve().as_posix())
    from arguments import Args as args
    from copy import deepcopy
    from Env.ArmRobot_gym import ArmRobotGymEnv as env 
    from logger import Logger
    from HER import her_sampler 
    # def limit_memory(maxsize):
    #     soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    #     resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))
    # limit_memory(1024*1024*1200)   # useful wait for verification
    Env = env(
        task_params = args.task_params, 
        env_pid = 0)
    mp.set_sharing_strategy('file_system')
    logger = Logger(logger="test")
    buffer =  replay_buffer(args.env_params, args.train_params, Env.compute_reward, logger)
    # get pseudo output data shape
    obs = Env.reset()
    store_item = ['obs', 'ag', 'g', 'acts', 'hands', 'next_obs', 'next_ag', 'r']
    for cycles_times in range(100):
        mb_store_dict = {item : [] for item in store_item}
        for rollouts_times in range(2):
            ep_store_dict = {item : [] for item in store_item}
            # reset the environment
            obs_all = Env.reset()
            obs, ag, g = obs_all['observation'], obs_all['achieved_goal'], obs_all['desired_goal']
            # start to collect samples
            for t in range(100):
                action = Env.random_action()
                next_obs_all, reward, done, _ = Env.step(action)
                next_obs_, next_ag = next_obs_all['observation'], next_obs_all['achieved_goal']
                store_data = {  'obs' : obs, 
                                'ag' : ag, 
                                'g' : g,
                                'acts' : np.zeros([6]), 
                                'hands' : np.zeros([6]),
                                'next_obs': next_obs_ if t != 100 - 1 else obs,
                                'next_ag' : next_ag,
                                'r': reward}
                # append rollouts
                for key, val in store_data.items():
                    ep_store_dict[key].append(val.copy())
                # re-assign the observation
                obs = next_obs_
                ag = next_ag
            for key in store_item:
                mb_store_dict[key].append(deepcopy(ep_store_dict[key]))
        # convert them into arrays
        store_data = [np.array(val) for key, val in mb_store_dict.items()]
        # store the episodes
        buffer.push(store_data)
        logger.info(f'current size is {buffer.current_size.value}')
        if cycles_times % 10 == 0:
            transition = buffer.sample(256)
    logger.info('testing code accept')
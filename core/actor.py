import torch
import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.resolve().as_posix())
from Env.ArmRobot_gym import ArmRobotGymEnv as robot_env
from core.buffer import replay_buffer
from core.logger import Logger
from core.util import num_to_tensor
import traceback
from torch.distributions.categorical import Categorical
from copy import deepcopy
import numpy as np 

class actor_worker:
    def __init__(self, train_params, env_params, actor_index, buffer, logger = Logger(logger="test")) -> None:
        for key, val in train_params.items():
            exec(f'self.{key} = "{val}"') if type(val) == str else exec(f'self.{key} = {val}' )
        #  create the network
        self.logger = logger
        self.actor_index = actor_index
        self.env_params = env_params
        self.buffer = buffer
        self.max_timesteps = env_params.max_timesteps
        self.store_interval = train_params.store_interval

    @torch.no_grad()
    def act(
        self,
        task_params,
        model,
        data_queue
    ):
        try:
            self.logger.info(f"Actor {self.actor_index} started.")
            env = robot_env(task_params)
            store_item = ['obs', 'ag', 'g', 'acts', 'hands', 'next_obs', 'next_ag', 'r']
            while True:
                mb_store_dict = {item : [] for item in store_item}
                normalizer = self.buffer.get_norm()
                for rollouts_times in range(self.store_interval):
                    ep_store_dict = {item : [] for item in store_item}
                    # reset the environment
                    obs_all = env.reset()
                    obs, ag, g = obs_all['observation'], obs_all['achieved_goal'], obs_all['desired_goal']
                    # start to collect samples
                    for t in range(self.max_timesteps):
                        actions, acts, hands = self._select_action(model, obs, g, normalizer)  # 输入的是numpy
                        next_obs_all, reward, done, _ = env.step(actions)
                        next_obs_, next_ag = next_obs_all['observation'], next_obs_all['achieved_goal']
                        store_data = {'obs' : obs, 
                                      'ag' : ag, 
                                      'g' : g,
                                      'acts' : acts,
                                      'hands' : hands,
                                      'next_obs': next_obs_ if t != self.max_timesteps - 1 else obs,
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
                # send data to data_queue
                data_queue.put(store_data, block = True)
                # real_size = self.buffer.check_real_cur_size()
                self.logger.info(f'actor {self.actor_index} send data, current data_queue size is {self.store_interval * data_queue.qsize()}')
        except KeyboardInterrupt:
            self.logger.critical(f"interrupt")
        except Exception as e:
            self.logger.error(f"Exception in worker process {self.actor_index}")
            traceback.print_exc()
            raise e

    def _select_action(self, model, obs, g, normalizer, explore=True):
        acts = np.ones([self.n_agents, self.dim_action])  # 2*3
        hands = np.ones([self.n_agents, self.dim_hand])  # 2*1
        actions = np.ones([self.n_agents, self.dim_action+1])  # 2*4
        obs_norm = normalizer['o_norm'].normalize_obs(obs)
        obs_norm = np.squeeze(obs_norm)  # 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
        g_norm = normalizer['g_norm'].normalize_g(g)
        for i in range(self.n_agents):
            sb_norm = obs_norm[i, :]
            inputs = np.concatenate([sb_norm, g_norm])
            act, hand_logits = model.actors[i](num_to_tensor(inputs))  # actor网络得到初步的act与hand_logits
            # 处理三维的xyz
            act = act.cpu().numpy().squeeze()  # 转化为数组形式 .aqueeze()去除一维
            # add the gaussian 增加高斯噪声实现探索
            if explore:
                act += self.noise_eps * self.action_max * np.random.randn(*act.shape)
                act = np.clip(act, -self.action_max, self.action_max)
                # random actions...
                random_actions = np.random.uniform(low=-self.action_max, high=self.action_max,
                                               size=self.dim_action)
                # choose if use the random actions
                act += np.random.binomial(1, self.random_eps, 1)[0] * (random_actions - act)
                hand_id = Categorical(hand_logits).sample().detach().cpu().numpy()  # 按概率抽样

            else:
                hand_id = np.argmax(hand_logits.detach().cpu().numpy(), axis=-1)
            acts[i, :] = act
            hands[i, :] = hand_logits
            # 将act与hand_end组合
            hand_act = [self.hand_states[hand_id]]  # 获取抽到末端状态位置对应的值
            actions_ = np.concatenate((act, np.array(hand_act)), axis = -1)
            actions[i, :] = actions_
        # actions进行step，acts是六维xyz，hands是六维末端
        return actions.reshape(-1), acts.reshape(-1), hands.reshape(-1)

if __name__ == '__main__':
    '''
    testing code
    '''
    from Env import ArmRobot_gym as env
    from model import Net
    from arguments import Args as args
    from torch import multiprocessing as mp
    import time

    # set threading num to avoid possess too much cpu resource
    torch.set_num_threads(1)
    # necessary for a larger shared memory buffer
    mp.set_sharing_strategy('file_system')

    env_params = args.env_params
    train_params = args.train_params
    actor_model = Net(env_params)
    for actor in actor_model.actors:
        actor.share_memory()
    logger = Logger(logger="test")
    only_lock = mp.Lock()
    buffer = replay_buffer(env_params, train_params, logger= logger)
    num_actors = 6
    workers = [actor_worker(args, train_params.store_interval, i, buffer, only_lock, logger) for i in range(num_actors)]
    logger.info("Starting actor process...")
    ctx = mp.get_context("fork")
    actor_processes = []
    for i in range(num_actors):
        actor = ctx.Process(
            target = workers[i].act,
            args = (
                args.task_params,
                actor_model,
            )
        )
        logger.info(f"Starting actor:{i} process...")
        actor.start()
        actor_processes.append(actor)
        time.sleep(2)
    print(actor_processes)
    time.sleep(100000)
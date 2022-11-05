
import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.resolve().as_posix())

from core.util import select_action
from core.model import actor
from Env.ArmRobot_gym import ArmRobotGymEnv as robot_env
from arguments import Args

import torch
import time
import traceback
import numpy as np 
from copy import deepcopy

env_params = Args.env_params
train_params = Args.train_params
task_params = Args.task_params
max_timesteps = env_params.max_timesteps
store_interval = train_params.store_interval
n_agents = env_params.n_agents

@torch.no_grad()
def actor_worker(
    data_queue,
    actor_queue,
    actor_index,
    logger,
):
    try:
        logger.info(f"Actor {actor_index} started.")
        env = robot_env(task_params, env_pid = actor_index)
        store_item = ['obs', 'ag', 'g', 'acts', 'hands', 'next_obs', 'next_ag', 'r']
        actors = [actor(env_params) for i in range(n_agents)]
        # initialize actor worker
        normalizer = None
        # sampling ..
        while True:
            # update model params periodly
            if not actor_queue.empty():
                data = actor_queue.get()
                normalizer = data['normalizer']
                for i in range(n_agents):
                    actors[i].load_state_dict(data['actor_dict'][i])
            # first time initialization
            elif not normalizer:
                time.sleep(5)
                continue
            mb_store_dict = {item : [] for item in store_item}
            for rollouts_times in range(store_interval):
                ep_store_dict = {item : [] for item in store_item}
                obs_all = env.reset_task() # reset the environment
                obs, ag, g = obs_all['observation'], obs_all['achieved_goal'], obs_all['desired_goal']
                # start to collect samples
                for t in range(max_timesteps):
                    actions, acts, hands = select_action(actors, obs, g, normalizer, explore = True)  # 输入的是numpy
                    next_obs_all, reward, done, _ = env.step(actions)
                    next_obs_, next_ag = next_obs_all['observation'], next_obs_all['achieved_goal']
                    store_data = {
                        'obs' : obs, 
                        'ag' : ag, 
                        'g' : g,
                        'acts' : acts,
                        'hands' : hands,
                        'next_obs': next_obs_ if t != max_timesteps - 1 else obs,
                        'next_ag' : next_ag,
                        'r': reward
                    }
                    # append rollouts
                    for key, val in store_data.items():
                        ep_store_dict[key].append(val.copy())
                    obs = next_obs_
                    ag = next_ag
                for key in store_item:
                    mb_store_dict[key].append(deepcopy(ep_store_dict[key]))
            # convert them into arrays
            store_data = [np.array(val) for key, val in mb_store_dict.items()]
            # send data to data_queue
            data_queue.put(store_data, block = True)
            # real_size = self.buffer.check_real_cur_size()
            logger.info(f'actor {actor_index} send data, current data_queue size is {store_interval * data_queue.qsize()}')
    except KeyboardInterrupt:
        logger.critical(f"interrupt")
    except Exception as e:
        logger.error(f"Exception in worker process {actor_index}")
        traceback.print_exc()
        raise e

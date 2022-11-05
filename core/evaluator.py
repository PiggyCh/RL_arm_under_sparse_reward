#! /usr/bin/env python
import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.resolve().as_posix())

import time
import os
import csv
from matplotlib import pyplot as plt

from core.util import select_action
from core.model import actor
from Env.ArmRobot_gym import ArmRobotGymEnv as robot_env

def initialize_csv(path):
    file_name = os.path.join(path, 'training_data.csv')
    keys = ['step' , 'actor_loss', 'critic_loss', 'success_rate']
    with open(file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
    return file_name
    
def evaluate_worker(
        train_params,
        env_params,
        task_params,
        plot_path,
        evalue_time,
        evalue_queue,
        logger
    ):
    csv_file_name = initialize_csv(plot_path)
    env = robot_env(task_params)
    actors = [actor(env_params) for i in range(env_params.n_agents)]
    while True:
        if not evalue_queue.empty():
            data = evalue_queue.get()
            evaluate_step = data['step']
            for i in range(env_params.n_agents):
                actors[i].load_state_dict(data['actor_dict'][i])
                actors[i].eval()
            is_successes = 0
            for i in range(evalue_time):
                observation = env.reset_task()
                # start to do the demo
                obs, g = observation['observation'], observation['desired_goal']
                for t in range(env_params['max_timesteps']):
                    action, _, _ = select_action(actors, obs, g, data['normalizer'], explore = False)
                    # put actions into the environment
                    observation_new, reward, _, info = env.step(action)
                    obs, g = observation_new['observation'], observation_new['desired_goal']
                    if info['is_success'] == 1:
                        is_successes += 1
                        break
            logger.info(f' evaluate_step : {evaluate_step} success rate:{is_successes/evalue_time}')
            plot(
                train_params.env_name,
                f'{plot_path}/plot.png', 
                csv_file_name, 
                {
                    'step' : evaluate_step,
                    'actor_loss' : data['actor_loss'],
                    'critic_loss': data['critic_loss'],
                    'success_rate': is_successes/evalue_time
                }
            )
        else:
            time.sleep(30)

def plot(test_env_name, plot_path, csv_file_name, data):
    total_data = {key : [] for key in data.keys()}
    # write
    with open(csv_file_name, mode = 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([value for value in data.values()])
    # read 
    with open(csv_file_name, mode = 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in total_data.keys():
                total_data[key].append(float(row[key]))
    total_data = {key : val for key,val in total_data.items()}
    N_COLUMN = 3
    fig, axes = plt.subplots(nrows = 1, ncols = N_COLUMN, figsize=(18,6))
    fig.suptitle(test_env_name, fontsize=10)

    for i, key in enumerate(total_data.keys()):
        if key == 'step':
            continue
        ax = axes[i-1]
        ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='x')
        ax.set_title(key)
        ax.plot(total_data["step"], total_data[key])
    plt.savefig(plot_path)
    plt.close()


#  from random import random
#  import numpy as np
#  from arguments import Args
# time = 0
# def generate_data(): 
#     global time
#     time += 1
#     return  {
#         'step' :2000*time,
#         'actor_loss' : 0.5 + random(),
#         'critic_loss': 0.6 + random(),
#         'success_rate': 0.8 + random()
#     }
# test_env_name = 'armrobot_push_ seed125_10_31_21'
# plot_path = os.path.join(Args.train_params.save_dir, test_env_name)
# csv_file_name = initialize_csv(plot_path, generate_data().keys())  
# for _ in range(1000):
#     plot(
#         test_env_name,
#         f'{plot_path}/plot.png', 
#         csv_file_name, 
#         generate_data()
#     )

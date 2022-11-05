#! /usr/bin/env python
import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.resolve().as_posix())

from get_demo_data_push_act4 import get_pick_demo
import multiprocessing as mp
import time
import numpy as np

ctx = mp.get_context("fork")
data_queue = ctx.Queue()
worker_num = 10
epsiode_sampling_num = 1000
task_type = 'push'
workers = []
for i in range(worker_num):
    actor = ctx.Process(
        target = get_pick_demo,
        args = (
            data_queue,
            i
            # need to ill arguments
        )
    )
    actor.start()
    workers.append(actor)
    time.sleep(1)
current_size = 0

keys = ['obs', 'ag', 'g', 'acts', 'hands', 'next_obs', 'next_ag', 'r', 'info']
container = {key : [] for key in keys}
while current_size < epsiode_sampling_num :
    if not data_queue.empty():
        recieved = data_queue.get()
        for key in keys:
            container[key].append(recieved[key])
        current_size = len(container[keys[0]])
        print(f'recieving data, current get {current_size}')

file_name = f"armrobot_{epsiode_sampling_num}_{task_type}_demo.npz"
save_data = {}
save_data = {key : np.array(container[key]).squeeze() for key in keys}
save_data['r'] = np.expand_dims(save_data['r'], axis=2)
np.savez_compressed(
    file_name, 
    obs=save_data['obs'],
    ag=save_data['ag'],
    g=save_data['g'],
    acts = save_data['acts'],
    hands=save_data['hands'],
    next_obs=save_data['next_obs'],
    next_ag=save_data['next_ag'],
    reward=save_data['r'],
    info=save_data['info']
)
print('compelete!')
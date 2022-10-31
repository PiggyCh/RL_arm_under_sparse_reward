# #! /usr/bin/env python
import random
import torch
import numpy as np 
from arguments import Args as args
from Env.ArmRobot_gym import ArmRobotGymEnv as Env
from core.logger import Logger
from core.model import Net
from core.actor import actor_worker
from core.learner import Learner
from core.evaluator import evaluate_worker
import torch.multiprocessing as mp
import time

# set logging level 
logger = Logger(logger="dual_arm_multiprocess")

def create_env(env_pid = 0):
    return Env(
        task_params = args.task_params, 
        env_pid = env_pid)

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def init_demo_buffer(fileName, buffer):
    '''  wait to test
    '''
    demo_data = np.load(fileName, allow_pickle=True)
    store_item = ['obs', 'ag', 'g', 'acts', 'hands', 'next_obs', 'next_ag', 'reward']
    store_data = [np.array(demo_data[key]) for key in store_item.items()]
    # demo_buffer.push(store_data)
    buffer.push(store_data)

def train():
    env_params = args.env_params
    train_params = args.train_params
    task_params = args.task_params
    actor_num = train_params.actor_num
    setup_seed(args.seed)
    logger.info(f'New experiment date: {args.date}, seed: {args.seed}')

    actor_model = Net(env_params)
    learner_model = Net(env_params, train_params.device)
    for actor in actor_model.actors:
        actor.share_memory()
    logger.info(f'Create actor on {actor_model.device}, create learner on {learner_model.device}')

    # fill blank for checkpoint

    # starting multiprocess
    ctx = mp.get_context("fork")
    actor_model.update(learner_model)
    # data_pipe 
    data_queue = ctx.Queue()
    evalue_queue = ctx.Queue()
    # create buffer (multiprocessing DDPG, larger buffer)
    logger.info("Starting learner process...")
    learner = Learner(            
        env_params,
        train_params,
        actor_model,
        learner_model,
        data_queue,
        evalue_queue,
        logger)
    buffer = learner.reset_buffer()
    if train_params.add_demo:
        init_demo_buffer(train_params.demo_name, buffer)
    # starting actor worker process
    workers = [actor_worker(train_params, env_params, i, buffer, logger) for i in range(actor_num)]
    logger.info("Starting actor process...")
    actor_processes = []
    for i in range(actor_num):
        actor = ctx.Process(
            target = workers[i].act,
            args = (
                task_params,
                actor_model,
                data_queue
            )
        )
        logger.info(f"Starting actor:{i} process...")
        actor.start()
        actor_processes.append(actor)
        time.sleep(2)
    # starting learner worker process
    learner_process = ctx.Process(
        target = learner.learn,
        args = ()
    )
    logger.info(f"Starting evaluate process...")
    evaluate_process = ctx.Process(
        target = evaluate_worker,
        args = (
            train_params,
            env_params,
            task_params,
            train_params.evalue_time,
            evalue_queue,
            logger
        )
    )
    evaluate_process.start()
    learner_process.start()
    logger.info(f"actor_process:{actor_processes}")
    logger.info(f"evaluate_process:{evaluate_process}")
    logger.info(f"learner_process:{learner_process}")
    learner_process.join()
    time.sleep(100000)

if __name__ == "__main__":
    # set threading num to avoid possess too much cpu resource
    import os
    os.environ['OMP_NUM_THREADS'] = "1"
    os.environ['MKL_NUM_THREADS'] = "1"
    torch.set_num_threads(1)
    # necessary for a larger shared memory buffer
    mp.set_sharing_strategy('file_system')
    train()
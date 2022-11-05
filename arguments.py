"""
Here are the params for training
"""
from easydict import EasyDict as edict
import time

class Args:
    task_type = "push"  # or "pick"
    Use_GUI = False  # GUI is for visualizing the training process
    time_date = time.localtime()
    date = f'{time_date.tm_mon}_{time_date.tm_mday}_{time_date.tm_hour}_{time_date.tm_min}'
    seed = 125  # 123
    action_dim = 4
    n_agent = 2
    clip_obs = 5
    actor_num = 8
    clip_range = 200
    action_bound = 0.5
    demo_length = 25  # 20
    env_params = edict({    
        'n_agents' :  n_agent,
        'dim_observation' : 24, 
        'dim_action' : 3,
        'dim_hand' :  3,
        'dim_achieved_goal' :  3,
        'clip_obs' : clip_obs,
        'dim_goal' :  3,
        'max_timesteps' : 100,
        'action_max' : 0.15
        })

    train_params = edict({
        # params for multipross
        'learner_step' : int(1e7),
        'update_tar_interval' : 40,
        'evalue_interval' : 4000,
        'evalue_time' : 25,  # evaluation num per epoch
        'store_interval': 10,
        'actor_num' : actor_num,
        'date' : date,
        'checkpoint' : None,
        'polyak' : 0.95,  # 软更新率
        'action_l2' : 1, #  actor_loss += self.args.action_l2 * (acts_real_tensor / self.env_params['action_max']).pow(2).mean()
        'noise_eps' : 0.01,  # epsillon 精度
        'random_eps' : 0.3,
        'theta' : 0.1, # GAIL reward weight
        'Is_train_discrim': True,
        'roll_time' : 2,
        'gamma' : 0.98,
        'batch_size' :  256,
        'buffer_size' : 1e6, 
        'hand_states' : [-1, 0, 0.5],
        'device' : 'cpu',
        'lr_actor' : 0.001,
        'lr_critic' : 0.001,
        'lr_disc' : 0.001,
        'clip_obs' : clip_obs,
        'clip_range' : 200,
        'add_demo' : True,
        'save_dir' : 'saved_models/',
        'seed' : seed,
        'env_name' : 'armrobot_' + str(task_type) + '_' + "seed" +str(seed) + '_' + str(date),
        'demo_name' : 'armrobot_100_push_demo.npz',
        'replay_strategy' : 'future',# 后见经验采样策略
        'replay_k' :  4  # 后见经验采样的参数
    })

    train_params.update(env_params)

    robot_params = edict({
        'useInverseKinematics': 1,  # 反动力学  IK
        'useSimulation': 1,  # 仿真
        'maxForce': 500,
        'ArmRobot_endRt': 11,  # 控制机器人右臂末端的节点
        'ArmRobot_endLt': 22, # 控制机器人左臂末端的节点
        'joint_ll': [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05],  # 定位滑块和转动(铰链)关节的下限。
        'joint_ul': [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05],  # 滑动块和转动关节的位置上限
        'IS_GUI' : Use_GUI,
        'plane_path' : "plane.urdf",
        'robot_URDF_path':'URDF_model/bmirobot_description/urdf/robotarm_description.urdf',
        'robot_base_pos': [-0.10, 0.00, 0.07],
        'robot_base_orientation': [0.0, 0.0, 0.0, 1.0],
        'table_path' : 'table/table.urdf',
        'table_base_pos' : [0, 0.3, -0.45],
        'limit_x' : [-1, 1],
        'limit_y' : [-1, 1],
        'limit_z' : [-1, 1],
        'r_arm_joint' : list(range(3, 10)), 
        'l_arm_joint' : list(range(14, 21)),
        'r_hand_joint' : [10, 11],
        'l_hand_joint' : [21, 22],
    })

    task_params = edict({
        'task_type' : task_type,
        'joint_num' : 24,
        'max_gen_time' : 100, # try to generate a different cube tareet pos
        'reward_type' : 'sparse',
        'pick_has_block' : True ,  # 环境中是否有物块
        'distance_threshold' : 0.05, 
        'n_substeps' : 20,
        'n_actions' : action_dim,
        '_timeStep' : 1. / 240.,
        'n_agent' : n_agent,
        '_action_bound' : 0.5,
        'action_high' : [action_bound] * action_dim * n_agent,
        'x_gen_range' : [-0.35, 0.35],
        'y_gen_range' : [0.3, 0.45],
        'z_gen_range' : [0.2, 0.5],
        'cube_pick' : "URDF_model/cube_small_pick.urdf",
        'cube_target' : "URDF_model/cube_small_target_pick.urdf",
    })

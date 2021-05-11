import os
import torch
from models import actor
from arguments import get_args, Args
import gym
import numpy as np
from bmirobot_env.bmirobot_push_F import bmirobotGympushEnv as bmenv
actions = []
observations = []
a_goals, d_goals = [], []
infos = []

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

if __name__ == '__main__':
    args = Args()
    # load the model param
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    model_path="saved_models/bmirobot-v3/125_True13_model.pt"
    #model_path = args.save_dir + args.env_name + '/' + str(args.seed) + '_' + str(args.add_demo) + '_model.pt'
    o_mean, o_std, g_mean, g_std, model,= torch.load(model_path, map_location=lambda storage, loc: storage)
    # create the environment+

    env = bmenv()
    #env = bmgv(renders=True, isDiscrete=False)
    #env = gym.make(args.env_name)
    # get the env param
    observation = env.reset()
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0],
                  'goal': observation['desired_goal'].shape[0],
                  'action': env.action_space.shape[0],
                  'action_max': env.action_space.high[0],
                  }
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()
    is_successes = 0
    for i in range(args.demo_length):
        observation = env.reset()
        # start to do the demo
        obs = observation['observation']
        ag = observation['achieved_goal']
        g = observation['desired_goal']
        episodeAcs, episodeObs, episodeInfo = [], [], []
        episodeAg, episodeDg = [], []

        episodeObs.append(obs.copy())
        episodeAg.append(ag.copy())
        max_episode_steps=150
        for t in range(max_episode_steps):
            #env.render()
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
            # put actions into the environment
            observation_new, reward, _, info = env.step(action)
            obs = observation_new['observation']

            ag = observation_new['achieved_goal']
            g = observation_new['desired_goal']

            episodeAcs.append(action)
            episodeObs.append(obs)
            episodeAg.append(ag)
            episodeDg.append(g)
            episodeInfo.append(info)
            if info['is_success']==1:
                break
        with open("end_effector_pos", 'a') as f:
            for ob in episodeObs:
                f.write(str(ob[:3]) + '\n')
        break
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))
        ######################################################################################
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ob_x=[]
    ob_y=[]
    ob_z=[]
    for ob in list(episodeObs):
        ob_x.append(ob[0])
        ob_y.append(ob[1])
        ob_z.append(ob[2])
    xData = np.arange(0, len(episodeObs), 1)
    yData1 = np.array(ob_x) # sigmod
    yData2 = np.array(ob_y) # Relu
    yData3 = np.array(ob_z) # Softplus

    ########################
    args = Args()
    # load the model param
    model_path="saved_models/127_True105_push_vf418_grip_model.pt"
    #model_path = args.save_dir + args.env_name + '/' + str(args.seed) + '_' + str(args.add_demo) + '_model.pt'
    o_mean, o_std, g_mean, g_std, model,model2 = torch.load(model_path, map_location=lambda storage, loc: storage)
    # create the environment+

    #env = bmenv()
    #env = bmgv(renders=True, isDiscrete=False)
    #env = gym.make(args.env_name)
    # get the env param
    observation = env.reset()
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0],
                  'goal': observation['desired_goal'].shape[0],
                  'action': env.action_space.shape[0],
                  'action_max': env.action_space.high[0],
                  }
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()
    is_successes = 0
    for i in range(args.demo_length):
        observation = env.reset()
        # start to do the demo
        obs = observation['observation']
        ag = observation['achieved_goal']
        g = observation['desired_goal']
        episodeAcs, episodeObs, episodeInfo = [], [], []
        episodeAg, episodeDg = [], []

        episodeObs.append(obs.copy())
        episodeAg.append(ag.copy())
        max_episode_steps=150
        for t in range(max_episode_steps):
            #env.render()
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
            action=np.clip(action,-0.1,0.1)
            # put actions into the environment
            observation_new, reward, _, info = env.step(action)
            obs = observation_new['observation']

            ag = observation_new['achieved_goal']
            g = observation_new['desired_goal']

            episodeAcs.append(action)
            episodeObs.append(obs)
            episodeAg.append(ag)
            episodeDg.append(g)
            episodeInfo.append(info)
            if info['is_success']==1:
                break
        with open("end_effector_pos", 'a') as f:
            for ob in episodeObs:
                f.write(str(ob[:3]) + '\n')
        break
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))
        ######################################################################################

    fig, ax = plt.subplots(figsize=(8, 4))
    ob_x1=[]
    ob_y1=[]
    ob_z1=[]
    for ob in list(episodeObs):
        ob_x1.append(ob[0])
        ob_y1.append(ob[1])
        ob_z1.append(ob[2])
    #########################

    xData_1 = np.arange(0, len(episodeObs), 1)
    yData1_1 = np.array(ob_x1)  # sigmod
    yData2_1 = np.array(ob_y1) # Relu
    yData3_1 = np.array(ob_z1)  # Softplus

    from scipy.interpolate import make_interp_spline

    import numpy as np

    # list_x_new = np.linspace(min(xData_1), max(xData_1), 1000)

    # list_y_smooth = make_interp_spline(xData_1, yData1_1, list_x_new)


############平滑#############################
    import matplotlib.pyplot as plt
   #  from scipy.interpolate import make_interp_spline
   # # T = np.array([6, 7, 8, 9, 10, 11, 12])
   #  #power = np.array([1.53E+03, 5.92E+02, 2.04E+02, 7.24E+01, 2.72E+01, 1.10E+01, 4.70E+00])
   #
   #  #from scipy.interpolate import spline
   #
   #  xnew = np.linspace(xData.min(), xData.max(), 300)  # 300 represents number of points to make between T.min and T.max
   #  power_smooth = make_interp_spline(xData, yData1, xnew)
    ############平滑#############################
    ############平滑#############################
    x_smooth1 = np.linspace(min(xData), max(xData),(max(xData)-min(xData))*2)
    y_smooth1 = make_interp_spline(xData, yData1)(x_smooth1)
    ax.plot(x_smooth1, y_smooth1, color='blue', linestyle='-', linewidth=2, label='x')
    x_smooth2 = np.linspace(min(xData_1), max(xData_1), (max(xData_1)-min(xData_1))*2)
    y_smooth2 = make_interp_spline(xData_1, yData1_1)(x_smooth2)
    ax.plot(x_smooth2, y_smooth2, color='skyblue', linestyle='-', linewidth=2, label='x\'')
   #
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    ax.grid()
    ax.set_xticks(xData_1)
    # ax.set_yticks(np.arange(0, 5, 0.5))
    ax.set_xlabel('timesteps')
    ax.set_ylabel('m')
    ax.legend()
    plt.savefig('1_11.png')
    # plt.savefig('./fig.png', dpi=100)
    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 4))
    ##################
    x_smooth1 = np.linspace(min(xData), max(xData),(max(xData)-min(xData))*2)
    y_smooth1 = make_interp_spline(xData, yData2)(x_smooth1)
    ax.plot(x_smooth1, y_smooth1, color='blue', linestyle='-', linewidth=2, label='y')
    x_smooth2 = np.linspace(min(xData_1), max(xData_1), (max(xData_1)-min(xData_1))*2)
    y_smooth2 = make_interp_spline(xData_1, yData2_1)(x_smooth2)
    ax.plot(x_smooth2, y_smooth2, color='skyblue', linestyle='-', linewidth=2, label='y\'')
    #
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    ax.grid()
    ax.set_xticks(xData_1)
    # ax.set_yticks(np.arange(0, 5, 0.5))
    ax.set_xlabel('timesteps')
    ax.set_ylabel('m')
    ax.legend()
    plt.savefig('2_11.png')
    # plt.savefig('./fig.png', dpi=100)
    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 4))
    ####################
    ##################
    x_smooth1 = np.linspace(min(xData), max(xData), (max(xData)-min(xData))*2)
    y_smooth1 = make_interp_spline(xData, yData3)(x_smooth1)
    ax.plot(x_smooth1, y_smooth1, color='blue', linestyle='-', linewidth=2, label='z')
    x_smooth2 = np.linspace(min(xData_1), max(xData_1), (max(xData_1)-min(xData_1))*2)
    y_smooth2 = make_interp_spline(xData_1, yData3_1)(x_smooth2)
    ax.plot(x_smooth2, y_smooth2, color='skyblue', linestyle='-', linewidth=2, label='z\'')
    #
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    ax.grid()
    ax.set_xticks(xData_1)
    # ax.set_yticks(np.arange(0, 5, 0.5))
    ax.set_xlabel('timesteps')
    ax.set_ylabel('m')

    ax.legend()
    plt.savefig('3_11.png')
    # plt.savefig('./fig.png', dpi=100)
    plt.clf()
    ####################
    ax.plot(xData, yData2, color='red', linestyle='-', linewidth=2, label='y',)
    ax.plot(xData, yData3, color='blue', linestyle='-', linewidth=2, label='z')
    ax.plot(xData_1, yData1_1, color='lime', linestyle='-', linewidth=2, label='x\'')
    ax.plot(xData_1, yData2_1, color='orangered', linestyle='-', linewidth=2, label='y\'')
    ax.plot(xData_1, yData3_1, color='slateblue', linestyle='-', linewidth=2, label='z\'')


        # ######################################################################################
        # is_successes += info['is_success']
        # if info['is_success'] == 1.0:
        #     actions.append(episodeAcs)
        #     observations.append(episodeObs)
        #     a_goals.append(episodeAg)
        #     d_goals.append(episodeDg)
        #     infos.append(episodeInfo)

    print('demo test success rate:', is_successes/args.demo_length)

    fileName = os.path.join("demo_data/", args.env_name)
    fileName += "_" + str(args.seed) + '_' + str(args.demo_length) + "_demo.npz"
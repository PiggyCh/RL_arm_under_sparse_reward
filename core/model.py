import torch as th
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.max_action = env_params.action_max
        obs_dim = env_params.dim_observation * env_params.n_agents  # 2*24=48
        act_dim = env_params.dim_action * env_params.n_agents  # 2*3=6
        hand_dim = env_params.dim_hand * env_params.n_agents  # 2*3=6
        self.FC1 = nn.Linear(obs_dim + act_dim + hand_dim + env_params.dim_goal, 512)   # 48+6+6+3
        self.FC2 = nn.Linear(512, 512)
        self.FC3 = nn.Linear(512, 256)
        self.q_out = nn.Linear(256, 1)
        self.RELU = nn.ReLU()

    def forward(self, obs_and_g, acts, hand):  # 前向传播  acts 6   hand 6
        combined = th.cat([obs_and_g, acts / self.max_action, hand], dim=1)  # 将各个agent的观察和动作联合到一起
        result = self.RELU(self.FC1(combined))  # relu为激活函数 if输入大于0，直接返回作为输入值；else 是0或更小，返回值0。
        result = self.RELU(self.FC2(result))
        result = self.RELU(self.FC3(result))
        q_value = self.q_out(result)
        return q_value

class Discriminator(nn.Module):
    def __init__(self, env_params):
        super().__init__()
        self.max_action = env_params.action_max
        obs_dim = env_params.dim_observation * env_params.n_agents # 2*24=48
        act_dim = env_params.dim_action * env_params.n_agents  # 2*3=6
        hand_dim = env_params.dim_hand * env_params.n_agents  # 2*3=6

        self.FC1 = nn.Linear(obs_dim + act_dim + hand_dim + env_params.dim_goal, 256)   # 48+6+6+3
        self.FC2 = nn.Linear(256, 256)
        self.FC3 = nn.Linear(256, 1)

        self.FC3.weight.data.mul_(0.1)
        self.FC3.bias.data.mul_(0.0)
        self.RELU = nn.Tanh()

    def forward(self, obs_and_g, acts, hand):  # 前向传播
        combined = th.cat([obs_and_g, acts / self.max_action, hand], dim=1)  # 将各个agent的观察和动作联合到一起
        result = th.tanh(self.FC1(combined))  # relu为激活函数 if输入大于0，直接返回作为输入值；else 是0或更小，返回值0。
        result = th.tanh(self.FC2(result))
        res = th.sigmoid(self.FC3(result))
        return res

    def get_reward(self, states, actions, hand):
        with th.no_grad():
            return th.log(self.forward(states, actions, hand))


class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params.action_max
        self.FC1 = nn.Linear(env_params.dim_observation + env_params.dim_goal, 256)  # 24+3
        self.FC2 = nn.Linear(256, 256)   # FC为full_connected ，即初始化一个全连接网络
        self.FC3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params.dim_action)
        self.hand_head = nn.Linear(256, env_params.dim_hand)
        self.RELU = nn.ReLU()
        self.Tanh = nn.Tanh()

    def forward(self, obs_and_g):
        result = self.RELU(self.FC1(obs_and_g))
        result = self.RELU(self.FC2(result))
        result = self.RELU(self.FC3(result))

        actions = self.max_action * self.Tanh(self.action_out(result))  # 采用tanh激活函数，确保输出在[-1,1]之间 3维
        hand_logits = F.softmax(self.hand_head(result), dim=-1)  # hand_logits 为末端状态选择的概率 3维
        # hand_logits = F.softmax(self.hand_head(result))  # hand_logits 为末端状态选择的概率 3维S
        return actions, hand_logits

class Net():
    def __init__(self, env_params, device = 'cpu'):
        self.device = device
        self.env_params = env_params
        self.n_agents = env_params.n_agents
        # self.disc = Discriminator(args.env_params)  # if imitation learning
        self.actors = [actor(env_params).to(device) for i in range(self.n_agents)]
        self.critics = [critic(env_params).to(device) for i in range(self.n_agents)]
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)
        # load the weights into the target networks 可以将预训练的参数权重加载到新的模型之中
        [self.actors_target[i].load_state_dict(self.actors[i].state_dict()) for i in range(self.n_agents)]
        [self.critics_target[i].load_state_dict(self.critics[i].state_dict()) for i in range(self.n_agents)]

    def update(self, model): 
        for i in range(len(self.actors)):
            self.actors[i].load_state_dict(model.actors[i].state_dict())

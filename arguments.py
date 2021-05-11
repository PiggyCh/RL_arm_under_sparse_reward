import argparse
#argparse的库可以在命令行中传入参数并让程序运行
"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    #type是传入的参数数据类型  help是该参数的提示信息  default是默认参数

    '''
    使用时：
    python demo.py -h
    '''
    parser.add_argument('--env-name', type=str,
                        default='HandManipulateBlockRotateZ-v0', help='the environment name')  # 'FetchPush-v1'
    parser.add_argument('--n-epochs', type=int, default=50,
                        help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=50,
                        help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=40,
                        help='the times to update the network')
    parser.add_argument('--save-interval', type=int, default=5,
                        help='the interval that save the trajectory')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str,
                        default='future', help='the HER strategy')
    parser.add_argument('--clip-return', type=float,
                        default=50, help='if clip the returns')
    parser.add_argument('--save-dir', type=str,
                        default='saved_models/', help='the path to save the models')
    parser.add_argument('--noise-eps', type=float,
                        default=0.2, help='noise eps')
    parser.add_argument('--random-eps', type=float,
                        default=0.3, help='random eps')
    parser.add_argument('--buffer-size', type=int,
                        default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4,
                        help='ratio to be replace')
    parser.add_argument('--clip-obs', type=float,
                        default=200, help='the clip ratio')
    parser.add_argument('--batch-size', type=int,
                        default=256, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98,
                        help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.001,
                        help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.001,
                        help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.95,
                        help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int,
                        default=10, help='the number of tests')
    parser.add_argument('--clip-range', type=float,
                        default=5, help='the clip range')
    parser.add_argument('--demo-length', type=int,
                        default=20, help='the demo length')
    parser.add_argument('--cuda', action='store_true',
                        help='if use gpu do the acceleration')
    parser.add_argument('--num-rollouts-per-mpi', type=int,
                        default=2, help='the rollouts per mpi')

    args = parser.parse_args()

    return args


class Args:
    def __init__(self):
        self.n_epochs = 200  # 50
        self.n_cycles = 50
        self.n_batches = 40
        self.save_interval = 5
        self.seed = 125  # 123
        self.num_workers = 19  # 1
        self.replay_strategy = 'future'
        self.clip_return = 50
        self.save_dir = 'saved_models/'
        self.noise_eps = 0.01
        self.random_eps = 0.3
        self.buffer_size = 1e6*1/2
        self.replay_k = 4  # replay with k random states which come from the same episode as the transition being replayed and were observed after it
        self.clip_obs = 200
        self.batch_size = 256
        self.gamma = 0.98
        self.action_l2 = 1
        self.lr_actor = 0.001
        self.lr_critic = 0.001
        self.polyak = 0.95  # 软更新率
        self.n_test_rollouts = 25 #在训练时测试次数
        self.clip_range = 5
        self.demo_length = 25  # 20
        self.cuda = False
        self.num_rollouts_per_mpi = 2
        self.add_demo = True  # add demo data or not
        self.demo_name="bmirobot_1000_push_demo.npz"
        #self.demo_name="bmirobot_1000_pick_demo.npz"
        self.train_type = "push" #or "pick"
        self.Use_GUI =True  #GUI is for visualizing the training process
        self.env_name = 'bmirobot_'+str(self.train_type)+" seed"+str( self.seed )
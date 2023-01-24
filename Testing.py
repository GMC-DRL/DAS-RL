import time

import gym
import torch, numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from utils import TensorboardLogger
from agent import *
from trainer import *
from env.ensemble import *
from env.cec_dataset import *
from env.optimizer import *
import env
import os
import tqdm
import warnings
from utils.utils import *
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
params = {
    'axes.labelsize':'20',
    'xtick.labelsize':'18',
    'ytick.labelsize':'18',
    'lines.linewidth':'3',
    'legend.fontsize':'24',
    'figure.figsize':'12,8',
}
plt.rcParams.update(params)


class DQNNet(nn.Module):
    def __init__(self, state_shape, action_shape, device):
        super().__init__()
        self.device = device
        self.model = nn.Sequential(*[
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape))
        ]).to(device)

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


class DQNNet_nf(nn.Module):
    def __init__(self, optimizer_num, device):
        super().__init__()
        self.device = device
        self.embedder1 = nn.Sequential(*[
            nn.Linear(dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.ReLU(),
        ]).to(device)
        self.embedder2 = nn.Sequential(*[
            nn.Linear(dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.ReLU(),
        ]).to(device)
        self.embedder3 = nn.Sequential(*[
            nn.Linear(dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.ReLU(),
        ]).to(device)
        self.embedder4 = nn.Sequential(*[
            nn.Linear(dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.ReLU(),
        ]).to(device)
        self.embedder_qr = nn.Sequential(*[
            nn.Linear((optimizer_num + 1) * 5, 64), nn.ReLU(),
            nn.Linear(64, 2), nn.ReLU(),
        ]).to(device)
        self.model = nn.Sequential(*[
            nn.Linear(15, 64), nn.Tanh(),
            nn.Linear(64, 16), nn.Tanh(),
            nn.Linear(16, optimizer_num), # nn.Softmax(),
        ]).to(device)

    def forward(self, obs):
        feature, best_move_op0, best_move_op1, worst_move_op0, worst_move_op1 = list(obs[:, 0]), list(obs[:, 1]), list(obs[:, 2]), list(obs[:, 3]), list(obs[:, 4])
        qr_hist = list(obs[:, 5])
        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature, dtype=torch.float).to(self.device)
        best_move_op0 = self.embedder1(torch.tensor(best_move_op0, dtype=torch.float).to(self.device))
        best_move_op1 = self.embedder2(torch.tensor(best_move_op1, dtype=torch.float).to(self.device))
        worst_move_op0 = self.embedder3(torch.tensor(worst_move_op0, dtype=torch.float).to(self.device))
        worst_move_op1 = self.embedder4(torch.tensor(worst_move_op1, dtype=torch.float).to(self.device))
        qr = self.embedder_qr(torch.tensor(qr_hist, dtype=torch.float).to(self.device))
        feature = torch.cat((feature, best_move_op0, best_move_op1, worst_move_op0, worst_move_op1, qr), dim=-1)
        batch = obs.shape[0]
        logits = self.model(feature.view(batch, -1))
        return logits


class DQNNet_mo(nn.Module):
    def __init__(self, dim, optimizer_num, device):
        super().__init__()
        self.device = device
        self.embedders = []
        for i in range(optimizer_num):
            self.embedders.append(nn.Sequential(*[
                nn.Linear(dim, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.ReLU(),
            ]).to(device))
            self.embedders.append(nn.Sequential(*[
                nn.Linear(dim, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.ReLU(),
            ]).to(device))
        self.embedder_qr = nn.Sequential(*[
            nn.Linear((optimizer_num + 1) * 5, 64), nn.ReLU(),
            nn.Linear(64, 2), nn.ReLU(),
        ]).to(device)
        self.model = nn.Sequential(*[
            nn.Linear(9 + optimizer_num * 2 + 2, 64), nn.Tanh(),
            nn.Linear(64, 16), nn.Tanh(),
            nn.Linear(16, optimizer_num), # nn.Softmax(),
        ]).to(device)

    def forward(self, obs, test=False):
        feature = list(obs[:, 0])
        qr_hist = list(obs[:, -1])
        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature, dtype=torch.float).to(self.device)
        moves = []
        for i in range(len(self.embedders)):
            moves.append(self.embedders[i](torch.tensor(list(obs[:, i + 1]), dtype=torch.float).to(self.device)))
        moves = torch.cat(moves, dim=-1)
        qr = self.embedder_qr(torch.tensor(qr_hist, dtype=torch.float).to(self.device))
        batch = obs.shape[0]
        feature = torch.cat((feature, moves, qr), dim=-1)
        logits = self.model(feature.view(batch, -1))
        if test:
            out = (feature.detach().cpu().tolist(), logits)
        else:
            out = logits
        return out


class DQNNet_mo_sep(nn.Module):
    def __init__(self, dim, optimizer_num, device):
        super().__init__()
        self.device = device
        self.embedders = nn.ModuleList([])
        for i in range(optimizer_num):
            self.embedders.append((nn.Sequential(*[
                nn.Linear(dim, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.ReLU(),
            ])).to(device))
            self.embedders.append(nn.Sequential(*[
                nn.Linear(dim, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.ReLU(),
            ]).to(device))
        self.embedder_qr = nn.Sequential(*[
            nn.Linear((optimizer_num + 1) * 5, 64), nn.ReLU(),
            nn.Linear(64, 2), nn.ReLU(),
        ]).to(device)
        self.embedder_final = nn.Sequential(*[
            nn.Linear(9 + optimizer_num * 2 + 2, 64), nn.Tanh(),
        ]).to(device)
        self.model = nn.Sequential(*[
            nn.Linear(64, 16), nn.Tanh(),
            nn.Linear(16, optimizer_num), # nn.Softmax(),
        ]).to(device)

    def forward(self, obs, test=False):
        feature = list(obs[:, 0])
        qr_hist = list(obs[:, -1])
        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature, dtype=torch.float).to(self.device)
        moves = []
        for i in range(len(self.embedders)):
            moves.append(self.embedders[i](torch.tensor(list(obs[:, i + 1]), dtype=torch.float).to(self.device)))
        moves = torch.cat(moves, dim=-1)
        qr = self.embedder_qr(torch.tensor(qr_hist, dtype=torch.float).to(self.device))
        batch = obs.shape[0]
        feature = torch.cat((feature, moves, qr), dim=-1).view(batch, -1)
        feature = self.embedder_final(feature)
        logits = self.model(feature)
        if test:
            out = (feature.detach().cpu().tolist(), logits)
        else:
            out = logits
        return out


class DQNNet_sep_woQ(nn.Module):
    def __init__(self, dim, optimizer_num, device):
        super().__init__()
        self.device = device
        self.embedders = nn.ModuleList([])
        for i in range(optimizer_num):
            self.embedders.append((nn.Sequential(*[
                nn.Linear(dim, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.ReLU(),
            ])).to(device))
            self.embedders.append(nn.Sequential(*[
                nn.Linear(dim, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.ReLU(),
            ]).to(device))
        self.embedder_final = nn.Sequential(*[
            nn.Linear(9 + optimizer_num * 2, 64), nn.Tanh(),
        ]).to(device)
        self.model = nn.Sequential(*[
            nn.Linear(64, 16), nn.Tanh(),
            nn.Linear(16, optimizer_num),  nn.Softmax(),
        ]).to(device)

    def forward(self, obs, test=False):
        feature = list(obs[:, 0])
        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature, dtype=torch.float).to(self.device)
        moves = []
        for i in range(len(self.embedders)):
            moves.append(self.embedders[i](torch.tensor(list(obs[:, i + 1]), dtype=torch.float).to(self.device)))
        moves = torch.cat(moves, dim=-1)
        batch = obs.shape[0]
        feature = torch.cat((feature, moves), dim=-1).view(batch, -1)
        feature = self.embedder_final(feature)
        logits = self.model(feature)
        if test:
            out = (feature.detach().cpu().tolist(), logits)
        else:
            out = logits
        return out


class DQNNet_sep_pod(nn.Module):
    def __init__(self, dim, optimizer_num, device):
        super().__init__()
        self.device = device
        self.embedders = nn.ModuleList([])
        for i in range(optimizer_num):
            self.embedders.append((nn.Sequential(*[
                nn.Linear(dim, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.ReLU(),
            ])).to(device))
            self.embedders.append(nn.Sequential(*[
                nn.Linear(dim, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.ReLU(),
            ]).to(device))
        self.embedder_final = nn.Sequential(*[
            nn.Linear(9 + optimizer_num * 2 + optimizer_num, 64), nn.Tanh(),
        ]).to(device)
        self.model = nn.Sequential(*[
            nn.Linear(64, 16), nn.Tanh(),
            nn.Linear(16, optimizer_num), # nn.Softmax(),
        ]).to(device)

    def forward(self, obs, test=False):
        feature = list(obs[:, 0])
        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature, dtype=torch.float).to(self.device)
        moves = []
        for i in range(len(self.embedders)):
            moves.append(self.embedders[i](torch.tensor(list(obs[:, i + 1]), dtype=torch.float).to(self.device)))
        opt_dist = torch.tensor(list(obs[:, -1]), dtype=torch.float).to(self.device)
        moves = torch.cat(moves, dim=-1)
        batch = obs.shape[0]
        feature = torch.cat((feature, moves, opt_dist), dim=-1).view(batch, -1)
        feature = self.embedder_final(feature)
        logits = self.model(feature)
        if test:
            out = (feature.detach().cpu().tolist(), logits)
        else:
            out = logits
        return out


class DQNNet_woQ_new(nn.Module):
    def __init__(self, dim, optimizer_num, device):
        super().__init__()
        self.device = device
        self.problem_embedder = nn.Sequential(*[
            nn.Linear(9, 32), nn.Tanh(),
            nn.Linear(32, 16), nn.ReLU(),
        ]).to(device)
        self.algorithm_embedder = nn.Sequential(*[
            nn.Linear(dim*2, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
        ]).to(device)
        self.embedder_final = nn.Sequential(*[
            nn.Linear(16*(3+1), 16), nn.ReLU(),
        ]).to(device)
        self.model = nn.Sequential(*[
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, optimizer_num),  nn.Softmax(),
        ]).to(device)


    def forward(self, obs, test=False, to_critic=False):
        feature = list(obs[:, 0])
        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature, dtype=torch.float).to(self.device)
        moves = []
        for i in range(1, len(list(obs[0])), 2):
            moves.append(self.algorithm_embedder(torch.cat((torch.tensor(list(obs[:, i]), dtype=torch.float), torch.tensor(list(obs[:, i + 1]), dtype=torch.float)), -1).to(self.device)))
        batch = obs.shape[0]
        problem_feature = self.problem_embedder(feature)
        algorithm_feature = torch.cat(moves, dim=-1)

        feature = self.embedder_final(torch.cat((problem_feature, algorithm_feature), -1))
        logits = self.model(feature)
        if test:
            out = (feature.detach().cpu().tolist(), logits)
        else:
            if to_critic:
                out = (logits, feature)
            else:
                out = logits
        return out


class DQNNet_woQ_new2(nn.Module):
    def __init__(self, dim, optimizer_num, device):
        super().__init__()
        self.device = device
        self.problem_embedder = nn.Sequential(*[
            nn.Linear(9, 32), nn.Tanh(),
            nn.Linear(32, 16), nn.ReLU(),
        ]).to(device)
        self.algorithm_best_embedder = nn.Sequential(*[
            nn.Linear(dim, 32), nn.ReLU(),
            nn.Linear(32, 8), nn.ReLU(),
        ]).to(device)
        self.algorithm_worst_embedder = nn.Sequential(*[
            nn.Linear(dim, 32), nn.ReLU(),
            nn.Linear(32, 8), nn.ReLU(),
        ]).to(device)
        self.embedder_final = nn.Sequential(*[
            nn.Linear(16*(3+1), 16), nn.ReLU(),
        ]).to(device)
        self.model = nn.Sequential(*[
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, optimizer_num),  nn.Softmax(),
        ]).to(device)

    def forward(self, obs, test=False, to_critic=False):
        feature = list(obs[:, 0])
        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature, dtype=torch.float).to(self.device)
        moves = []
        for i in range(1, len(list(obs[0])), 2):
            moves.append(self.algorithm_best_embedder(torch.tensor(list(obs[:, i]), dtype=torch.float).to(self.device)))
            moves.append(self.algorithm_worst_embedder(torch.tensor(list(obs[:, i + 1]), dtype=torch.float).to(self.device)))
        batch = obs.shape[0]
        problem_feature = self.problem_embedder(feature)
        algorithm_feature = torch.cat(moves, dim=-1)

        feature = self.embedder_final(torch.cat((problem_feature, algorithm_feature), -1))
        logits = self.model(feature)
        if test:
            out = (feature.detach().cpu().tolist(), logits)
        else:
            if to_critic:
                out = (logits, feature)
            else:
                out = logits
        return out


class PPO_critic(nn.Module):
    def __init__(self, dim, optimizer_num, device):
        super().__init__()
        self.device = device
        self.embedders = nn.ModuleList([])
        for i in range(optimizer_num):
            self.embedders.append((nn.Sequential(*[
                nn.Linear(dim, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.ReLU(),
            ])).to(device))
            self.embedders.append(nn.Sequential(*[
                nn.Linear(dim, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.ReLU(),
            ]).to(device))
        self.embedder_final = nn.Sequential(*[
            nn.Linear(9 + optimizer_num * 2, 64), nn.Tanh(),
        ]).to(device)
        self.model = nn.Sequential(*[
            nn.Linear(64, 16), nn.Tanh(),
            nn.Linear(16, 1), # nn.Softmax(),
        ]).to(device)

    def forward(self, obs):
        feature = list(obs[:, 0])
        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature, dtype=torch.float).to(self.device)
        moves = []
        for i in range(len(self.embedders)):
            moves.append(self.embedders[i](torch.tensor(list(obs[:, i + 1]), dtype=torch.float).to(self.device)))
        moves = torch.cat(moves, dim=-1)
        batch = obs.shape[0]
        feature = torch.cat((feature, moves), dim=-1).view(batch, -1)
        feature = self.embedder_final(feature)
        batch = obs.shape[0]
        bl_val = self.model(feature.view(batch, -1))
        return bl_val


class PPO_critic_new(nn.Module):
    def __init__(self, dim, optimizer_num, device):
        super().__init__()
        self.device = device
        self.problem_embedder = nn.Sequential(*[
            nn.Linear(9, 32), nn.Tanh(),
            nn.Linear(32, 16), nn.ReLU(),
        ]).to(device)
        self.algorithm_embedder = nn.Sequential(*[
            nn.Linear(dim*2, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
        ]).to(device)
        self.embedder_final = nn.Sequential(*[
            nn.Linear(16*(3+1), 16), nn.ReLU(),
        ]).to(device)
        self.model = nn.Sequential(*[
            nn.Linear(16, 32), nn.Tanh(),
            nn.Linear(32, 1),  # nn.Softmax(),
        ]).to(device)

    def forward(self, obs):
        feature = list(obs[:, 0])
        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature, dtype=torch.float).to(self.device)
        moves = []
        for i in range(1, len(list(obs[0])), 2):
            moves.append(self.algorithm_embedder(torch.cat((torch.tensor(list(obs[:, i]), dtype=torch.float),
                                                            torch.tensor(list(obs[:, i + 1]), dtype=torch.float)),
                                                           -1).to(self.device)))
        batch = obs.shape[0]
        problem_feature = self.problem_embedder(feature)
        algorithm_feature = torch.cat(moves, dim=-1)

        feature = self.embedder_final(torch.cat((problem_feature, algorithm_feature), -1))
        values = self.model(feature)
        return values


class PPO_critic_new2(nn.Module):
    def __init__(self, dim, optimizer_num, device):
        super().__init__()
        self.device = device
        self.model = nn.Sequential(*[
            nn.Linear(16, 32), nn.Tanh(),
            nn.Linear(32, 1),  # nn.Softmax(),
        ]).to(device)

    def forward(self, feature):
        return self.model(feature)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # parameters
    # VectorEnv = env.DummyVectorEnv
    VectorEnv = env.SubprocVectorEnv
    problem = ['Schwefel']
    subproblems = ['Ackley', 'Ellipsoidal', 'Griewank', 'Rastrigin']
    sublength = [0.1, 0.2, 0.2, 0.2, 0.3]
    Comp_lamda = [1, 10, 1]
    Comp_sigma = [10, 20, 30]
    shifted=True
    rotated=True
    Train_set = 1024
    Test_set = 1024
    rl = 'PPO'  # DQN / PPO
    buf_size = 30000
    rep_size = 3000
    dim = 10
    batch_size = 16
    MaxFEs = 200000
    period = 2500
    Epoch = 500
    epsilon = 0.3
    epsilon_decay = 0.99
    lr = 1e-5
    critic_lr = 1e-5
    lr_decay = 1
    sample_times = 2
    sample_size = -1
    sample_FEs_type = 2
    n_step = 5
    k_epoch = int(0.3*(MaxFEs // period))
    testing_repeat = 1
    testing_internal = 5
    testing_seeds = 1
    save_internal = 5
    test_seed = 1
    data_gen_seed = 2
    torch_seed = 1
    optimizers = ['NL_SHADE_RSP', 'MadDE',  'JDE21']  # 
    # optimizers = ['DE_rand2_B', 'DE_ctb_B', 'MadDE']  # 'DE_best1_B', 'DE_rtb_B','DE_best2_B' 
    state_dict = None  # 'save_policy_PPO/20220816T214023/PPO-20220816T214023-199.pth'
    device = 'cuda:0'
    resume_from_log = False
    run_time = time.strftime("%Y%m%dT%H%M%S")
    plotting_color = ['r', 'g', 'b']

    # initial
    np.random.seed(data_gen_seed)
    torch.manual_seed(torch_seed)
    data_loader = Training_Dataset(filename=None, dim=dim, num_samples=Train_set, problems=problem, biased=False, shifted=shifted, rotated=rotated,
                                   batch_size=batch_size, save_generated_data=False, problem_list=subproblems, 
                                   problem_length=sublength, indicated_specific=True, indicated_dataset=None)
    test_data = Training_Dataset(filename=None, dim=dim, num_samples=Test_set, problems=problem, biased=False, shifted=shifted, rotated=rotated,
                                 batch_size=batch_size, save_generated_data=False, problem_list=subproblems, 
                                 problem_length=sublength, indicated_specific=True, indicated_dataset=None)
    ensemble = Ensemble(optimizers, Schwefel(dim, np.random.rand(dim), np.eye(dim), 0), period, MaxFEs, sample_times, sample_size)
    np.random.seed(0)

    print('=' * 75)
    print('Running Setting:')
    print((f'Shifted ' if shifted else f'Unshifted ') + 
          (f'Rotated ' if rotated else f'Unrotated ') + 
          f'Problem: {problem} with Dim: {dim}\n'
          f'Train Dataset: {Train_set} and Test Dataset: {Test_set}\n'
          f'MaxFEs: {MaxFEs} with Period: {period}\n'
          f'Feature Sample Times: {sample_times} with Sample Size: {sample_size if sample_size > 0 else "population"}\n'
          f'External FEs Type: {sample_FEs_type}\n'
          f'Optimizers: {optimizers}\n'
          f'Agent: {rl}\n'
          f'Replay Buffer: {buf_size}\n'
          f'Replay Size: {rep_size}\n'
          f'K Epoch: {k_epoch}\n'
          f'Batch Size: {batch_size}\n'
          f'Learning Rate: {lr} with decay: {lr_decay}\n'
          f'Epoch: {Epoch}\n'
          f'Test Internal: {testing_internal} with Test Repeat: {testing_repeat}\n'
          f'Device: {device}\n'
          f'Env: {VectorEnv.__name__}\n'
          f'Loaded Model: {state_dict}\n'
          f'Runtime: {run_time}'
          )
    print('=' * 75)

    baselines = []
    for optimizer in optimizers:
        baselines.append(eval(optimizer)(dim))
    baselines.append(random_optimizer(dim, baselines))

    state_shape = ensemble.observation_space.shape or ensemble.observation_space.n
    action_shape = ensemble.action_space.shape or ensemble.action_space.n

    if rl == 'DQN':
        # DQN
        net = DQNNet_sep_woQ(dim, action_shape, device)
        if state_dict is not None:
            net.load_state_dict(torch.load(state_dict, map_location=device))
        optim = torch.optim.Adam(net.parameters(), lr=lr)
        policy = DQN(net, optim, buf_size, rep_size, device=device)
    else:
        # PPO
        net = DQNNet_woQ_new2(dim, action_shape, device)
        critic = PPO_critic_new2(dim, action_shape, device)
        if state_dict is not None:
            model = torch.load(state_dict, map_location=device)
            net.load_state_dict(model['actor'])
            # matrix = torch.eye(15).to(device)
            # x = net.model(net.embedder_final(matrix)).detach()
            # print(x / x.sum(0))
            critic.load_state_dict(model['critic'])
            # y = torch.abs(critic.model(net.embedder_final(matrix)).detach())
            # print(y / y.sum())
        optim = torch.optim.Adam(
            [{'params': net.parameters(), 'lr': lr}] +
            [{'params': critic.parameters(), 'lr': critic_lr}])
        policy = PPO(net, critic, optim, device=device)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, lr_decay)

    # 初始化logger
    writer = SummaryWriter('log/' + rl + '-' + run_time)
    logger = TensorboardLogger(writer, train_interval=batch_size, update_interval=batch_size)
    log_path = 'save_policy_' + rl + '/' + run_time
    pic_path = 'log/' + rl + '-' + run_time + '/test_pic'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(pic_path):
        os.makedirs(pic_path)

    def save_fn(policy, epoch, stamp):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy-' + stamp + f'-{epoch}.pth'))

    policy.save(log_path, 0, run_time)

    # 迭代次数和数据记录
    batch_count = 0
    test_steps = 0
    best_epoch = -1
    best_FEs = MaxFEs
    best_descent = 0
    avg_base_cost = 1e-8
    avg_baselines = None
    avg_baselines_FEs = None

    # 训练开始前测试
    if testing_internal > 0:
        print('testing...')
        time.sleep(0.1)
        test_feature = [[]] * len(optimizers)
        act_count = np.zeros(len(optimizers))
        avg_descent = np.zeros(ensemble.max_step)
        avg_FEs = 0
        for bid, problems in enumerate(test_data):
            envs = [lambda e=p: Ensemble(optimizers, e, period, MaxFEs, sample_times, sample_size, seed=testing_seeds,
                                         sample_FEs_type=sample_FEs_type) for i, p in enumerate(problems)]

            test_envs = VectorEnv(envs)
            batch_num = test_data.N // test_data.batch_size

            if rl == 'DQN':
                descent, FEs, label, act = Q_test(policy, test_envs, test_steps, testing_repeat, ensemble.max_step, bid, batch_num)
                for i in range(len(optimizers)):
                    for j in range(len(label[i])):
                        test_feature[i].append(label[i][j])
            else:
                descent, FEs, label, act = Policy_test(policy, test_envs, test_steps, testing_repeat, ensemble.max_step, bid, batch_num)
                for i in range(len(optimizers)):
                    for j in range(len(label[i])):
                        test_feature[i].append(label[i][j])
            act_count += act
            test_envs.close()
            avg_descent += descent
            avg_FEs += FEs
        avg_descent /= test_data.N // test_data.batch_size
        avg_FEs /= test_data.N // test_data.batch_size
        # print(avg_descent[-1])
        total_feature = np.concatenate(test_feature, 0)


        print('baseline testing...')
        time.sleep(0.1)

        avg_baselines = {baseline.__class__.__name__: np.zeros(ensemble.max_step) for baseline in baselines}
        avg_baselines_FEs = {baseline.__class__.__name__: 0 for baseline in baselines}
        avg_base_cost = 0
        for bid, problems in enumerate(test_data):
            envs = [lambda e=p: Ensemble(optimizers, e, MaxFEs, MaxFEs, sample_times, sample_size, seed=testing_seeds,
                                         record_period=period) for i, p in enumerate(problems)]
            base_test_env = VectorEnv(envs)
            batch_num = test_data.N // test_data.batch_size
            avg_baseline, avg_baseline_FEs, avg_gbest = baseline_test(baselines, base_test_env, testing_repeat, MaxFEs, period, bid, batch_num)
            avg_base_cost += avg_gbest
            base_test_env.close()
            for baseline in baselines:
                avg_baselines[baseline.__class__.__name__] += avg_baseline[baseline.__class__.__name__]
                avg_baselines_FEs[baseline.__class__.__name__] += avg_baseline_FEs[baseline.__class__.__name__]
        for baseline in baselines:
            avg_baselines[baseline.__class__.__name__] /= test_data.N // test_data.batch_size
            avg_baselines_FEs[baseline.__class__.__name__] /= test_data.N // test_data.batch_size
        avg_base_cost /= (test_data.N // test_data.batch_size)
        # avg_base_cost = max(avg_base_cost, 1e-8)
        avg_base_cost = 1e-8
        # 记录测试结果
        data = {'ensemble': avg_FEs}
        for k, v in avg_baselines_FEs.items():
            data[k] = v
        logger.write_together('test/FEs', test_steps, data)
        data = {'ensemble': avg_descent[-1]}
        for k, v in avg_baselines.items():
            data[k] = v[-1]
        logger.write_together('test/descent', test_steps, data)
        act_count /= np.sum(act_count)
        logger.write_together('test/action', test_steps, {f'action{i}': act_count[i] for i in range(len(act_count))})

        if best_epoch < 0 or best_FEs > avg_FEs or (best_descent < avg_descent[-1] and best_FEs == avg_FEs):
            best_epoch, best_FEs, best_descent = 0, avg_FEs, avg_descent[-1]
        plot_with_baseline(0, logger, avg_descent, avg_baselines)

        test_steps += 1
        print(f'best testing descent: {best_descent}, ending FEs: {best_FEs} in epoch {best_epoch}')
        for baseline in baselines:
            print(f'baseline {baseline.__class__.__name__} descent: {avg_baselines[baseline.__class__.__name__][-1]} ')
        time.sleep(0.1)

    total_steps = 0
    train_steps = 0
    for epoch in range(Epoch):
        data_loader.shuffle()
        for bid, problems in enumerate(data_loader):
            # 将batch问题转化为并行环境
            envs = [lambda e=p: Ensemble(optimizers, e, period, MaxFEs, sample_times, sample_size, sample_FEs_type=sample_FEs_type, terminal_error=avg_base_cost) for i, p in enumerate(problems)]

            train_envs = VectorEnv(envs)
            batch_num = data_loader.N // data_loader.batch_size
            if rl == 'DQN':
                total_steps = Q_train_v2(policy, train_envs, logger, epsilon, total_steps, ensemble.max_step, epoch, bid, batch_num)
            else:
                total_steps = Policy_train_v3(policy, train_envs, logger, total_steps, train_steps, k_epoch, ensemble.max_step, epoch, bid, batch_num)
            train_steps += k_epoch

            train_envs.close()

        epsilon *= epsilon_decay
        lr_scheduler.step()
        logger.write('train/learning rate', epoch, {'train/lr': lr_scheduler.get_lr()[-1]})
        if (epoch + 1) % save_internal == 0:
            policy.save(log_path, epoch, run_time)

        # 一个epoch完进行测试
        if testing_internal > 0 and (epoch + 1) % testing_internal == 0:
            print('testing...')
            time.sleep(0.1)

            test_feature = [[]] * len(optimizers)
            act_count = np.zeros(len(optimizers))
            avg_descent = np.zeros(ensemble.max_step)
            avg_FEs = 0
            for bid, problems in enumerate(test_data):
                envs = [
                    lambda e=p: Ensemble(optimizers, e, period, MaxFEs, sample_times, sample_size, seed=testing_seeds,
                                         sample_FEs_type=sample_FEs_type) for i, p in enumerate(problems)]

                test_envs = VectorEnv(envs)
                batch_num = test_data.N // test_data.batch_size
                if rl == 'DQN':
                    descent, FEs, label, act = Q_test(policy, test_envs, test_steps, testing_repeat, ensemble.max_step, bid, batch_num)
                    for i in range(len(optimizers)):
                        for j in range(len(label[i])):
                            test_feature[i].append(label[i][j])
                else:
                    descent, FEs, label, act = Policy_test(policy, test_envs, test_steps, testing_repeat, ensemble.max_step, bid, batch_num)
                    for i in range(len(optimizers)):
                        for j in range(len(label[i])):
                            test_feature[i].append(label[i][j])
                act_count += act
                test_envs.close()
                avg_descent += descent
                avg_FEs += FEs
            avg_descent /= test_data.N // test_data.batch_size
            avg_FEs /= test_data.N // test_data.batch_size
            total_feature = np.concatenate(test_feature, 0)

            # 记录测试结果
            data = {'ensemble': avg_FEs}
            for k, v in avg_baselines_FEs.items():
                data[k] = v
            logger.write_together('test/FEs', test_steps, data)
            data = {'ensemble': avg_descent[-1]}
            for k, v in avg_baselines.items():
                data[k] = v[-1]
            logger.write_together('test/descent', test_steps, data)
            act_count /= np.sum(act_count)
            logger.write_together('test/action', test_steps, {f'action{i}': act_count[i] for i in range(len(act_count))})

            if best_epoch < 0 or best_descent < avg_descent[-1] or (best_FEs > avg_FEs and best_descent == avg_descent[-1]):
                best_epoch, best_FEs, best_descent = epoch + 1, avg_FEs, avg_descent[-1]
            plot_with_baseline(epoch + 1, logger, avg_descent, avg_baselines)

            test_steps += 1

            print(f'best testing descent: {best_descent}, ending FEs: {best_FEs} in epoch {best_epoch}')
            print(f'testing descent: {avg_descent[-1]}, ending FEs: {avg_FEs} in epoch {epoch}')
            for baseline in baselines:
                print(f'baseline {baseline.__class__.__name__} descent: {avg_baselines[baseline.__class__.__name__][-1]}')
            time.sleep(0.1)

    # print(f'Finished training! Use {result["duration"]}')


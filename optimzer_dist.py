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
from Testing import *
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

problem = ['Schwefel']
dim = 10
optimizers = ['NL_SHADE_RSP', 'MadDE', 'JDE21']  #
state_dict = 'save_policy_PPO/20220816T214023/PPO-20220816T214023-199.pth'
subproblems = None  # ['Rastrigin', 'Happycat', 'Ackley', 'Discus', 'Rosenbrock']
sublength = None  # [0.3, 0.3, 0.4]
VectorEnv=env.DummyVectorEnv
Test_set=128
period=2500
rl = 'PPO'


def plot_optimizer_dist():
    MaxFEs = 200000
    sample_FEs_type = 2
    sample_times = 2
    sample_size = -1
    data_gen_seed = 2
    testing_seeds = 1
    run_time = time.strftime("%Y%m%dT%H%M%S")
    plotting_color = ['r', 'g', 'b']
    device = 'cuda:0'
    num_optimizers = len(optimizers)
    np.random.seed(data_gen_seed)
    test_data = Training_Dataset(filename=None, dim=dim, num_samples=Test_set, problems=problem, biased=False,
                                 batch_size=1, save_generated_data=False, problem_list=subproblems,
                                 problem_length=sublength, indicated_specific=True, )
    ensemble = Ensemble(optimizers, Schwefel(dim, np.random.rand(dim), np.eye(dim), 0), period, MaxFEs, sample_times,sample_size)

    action_shape = ensemble.action_space.shape or ensemble.action_space.n

    if rl == 'DQN':
        # DQN
        net = DQNNet_sep_woQ(dim, action_shape, device)
        if state_dict is not None:
            net.load_state_dict(torch.load(state_dict, map_location=device))
        optim = torch.optim.Adam(net.parameters(), lr=0)
        policy = DQN(net, optim, 0, 0, device=device)
    else:
        # PPO
        net = DQNNet_sep_woQ(dim, action_shape, device)
        critic = PPO_critic(dim, action_shape, device)
        if state_dict is not None:
            model = torch.load(state_dict, map_location=device)
            net.load_state_dict(model['actor'])
            critic.load_state_dict(model['critic'])
        optim = torch.optim.Adam(
            [{'params': net.parameters(), 'lr': 0}] +
            [{'params': critic.parameters(), 'lr': 0}])
        policy = PPO(net, critic, optim, device=device)

    print('=' * 75)
    print('Testing Setting:')
    print(f'Problem: {problem} with Dim: {dim}\n'
          f'Test Dataset: {Test_set}\n'
          f'MaxFEs: {MaxFEs} with Period: {period}\n'
          f'Feature Sample Times: {sample_times} with Sample Size: {sample_size if sample_size > 0 else "population"}\n'
          f'External FEs Type: {sample_FEs_type}\n'
          f'Optimizers: {optimizers}\n'
          f'Agent: DQN\n'
          f'Device: {device}\n'
          f'Env: {VectorEnv.__name__}\n'
          f'Loaded Model: {state_dict}\n'
          f'Runtime: {run_time}'
          )
    print('=' * 75)

    total_act_count = np.zeros((len(optimizers), MaxFEs//period))
    for bid, problems in enumerate(test_data):
        testing_env = VectorEnv([lambda: Ensemble(optimizers, p, period, MaxFEs, sample_times, sample_size, seed=testing_seeds, sample_FEs_type=sample_FEs_type) for i, p in enumerate(problems)])
        obs = testing_env.reset()
        is_done = False
        info = None
        avg_descent = []
        avg_FEs = 0
        feature_label = [[] for _ in range(num_optimizers)]
        act_rew = np.zeros(num_optimizers)
        act_count = np.zeros(num_optimizers)
        value = []
        act_rec = []
        pbar = tqdm.tqdm(total=ensemble.max_step,
                         desc=f'testing batch {bid + 1} / {Test_set}',
                         bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        groups = testing_env.get_env_attr('population')
        instances = testing_env.get_env_attr('problem')
        count = testing_env.get_env_attr('FEs')
        # for i in range(testing_env.env_num):
        #     plot(groups[i].group, instances[i], 'ensemble', count[i])
        step = 0
        while not is_done:
            feature, logits = policy.actor_forward_without_grad(obs, test=True)
            probs, log_probs, act = policy.actor_sample(logits)
            probs = probs.detach().cpu()
            actions = [{'action': act[i].cpu(), 'qvalue': probs[i]} for i in range(testing_env.env_num)]
            obs_next, rewards, is_done, info = testing_env.step(actions)
            obs = obs_next
            total_act_count[act][step] += 1
            step += 1
            pbar.update()
            for i in range(testing_env.env_num):
                feature_label[act[i]].append(feature[i])
                act_count[act[i]] += 1
            value.append(rewards[0])
            if value[-1] < 1e-2:
                value[-1] = -0.01

        avg_descent.append(mean_info(info, 'descent_seq'))
        avg_FEs += mean_info(info, 'FEs')
        avg_descent = np.array(avg_descent)
        pbar.close()

        print(f'testing descent: {avg_descent[0][-1]}, ending FEs: {avg_FEs}')
        print(f'action selected count: {act_count}')
        print(value)
        print(act_rec)
        print('plotting')
        color = [plotting_color[act_rec[i]] for i in range(len(act_rec))]
    fig, ax = plt.subplots(1,1,figsize=(20,9))
    ax = plt.gca()
    ax.stackplot(np.arange(MaxFEs//period), total_act_count / Test_set, labels=optimizers)
    ax.legend(fontsize=10, ncol=4)
    plt.show()


if __name__ == '__main__':
    plot_optimizer_dist()

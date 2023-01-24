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
optimizers = ['NL_SHADE_RSP', 'MadDE', 'JDE21']  #
state_dict = 'save_policy_PPO/20220816T214023/PPO-20220816T214023-199.pth'
subproblems = None  # ['Rastrigin', 'Happycat', 'Ackley', 'Discus', 'Rosenbrock']
sublength = None  # [0.3, 0.3, 0.4]
VectorEnv = env.DummyVectorEnv


def GlobalTesting(problem,
                  optimizers,
                  state_dict,
                  dim=10,
                  subproblems=None,
                  sublength=None,
                  Test_set=10,
                  batch_size=4,
                  period=2500,
                  sample_FEs_type=2,
                  testing_repeat=1,
                  testing_seeds=1,
                  VectorEnv=env.SubprocVectorEnv,
                  device='cuda:1',
                  test_baseline=True,
                  ):
    warnings.filterwarnings("ignore")
    # parameters
    # VectorEnv = env.SubprocVectorEnv
    rl = 'PPO'  # DQN / PPO

    MaxFEs = 200000
    Epoch = 100
    sample_times = 2
    sample_size = -1
    k_epoch = 10
    data_gen_seed = 2
    run_time = time.strftime("%Y%m%dT%H%M%S")
    plotting_color = ['r', 'g', 'b']

    # initial
    np.random.seed(data_gen_seed)
    test_data = Training_Dataset(filename=None, dim=dim, num_samples=Test_set, problems=problem, biased=False,
                                 batch_size=batch_size, save_generated_data=False, problem_list=subproblems,
                                 problem_length=sublength, indicated_specific=True,)
    ensemble = Ensemble(optimizers, Schwefel(dim, np.random.rand(dim), np.eye(dim), 0), period, MaxFEs, sample_times, sample_size)

    print('=' * 75)
    print('Testing Setting:')
    print(f'Problem: {problem} with Dim: {dim}\n'
          f'Test Dataset: {Test_set} with Test Repeat: {testing_repeat}\n'
          f'MaxFEs: {MaxFEs} with Period: {period}\n'
          f'Feature Sample Times: {sample_times} with Sample Size: {sample_size if sample_size > 0 else "population"}\n'
          f'External FEs Type: {sample_FEs_type}\n'
          f'Optimizers: {optimizers}\n'
          f'Agent: {rl}\n'
          f'Batch Size: {batch_size}\n'
          f'Epoch: {Epoch}\n'
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

    # 初始化logger
    writer = SummaryWriter('Test_log/' + rl + '-Test-' + run_time)
    logger = TensorboardLogger(writer, train_interval=batch_size, update_interval=batch_size)
    pic_path = 'Test_log/' + rl + '-Test-' + run_time + '/test_pic'
    if not os.path.exists(pic_path):
        os.makedirs(pic_path)

    print('testing...')
    time.sleep(0.1)

    test_feature = [[np.zeros(64)], [np.zeros(64)], [np.zeros(64)]]
    avg_descent = np.zeros(ensemble.max_step)
    avg_FEs = 0
    Rewards_abs = np.zeros(3)
    Rewards = np.zeros(3)
    Rewards_at = np.zeros(3)
    for bid, problems in enumerate(test_data):
        envs = [lambda e=p: Ensemble(optimizers, e, period, MaxFEs, sample_times, sample_size, seed=testing_seeds, sample_FEs_type=sample_FEs_type) for i, p in enumerate(problems)]

        test_envs = VectorEnv(envs)
        batch_num = test_data.N // test_data.batch_size
        if rl == 'DQN':
            descent, FEs, label, _, rewards_abs, rewards, rewards_at = Q_test(policy, test_envs, 0, testing_repeat, ensemble.max_step, bid, batch_num, ret_rew=True)
            Rewards_abs += rewards_abs
            Rewards += rewards
            Rewards_at += rewards_at
            for i in range(len(optimizers)):
                for j in range(len(label[i])):
                    test_feature[i].append(label[i][j])
        else:
            descent, FEs, label, act = Policy_test(policy, test_envs, 0, testing_repeat, ensemble.max_step, bid, batch_num)
            for i in range(len(optimizers)):
                for j in range(len(label[i])):
                    test_feature[i].append(label[i][j])

        test_envs.close()
        avg_descent += descent
        avg_FEs += FEs
    avg_descent /= test_data.N // test_data.batch_size
    avg_FEs /= test_data.N // test_data.batch_size
    print(f'testing descent: {avg_descent[-1]}, ending FEs: {avg_FEs}')
    print(f'action selected count: [{len(test_feature[0])}, {len(test_feature[1])}, {len(test_feature[2])}]')

    print('plotting')
    total_feature = np.concatenate(test_feature, 0)

    tsne = TSNE(n_components=2)
    f_label = tsne.fit_transform(total_feature)
    pic = plt.figure()
    index = 0
    for i in range(len(optimizers)):
        plt.scatter(f_label[index:index + len(test_feature[i]), 0],
                    f_label[index:index + len(test_feature[i]), 1],
                    c=plotting_color[i], s=2)
        index += len(test_feature[i])
    pic.savefig(pic_path + f'/test')
    # print(avg_descent[-1])

    if test_baseline:
        print('baseline testing...')
        time.sleep(0.1)

        avg_baselines = {baseline.__class__.__name__: np.zeros(ensemble.max_step) for baseline in baselines}
        avg_baselines_FEs = {baseline.__class__.__name__: 0 for baseline in baselines}
        avg_base_cost = 0
        for bid, problems in enumerate(test_data):
            base_test_env = VectorEnv([lambda e=p: Ensemble(optimizers, e, MaxFEs, MaxFEs, sample_times, sample_size, seed=testing_seeds, record_period=period) for i, p in enumerate(problems)])
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
        plot_with_baseline(0, logger, avg_descent, avg_baselines)
    # 记录测试结果
    data = {'ensemble': avg_FEs}
    if test_baseline:
        for k, v in avg_baselines_FEs.items():
            data[k] = v
    logger.write_together('test/FEs', 0, data)
    data = {'ensemble': avg_descent[-1]}
    if test_baseline:
        for k, v in avg_baselines.items():
            data[k] = v[-1]
    logger.write_together('test/descent', 0, data)

    print(f'testing descent: {avg_descent[-1]}, ending FEs: {avg_FEs}')
    print(f'action selected count: [{len(test_feature[0])}, {len(test_feature[1])}, {len(test_feature[2])}]')

    if test_baseline:
        for baseline in baselines:
            print(f'baseline {baseline.__class__.__name__} descent: {avg_baselines[baseline.__class__.__name__][-1]} ')


if __name__ == '__main__':
    # state_dict = 'save_policy_DQN/20220530T191922/DQN-20220530T191922-0.pth'
    # GlobalTesting(problem, optimizers, state_dict, subproblems=subproblems, sublength=sublength, VectorEnv=VectorEnv)
    #
    # state_dict = 'save_policy_DQN/20220530T191922/DQN-20220530T191922-9.pth'
    # GlobalTesting(problem, optimizers, state_dict, subproblems=subproblems, sublength=sublength, VectorEnv=VectorEnv, test_baseline=False)

    # state_dict = 'save_policy_DQN/20220621T222004/DQN-20220621T222004-29.pth'
    GlobalTesting(problem, optimizers, state_dict, subproblems=subproblems, sublength=sublength, VectorEnv=VectorEnv, test_baseline=False)






















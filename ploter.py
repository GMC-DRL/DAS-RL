import os

import torch
from sklearn.manifold import TSNE
from env.cec_dataset import *
from env.optimizer import *
from env.cec_test_func import *
from env.ensemble import *
from Testing import *
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


def f(x, y, instance):
    return instance.func(np.array(torch.cat((torch.tensor(x).unsqueeze(-1),torch.tensor(y).unsqueeze(-1),), -1).unsqueeze(0)))


# def plot(group, instance, optimizer, FEs):
#     fig = plt.figure(figsize=(5, 5))
#     ax = plt.axes(projection='3d')
#     xx = np.linspace(-100, 100, 100)
#     yy = np.linspace(-100, 100, 100)
#     Z = np.zeros([len(xx), len(yy)])
#     for i in range(len(xx)):
#         for j in range(len(yy)):
#             Z[j, i] = f(xx[i], yy[j], instance)
#     X, Y = np.meshgrid(xx, yy)
#     ax.view_init(45, 135)
#     ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
#                     cmap='viridis', edgecolor='none', alpha=0.2)
#     cost = instance.func(group)
#     ax.scatter(group[:, 0], group[:, 1], cost, s=10, marker='o', c='r')
#     b = np.argmin(cost)
#     ax.scatter(group[b, 0], group[b, 1], cost[b], s=50, marker='o', c='black')
#     title = ''
#     plt.title(optimizer + '_' + instance.__class__.__name__ + '_FEs_' + str(FEs))
#     # path = 'pic/' + optimizer + '_dist_pic'
#     # if not os.path.exists(path):
#     #     os.makedirs(path)
#     # plt.savefig(path + '/' + instance.__class__.__name__ + '_FEs_' + str(FEs))
#     plt.show()

def plot(instance):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    xx = np.linspace(-100, 100, 1000)
    yy = np.linspace(-100, 100, 1000)
    X, Y = np.meshgrid(xx, yy)
    Z = instance.func(np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=-1)).reshape(1000, 1000)
    ax.view_init(20, 45)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none', alpha=1.0)
    plt.savefig('pic/' + instance.__class__.__name__ + '.png')
    plt.show()


def plot2D(instance):
    S = 50
    x = np.linspace(-S, S, 1000)
    y = np.linspace(-S, S, 1000)
    X, Y = np.meshgrid(x, y)
    height = instance.func(np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=-1)).reshape(1000, 1000)
    plt.contourf(X, Y, height, 15, alpha=0.75)
    C = plt.contour(X, Y, height, 15, colors='black', linewidths=0.5)
    plt.clabel(C, inline=True, fontsize=5)
    # plt.show()


def scatterPop(pop):
    plt.scatter(pop[:, 0], pop[:, 1], s=10, c='r')

device = 'cuda:1'
# if __name__ == '__main__':
#     np.random.seed(114514)
#     torch.manual_seed(114514)
#     period = 500
#     dim = 2
#     state_dict = 'save_policy_PPO/20220923T001442/PPO-20220923T001442-9.pth'
#     net = DQNNet_sep_woQ(dim, 3, device)
#     critic = PPO_critic(dim, 3, device)
#     if state_dict is not None:
#         model = torch.load(state_dict, map_location=device)
#         net.load_state_dict(model['actor'])
#         critic.load_state_dict(model['critic'])
#     optim = torch.optim.Adam(
#         [{'params': net.parameters(), 'lr': 0}] +
#         [{'params': critic.parameters(), 'lr': 0}])
#     policy = PPO(net, critic, optim, device=device)
#     optimizer = MadDE(dim)
#     optimizers = ['NL_SHADE_RSP', 'MadDE', 'JDE21']
#     instance = Schwefel(dim, np.random.rand(dim)*80-40, rotate_gen(dim), 0)
#     env = Ensemble(['NL_SHADE_RSP', 'MadDE', 'JDE21'], instance, period, 50000, 2, -1)
#     obs = env.reset()
#     j=0
#     acts =[]
#     while not env.done:
#         feature, logits = policy.actor_forward_without_grad(obs, test=True)
#         probs, log_probs, act = policy.actor_sample(logits)
#         probs = probs.detach().cpu()
#         act = act.cpu()
#         actions = {'action': act, 'qvalue': probs}
#         obs, _,_,_ = env.step(actions)
#         acts.append(optimizers[act])
#     # data = np.zeros((8, 2))
#     # with open('CF2_M_D2_opt.dat', 'r') as file:
#     #     for i in range(8):
#     #         str = file.readline()
#     #         data[i][0] = float(str.split()[0])
#     #         data[i][1] = float(str.split()[1])
#     # sub_problems = [cec_test_func.Rastrigin(dim, data[0], np.eye(dim), 0),
#     #                 cec_test_func.Rastrigin(dim, data[1], np.eye(dim), 0),
#     #                 cec_test_func.Weierstrass(dim, data[2], np.eye(dim), 0),
#     #                 cec_test_func.Weierstrass(dim, data[3], np.eye(dim), 0),
#     #                 cec_test_func.Griewank(dim, data[4], np.eye(dim), 0),
#     #                 cec_test_func.Griewank(dim, data[5], np.eye(dim), 0),
#     #                 cec_test_func.Sphere(dim, data[6], np.eye(dim), 0),
#     #                 cec_test_func.Sphere(dim, data[7], np.eye(dim), 0),
#     #                 ]
#     # instance = cec_test_func.Composition(dim, 8, [1, 1, 10, 10, 1 / 10, 1 / 10, 1 / 7, 1 / 7], [1, 1, 1, 1, 1, 1, 1, 1],
#     #                               [0, 0, 0, 0, 0, 0, 0, 0, ], 0, sub_problems)
#         rec=[1000, 2500, 5000, 10000, 20000, 50000]
#         # if j < 6 and (env.FEs >= rec[j] or env.done):
#         # if not env.done:
#         plt.figure(figsize=(6, 6))
#         plot2D(instance)
#         scatterPop(env.population.group)
#         print(env.population.gbest)
#         print((j+1)*period)
#         plt.savefig(f'pic/_{(j+1)*period}.jpg')
#         plt.show()
#         j+=1
#     print(env.FEs)
#     print(acts)

np.random.seed(1)
dim = 2
instance = Bent_cigar(dim, np.zeros(dim), np.eye(dim), 0)
# sub_problems = [Rastrigin(dim, np.random.rand(dim)*160-80, np.eye(dim), 0),
#                 Griewank(dim, np.array([0, 50]), np.eye(dim), 0),
#                 Schwefel(dim, np.random.rand(dim)*160-80, np.eye(dim), 0),
#
#                 ]
# instance = cec_test_func.Composition(dim, 3, [1, 10, 1], [10, 20, 30, ],
#                                   [0, 100, 200, ], 0, sub_problems)
plot(instance)


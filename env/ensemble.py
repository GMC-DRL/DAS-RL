import time

import gym
from gym import spaces
import numpy as np
from .optimizer import *
from .utils import *
from .Population import *
import warnings
import copy
import collections


class Ensemble(gym.Env):
    def __init__(self, optimizers, problem, period, MaxFEs, sample_times, sample_size, seed=0, qr_len=5, record_period=-1, sample_FEs_type=2, terminal_error=1e-8):
        self.dim = problem.dim
        self.MaxFEs = MaxFEs
        self.period = period
        self.max_step = MaxFEs // period
        self.optimizers = []
        for optimizer in optimizers:
            self.optimizers.append(eval(optimizer)(self.dim))
        self.sample_size = sample_size
        self.sample_times = sample_times
        self.best_history = [[] for _ in range(len(optimizers))]
        self.worst_history = [[] for _ in range(len(optimizers))]
        self.q_r_history = [np.zeros(len(self.optimizers) + 1)] * qr_len
        self.qr_len = qr_len
        self.optimzer_used = np.zeros(len(self.optimizers))

        self.problem = problem
        self.n_dim_obs = 6
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_dim_obs,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(self.optimizers))
        self.sample_FEs_type = sample_FEs_type
        self.baseline = False
        if record_period > 0:
            self.baseline = True
        self.record_period = record_period if record_period > 0 else period
        self.Fevs = np.array([])
        self.seed = seed
        self.final_obs = None
        self.terminal_error = terminal_error

    def local_sample(self):
        samples = []
        costs = []
        min_len = 1e9
        sample_size = self.sample_size if self.sample_size > 0 else self.population.NP
        for i in range(self.sample_times):
            sample, _, _ = self.optimizers[np.random.randint(len(self.optimizers))].step(copy.deepcopy(self.population),
                                                                                         self.problem,
                                                                                         self.FEs,
                                                                                         self.FEs + sample_size,
                                                                                         self.MaxFEs)
            samples.append(sample)
            cost = sample.cost
            costs.append(cost)
            min_len = min(min_len, cost.shape[0])
        if self.sample_FEs_type > 0:
            if self.FEs % self.record_period + sample_size * self.sample_times >= self.record_period and not self.done:
                # print(self.Fevs.shape[0])
                self.Fevs = np.append(self.Fevs, self.population.gbest)
                # print(self.Fevs.shape[0])
            self.FEs += sample_size * self.sample_times
            if self.FEs >= self.MaxFEs:
                self.done = True
        for i in range(self.sample_times):
            costs[i] = costs[i][:min_len]
        return np.array(samples), np.array(costs)

    # observed env state
    def observe(self):
        samples, sample_costs = self.local_sample()
        feature = self.population.get_feature(self.problem,
                                              sample_costs,
                                              self.cost_scale_factor,
                                              self.FEs / self.MaxFEs)

        best_move = np.zeros((len(self.optimizers), self.dim)).tolist()
        worst_move = np.zeros((len(self.optimizers), self.dim)).tolist()
        move = np.zeros((len(self.optimizers) * 2, self.dim)).tolist()
        for i in range(len(self.optimizers)):
            if len(self.best_history[i]) > 0:
                move[i*2] = np.mean(self.best_history[i], 0).tolist()
                move[i * 2 + 1] = np.mean(self.worst_history[i], 0).tolist()
                best_move[i] = np.mean(self.best_history[i], 0).tolist()
                worst_move[i] = np.mean(self.worst_history[i], 0).tolist()
        # move = list(np.concatenate((best_move, worst_move), 0))
        move.insert(0, feature)
        # move.append(np.array(self.q_r_history).reshape(-1))
        # move.append(np.array((self.optimzer_used / np.sum(self.optimzer_used)) if np.sum(self.optimzer_used) > 0 else np.zeros(len(self.optimizers))).reshape(-1))
        return move

    def seed(self, seed=None):
        np.random.seed(seed)

    # initialize env
    def reset(self):
        np.random.seed(self.seed)
        self.population = Population(self.dim)
        self.population.initialize_costs(self.problem)
        self.cost_scale_factor = self.population.gbest
        self.FEs = self.population.NP
        self.Fevs = np.array([])
        self.done = (self.population.gbest <= self.terminal_error or self.FEs >= self.MaxFEs)
        if not self.baseline:
            return self.observe()

    def step(self, action):
        warnings.filterwarnings("ignore")
        if not self.done:
            act = action['action']
            qvalue = action['qvalue']

            last_cost = self.population.gbest
            pre_best = self.population.gbest_solution
            pre_worst = self.population.group[np.argmax(self.population.cost)]
            period = self.period if act < len(self.optimizers) else self.record_period
            Fevs = []
            start = time.time()
            end = self.FEs + self.period
            while self.FEs < end and self.FEs < self.MaxFEs and self.population.gbest > self.terminal_error:
                if action['action'] >= len(self.optimizers):
                    act = np.random.randint(len(self.optimizers))
                    
                optimizer = self.optimizers[act]
                FEs_end = self.FEs + period
                if self.sample_FEs_type == 1:
                    # FEs_end -= self.sample_times * self.sample_size
                    FEs_end -= self.FEs % period

                self.population, Fev, self.FEs = optimizer.step(self.population,
                                                                 self.problem,
                                                                 self.FEs,
                                                                 FEs_end,
                                                                 self.MaxFEs,
                                                                 self.record_period,
                                                                 )
                Fevs.extend(Fev)
            end = time.time()
            self.optimzer_used[act] += 1
            pos_best = self.population.gbest_solution
            pos_worst = self.population.group[np.argmax(self.population.cost)]
            self.best_history[act].append((pos_best - pre_best) / 200)
            self.worst_history[act].append((pos_worst - pre_worst) / 200)
            self.done = (self.population.gbest <= self.terminal_error or self.FEs >= self.MaxFEs)
            # if self.done and np.max(self.time_rec) > 2:
            #     print(self.time_rec)
            # reward = 1 * (last_cost - self.population.gbest) / last_cost
            # reward = 1 if self.population.gbest < last_cost else 0
            reward = max((last_cost - self.population.gbest) / self.cost_scale_factor, 0)
            # reward = max((last_cost - self.population.gbest) / last_cost, 0)
            # reward = np.arctan(10 * reward)
            # reward = 1.0 if ((last_cost - self.population.gbest) / last_cost) > 0.0 else 0.0
            # reward = np.sqrt(reward)
            # reward = -self.population.gbest / self.cost_scale_factor
            # reward = -np.max(self.population.cost) / self.cost_scale_factor
            self.q_r_history.append(np.concatenate((qvalue, [reward])))
            self.q_r_history = self.q_r_history[-self.qr_len:]
            self.Fevs = np.append(self.Fevs, Fevs)
            sample_size = self.sample_size if self.sample_size > 0 else self.population.NP
            # print(self.FEs, self.FEs + self.sample_times * sample_size)
            # print(self.Fevs.shape[0])

            if self.baseline:
                observe = None
            else:
                observe = self.observe()
                self.final_obs = observe
            while self.Fevs.shape[0] < self.MaxFEs // self.record_period and self.done:
                if self.Fevs.shape[0] > 1:
                    self.Fevs = np.append(self.Fevs, self.Fevs[-1])
                else:
                    self.Fevs = np.append(self.Fevs, 0.0)
            return observe, reward, self.done, {'info': Info(descent_seq=1 - np.array(self.Fevs) / self.cost_scale_factor,
                                                                    done=self.done,
                                                                    FEs=self.FEs,
                                                                    descent=1 - self.population.gbest / self.cost_scale_factor,
                                                                    best_cost=self.population.gbest,
                                                                    )
                                                       }  # next state, reward, is done, info
        else:
            return self.final_obs, -1, self.done, {'info': Info(descent_seq=1 - np.array(self.Fevs) / self.cost_scale_factor,
                                                               done=self.done,
                                                               FEs=self.FEs,
                                                               descent=1 - self.population.gbest / self.cost_scale_factor,
                                                               best_cost=self.population.gbest,
                                                               )
                                                  }  # next state, reward, is done, info


class random_optimizer:
    def __init__(self, dim):
        self.dim = dim

    # an uniform interface for testing
    def test_run(self,
                 problem,  # the problem instance to be optimize
                 seed,  # the random seed for running to ensure fairness
                 MaxFEs  # the max number of evaluations
                 ):
        np.random.seed(seed)
        # initialize population and optimizers
        population = Population(self.dim)
        population.initialize_costs(problem)
        factor = population.gbest
        k = 0
        Fevs = np.array([])
        while population.NP >= int(np.power(self.dim, k / 5 - 3) * MaxFEs):
            Fevs = np.append(Fevs, np.min(population.cost[:int(np.power(self.dim, k / 5 - 3) * MaxFEs)]))
            k += 1
        jde = JDE21(self.dim)
        shade = NL_SHADE_RSP(self.dim)
        prob = 0.5
        period = 5000
        end_fes = 0
        for i in range(population.NP, MaxFEs, period):
            # randomly select an optimizer
            if np.random.random() < prob:
                population, fes, end_fes = jde.step(population, problem, i, min(i + 5000, MaxFEs), MaxFEs)
            else:
                population, fes, end_fes = shade.step(population, problem, i, min(i + 5000, MaxFEs), MaxFEs)
            Fevs = np.append(Fevs, fes)
            if population.gbest < 1e-8:
                break
        while Fevs.shape[0] < 16:
            Fevs = np.append(Fevs, 0.0)
        return 1 - Fevs[-1] / factor, end_fes


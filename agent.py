import torch
import numpy as np
import collections
import copy
import random
import torch.nn.functional as F
from torch.distributions import Categorical, Distribution
from utils.utils import *
import os


class DQN:
    def __init__(self,
                 net,
                 optim,
                 mem_size,
                 replay_size,
                 gamma=0.99,
                 n_steps=10,
                 device='cpu',
                 ):
        self.device = device
        self.predict_net = net.to(self.device)
        # self.init_parameters()
        self.optimizer = optim
        # self.memory = collections.deque(maxlen=mem_size)
        self.memory = ReplayBuffer(maxlen=mem_size)
        self.replay_size = replay_size
        self.gamma = gamma
        self.n_steps = n_steps

    def init_parameters(self):
        for param in self.predict_net.parameters():
            stdv = 1. / np.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def save(self, log_path, epoch, run_time):
        torch.save(self.predict_net.state_dict(), os.path.join(log_path, 'DQN-' + run_time + f'-{epoch}.pth'))

    def load(self, path):
        models = torch.load(path)
        self.predict_net.load_state_dict(models['predict_net'])

    def update_memory(self, transitions):
        for transition in transitions:
            self.memory.append(transition)

    def predict(self,x):
        with torch.no_grad():
            return self.predict_net(x)

    def predict_q_value(self, x, test=False):
        with torch.no_grad():
            return self.predict_net(x, test)

    def epsilon_greedy_policy(self, qvalues, epsilon):
        actions = []
        for i in range(qvalues.shape[0]):
            if np.random.rand() < epsilon:
                actions.append(np.random.randint(0, qvalues.shape[1]))
            else:
                actions.append(torch.argmax(qvalues[i]).item())
        return actions

    def softmax_explore_policy(self, qvalues):
        values = 10 * qvalues
        values = np.exp(values)

        actions = []
        for i in range(qvalues.shape[0]):
            s = 0.0
            for j in range(values.shape[1]):
                s += values[i][j]
            actions.append(np.random.choice(np.arange(qvalues.shape[1]), p=values[i]/s))
        return actions

    def replay(self):
        if self.memory.size() >= self.replay_size:
            # transitions = random.sample(self.memory,self.replay_size)
            # Xs = np.array([x[0] for x in transitions])
            # q_vals = self.predict_q_value(Xs)
            # next_Xs = np.array([x[2] for x in transitions])
            # action = [x[1] for x in transitions]
            # action = torch.LongTensor(action).unsqueeze(1).to(self.device)
            # reward = [x[3] for x in transitions]
            # reward = torch.Tensor(reward).to(self.device)
            # is_done = [x[4] for x in transitions]
            # is_done = torch.BoolTensor(is_done).to(self.device)
            Xs, action, next_Xs, reward, is_done = self.memory.n_step_replay(self.replay_size)
            q_vals = self.predict_q_value(Xs).cpu()

            next_q_vals = self.predict_q_value(next_Xs).cpu()
            q_vals_action = torch.where(torch.tensor(is_done), torch.tensor(reward), torch.tensor(reward) + self.gamma * torch.max(next_q_vals, dim=-1).values)

            q_vals.scatter_(-1, torch.tensor(action, dtype=torch.int64).unsqueeze(1), torch.tensor(q_vals_action.unsqueeze(1), dtype=q_vals.dtype))
            q_val_pred = self.forward(Xs)
            qloss = F.mse_loss(q_val_pred, q_vals.to(self.device)).requires_grad_()
            return q_vals, qloss

        return torch.tensor([0.], requires_grad=True).to(self.device), torch.tensor(0., requires_grad=True).to(self.device)

    def forward(self, x):
        return self.predict_net(x)

    def learning(self):
        self.optimizer.zero_grad()
        # qvals, q_loss = self.replay()
        qvals, q_loss = self.n_step_replay()
        total_loss = q_loss
        total_loss.backward()
        self.optimizer.step()
        return qvals.mean().cpu().item(), q_loss.cpu().item(),

    def n_step_replay(self):
        if self.memory.size() >= self.replay_size:
            n_step = self.n_steps
            states, actions, next_states, rewards, dones = self.memory.n_step_replay(self.replay_size, n_step)
            gammas = np.ones(n_step + 1)
            for i in range(1, n_step + 1):
                gammas[i] = gammas[i-1] * self.gamma
            q_vals = self.predict_q_value(states).cpu()
            next_q_vals = torch.max(self.predict_q_value(next_states), -1).values.cpu()
            returns = np.sum(rewards * gammas[:n_step], -1)
            target_q = np.array(next_q_vals) * gammas[-1] + returns
            target_q[dones] = returns[dones]
            q_vals.scatter_(-1, torch.tensor(actions, dtype=torch.int64).unsqueeze(1), torch.tensor(target_q, dtype=q_vals.dtype).unsqueeze(1))
            q_val_pred = self.forward(states)
            qloss = F.mse_loss(q_val_pred, q_vals.to(self.device)).requires_grad_()
            return q_vals, qloss

        return torch.tensor([0.], requires_grad=True).to(self.device), torch.tensor(0., requires_grad=True).to(self.device)


class PPO:
    def __init__(self,
                 actor,
                 critic,
                 optim,
                 gamma=0.99,
                 eps_clip=0.1,
                 max_grad_norm=0.1,
                 device='cpu',
                 ):
        self.device = device
        self.actor = actor.to(self.device)
        self.actor_softmax = torch.nn.Softmax().to(self.device)
        self.critic = critic.to(self.device)
        # self.init_parameters()
        self.optimizer = optim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.max_grad_norm = max_grad_norm

    def init_parameters(self):
        for param in self.actor.parameters():
            stdv = 1. / np.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
        for param in self.critic.parameters():
            stdv = 1. / np.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def save(self, log_path, epoch, run_time):
        torch.save({
                    'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict(),
                },
                os.path.join(log_path, 'PPO-' + run_time + f'-{epoch}.pth'))

    def load(self, path):
        models = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(models['actor'])
        self.critic.load_state_dict(models['critic'])

    def actor_forward(self, x, test=False, to_critic=False):
        return self.actor(x, test, to_critic)

    def actor_forward_without_grad(self, x, test=False):
        with torch.no_grad():
            return self.actor_forward(x, test)

    def actor_sample(self, probs, fix_action=None):
        # probs = self.actor_softmax(probs)
        policy = Categorical(probs)
        if fix_action is None:
            actions = policy.sample()
        else:
            actions = fix_action
        select_probs = policy.log_prob(actions)
        return probs, select_probs, actions

    def critic_forward(self, x):
        bl_val = self.critic(x)
        baseline_val_detached = bl_val.detach()
        return baseline_val_detached, bl_val

    def critic_forward_without_grad(self, x):
        with torch.no_grad():
            return self.critic_forward(x)

    def learn(self, memory, k_epoch, logger, log_steps):
        length = len(memory['rewards'])
        old_states = memory['states']  # episode length * batch_size * state dim
        old_logprobs = []
        for tt in range(length):
            old_logprobs.append(memory['logprobs'][tt])
        old_logprobs = torch.cat(old_logprobs).view(-1)
        actions = memory['actions']
        old_value = None
        for k in range(k_epoch):
            if k == 0:
                logprobs = []
                bl_val_detached = []
                bl_val = []
                for tt in range(length):
                    logprobs.append(memory['logprobs'][tt])
                    bl_val_detached.append(memory['bl_val_detached'][tt])
                    bl_val.append(memory['bl_val'][tt])
            else:
                logprobs = []
                bl_val_detached = []
                bl_val = []
                for tt in range(length):
                    logits, feature = self.actor_forward(old_states[tt], to_critic=True)
                    _, batch_log_likelyhood, batch_action = self.actor_sample(logits, actions[tt])
                    logprobs.append(batch_log_likelyhood)
                    baseline_val_detached, baseline_val = self.critic_forward(feature)
                    bl_val_detached.append(baseline_val_detached)
                    bl_val.append(baseline_val)

            logprobs = torch.cat(logprobs).view(-1)
            bl_val_detached = torch.cat(bl_val_detached).view(-1)
            bl_val = torch.cat(bl_val).view(-1)
            Reward = []
            reward_reversed = memory['rewards'][::-1]

            R = self.critic_forward(self.actor_forward(old_states[-1], to_critic=True)[1])[0].view(-1)
            for r in range(len(reward_reversed)):
                R = R * self.gamma + torch.tensor(reward_reversed[r], dtype=torch.float32).to(self.device)
                Reward.append(R)
            # Reward = torch.stack(Reward[::-1], 0)  # n_step, bs
            # Reward = Reward.view(-1)
            Reward = torch.cat(Reward[::-1]).view(-1)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = Reward - bl_val_detached
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            reinforce_loss = -torch.min(surr1, surr2).mean()
            if old_value is None:
                baseline_loss = ((bl_val - Reward) ** 2).mean()
                old_value = bl_val.detach()
            else:
                vpredclipped = old_value + torch.clamp(bl_val - old_value, - self.eps_clip, self.eps_clip)
                v_max = torch.max(((bl_val - Reward) ** 2), ((vpredclipped - Reward) ** 2))
                baseline_loss = v_max.mean()

            approx_kl_divergence = (.5 * (old_logprobs.detach() - logprobs) ** 2).mean().detach()
            approx_kl_divergence[torch.isinf(approx_kl_divergence)] = 0

            loss = baseline_loss + reinforce_loss  # - 1e-5 * entropy.mean()
            self.optimizer.zero_grad()
            loss.backward()

            grad_norms = 0
            if self.max_grad_norm > 0:
                grad_norms = clip_grad_norms(self.optimizer.param_groups, self.max_grad_norm)
            self.optimizer.step()
            logger.write('train/Reward', log_steps, {'train/Reward': R.mean().cpu().item()})
            logger.write('train/ratios', log_steps, {'train/ratios': ratios.mean().cpu().item()})
            logger.write('train/baseline_loss', log_steps, {'train/baseline_loss': baseline_loss.cpu().item()})
            logger.write('train/reinforce_loss', log_steps, {'train/reinforce_loss': reinforce_loss.cpu().item()})
            logger.write('train/loss', log_steps, {'train/loss': loss.cpu().item()})
            logger.write('train/kl', log_steps, {'train/kl': approx_kl_divergence})
            if self.max_grad_norm > 0:
                grad_norms, grad_norms_clipped = grad_norms
                logger.write('train/actor_grad', log_steps, {'train/actor_grad': grad_norms[0]})
                logger.write('train/critic_grad', log_steps, {'train/critic_grad': grad_norms[1]})
            log_steps += 1


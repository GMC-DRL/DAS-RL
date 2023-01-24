import numpy as np
import torch


def mean_info(info, key):
    value = 0
    for i in range(len(info)):
        v = np.array(info[i]['info'].get()[key])
        value += info[i]['info'].get()[key]
    return value / len(info)


def plot_with_baseline(step, logger, ensemble, baselines):
    for i in range(ensemble.shape[0]):
        data = {'ensemble': ensemble[i]}
        for k, v in baselines.items():
            data[k] = v[i]
        logger.write_together(f'test/test{step}', i, data)


def to_transition(obs, act, obs_n, rew, done):
    bs = rew.shape[0]
    transitions = []
    for i in range(bs):
        if rew[i] >= 0:
            transitions.append((obs[i], act[i], obs_n[i], rew[i], done[i]))
    return transitions


def obs_max_min(obs):
    up = -1e15
    lo = 1e15
    unnormal = ()
    for ob in obs:
        for i in range(len(ob)):
            up = max(np.max(ob[i]), up)
            if up > 10:
                unnormal = (i, np.argmax(ob[i]))
                print(ob)
            lo = min(np.min(ob[i]), lo)
            if lo < -10:
                unnormal = (i, np.argmin(ob[i]))
                print(ob)
    return up, lo, unnormal


def log_obs(logger, obs, step):
    ob = obs[0]
    d = 0
    for item in ob:
        for value in item:
            logger.write(f'obs/obs{d}', step, {f'obs/obs{d}': value})
            d += 1


def clip_grad_norms(param_groups, max_norm=np.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm(
            group['params'],
            max_norm if max_norm > 0 else np.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for idx, group in enumerate(param_groups)
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


class ReplayBuffer:
    def __init__(self, maxlen=30000):
        self.state = np.array([], dtype=np.object)
        self.action = np.array([], dtype=np.int)
        self.next_state = np.array([], dtype=np.object)
        self.reward = np.array([])
        self.done = np.array([], dtype=np.bool)
        self.maxlen = maxlen

    def size(self):
        return self.state.shape[0]

    def append(self, obs, act, obs_n, rew, done):
        self.state = np.append(self.state, obs).reshape(-1, obs.shape[1])
        self.action = np.append(self.action, act)
        self.next_state = np.append(self.next_state, np.array(obs_n, dtype=np.object)).reshape(-1, obs_n.shape[1])
        self.reward = np.append(self.reward, rew)
        self.done = np.append(self.done, done)
        self.full_check()

    def full_check(self):
        if self.state.shape[0] > self.maxlen:
            self.state = self.state[-self.maxlen:]
            self.action = self.action[-self.maxlen:]
            self.next_state = self.next_state[-self.maxlen:]
            self.reward = self.reward[-self.maxlen:]
            self.done = self.done[-self.maxlen:]

    def n_step_replay(self, replay_size, n_step=1):
        index = np.random.choice(self.size(), size=replay_size, replace=False)
        states = self.state[index]
        actions = self.action[index]
        rewards = [self.reward[index]]
        done = self.done[index]
        for _ in range(n_step - 1):
            index = np.minimum(index + 1, self.size() - 1)
            reward = self.reward[index]
            reward[done] = 0
            rewards.append(reward)
            done = done + self.done[index]
        rewards = np.stack(rewards).transpose()
        next_state = self.next_state[index]
        done = done + self.done[index]
        if n_step == 1:
            rewards = rewards.squeeze()
        return states, actions, next_state, rewards, done

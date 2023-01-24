import time

import numpy as np
import tqdm
from utils.utils import *


def Q_train(agent,
            training_env,
            logger,
            epsilon,
            total_steps,
            max_steps,
            epoch,
            bid,
            b_num,
            ):
    def done_check(done):
        return np.sum(done) > 0

    obs = training_env.reset()
    has_done = False
    pbar = tqdm.tqdm(total=max_steps,
                     desc=f'training epoch {epoch} batch {bid + 1} / {b_num}',
                     bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    while not has_done:
        log_obs(logger, obs, total_steps)
        # up, lo, unnormal = obs_max_min(obs)
        # logger.write_together('train/obs', total_steps, {'upper': up, 'lower': lo})
        # if up > 10 or lo < -10:
        #     print(unnormal)
        #     exit(0)
        qvalues = agent.predict_q_value(obs)
        act = agent.epsilon_greedy_policy(qvalues, epsilon)
        actions = [{'action': act[i], 'qvalue': qvalues[i].cpu()} for i in range(training_env.env_num)]
        obs_next, rewards, is_done, info = training_env.step(actions)
        agent.update_memory(to_transition(obs, act, obs_next, rewards, is_done))
        returns, loss = agent.learning()
        obs = obs_next
        has_done = done_check(is_done)

        avg_q = qvalues.mean(0)
        logger.write('train/rewards', total_steps, {'train/rewards': rewards.mean()})
        logger.write('train/loss', total_steps, {'train/loss': loss})
        logger.write('train/returns', total_steps, {'train/returns': returns})
        act_count = [0] * avg_q.shape[0]
        for i in range(training_env.env_num):
            act_count[act[i]] += 1
        logger.write_together('train/action', total_steps, {f'action{i}': act_count[i] / training_env.env_num for i in range(len(act_count))})
        logger.write_together('train/qvalue', total_steps, {f'action{i}': avg_q[i] for i in range(avg_q.shape[0])})

        if has_done:
            logger.write('train/FEs', total_steps, {'train/FEs': mean_info(info, 'FEs')})
            logger.write('train/descent', total_steps, {'train/descent': mean_info(info, 'descent')})
        total_steps += 1
        pbar.update()
    pbar.close()
    return total_steps


def Q_train_v2(agent,
            training_env,
            logger,
            epsilon,
            total_steps,
            max_steps,
            epoch,
            bid,
            b_num,
            ):
    def done_check(done):
        return np.sum(done) >= done.shape[0]

    obs = training_env.reset()
    all_done = False
    pbar = tqdm.tqdm(total=max_steps,
                     desc=f'training epoch {epoch} batch {bid + 1} / {b_num}',
                     bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    bs = training_env.env_num
    obs_rec = []
    rew_rec = []
    act_rec = []
    done_rec = []
    step_count = np.zeros(bs, dtype=np.int32)
    while not all_done:
        # log_obs(logger, obs, total_steps)
        qvalues = agent.predict_q_value(obs)
        act = agent.epsilon_greedy_policy(qvalues, epsilon)
        actions = [{'action': act[i], 'qvalue': qvalues[i].cpu()} for i in range(training_env.env_num)]
        obs_next, rewards, is_done, info = training_env.step(actions)
        obs_rec.append(obs)
        act_rec.append(np.array(act))
        rew_rec.append(rewards)
        done_rec.append(is_done)
        step_count[is_done == False] += 1
        returns, loss = agent.learning()
        obs = obs_next
        all_done = done_check(is_done)

        avg_q = qvalues.mean(0)
        logger.write('train/rewards', total_steps, {'train/rewards': rewards.mean()})
        logger.write('train/loss', total_steps, {'train/loss': loss})
        logger.write('train/returns', total_steps, {'train/returns': returns})
        act_count = [0] * avg_q.shape[0]
        for i in range(training_env.env_num):
            act_count[act[i]] += 1
        logger.write_together('train/action', total_steps, {f'action{i}': act_count[i] / training_env.env_num for i in range(len(act_count))})
        logger.write_together('train/qvalue', total_steps, {f'action{i}': avg_q[i] for i in range(avg_q.shape[0])})

        if all_done:
            logger.write('train/FEs', total_steps, {'train/FEs': mean_info(info, 'FEs')})
            logger.write('train/descent', total_steps, {'train/descent': mean_info(info, 'descent')})
            obs_rec.append(obs)

            step_count += 1
            award = max_steps / step_count
            rew_rec = np.array(rew_rec, dtype=np.float32)
            award = np.array(award, dtype=np.float32)
            # for i in range(bs):
            #     rew_rec[:, i] *= info[i]['info'].get()['descent']
            for i in range(np.max(step_count)):
                rew_rec[i] *= award

                # agent.update_memory(to_transition(obs_rec[i], act_rec[i], obs_rec[i + 1], rew_rec[i], done_rec[i]))
                obs, act, next_obs, rew, done = obs_rec[i], act_rec[i], obs_rec[i + 1], rew_rec[i], done_rec[i]
                agent.memory.append(obs[rew >= 0], act[rew >= 0], next_obs[rew >= 0], rew[rew >= 0], done[rew >= 0])
        total_steps += 1
        pbar.update()
    pbar.close()

    return total_steps


def Q_test(agent,
           testing_env,
           test_steps,
           test_repeat,
           max_steps,
           bid,
           b_num,
           ):
    def done_check(done):
        return np.sum(done) >= done.shape[0]

    avg_descent = []
    avg_FEs = 0
    pbar = tqdm.tqdm(total=max_steps * test_repeat,
                     desc=f'testing {test_steps} batch {bid + 1} / {b_num}',
                     bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    feature_label = [[], [], []]
    act_rew = np.zeros(3)
    act_count = np.zeros(3)

    for repeat in range(test_repeat):
        obs = testing_env.reset()
        all_done = False
        info = None
        while not all_done:
            feature, qvalues = agent.predict_q_value(obs, test=True)

            act = torch.argmax(qvalues, -1).cpu()

            # act = agent.epsilon_greedy_policy(qvalues, 0.3)
            actions = [{'action': act[i], 'qvalue': qvalues[i].cpu()} for i in range(testing_env.env_num)]
            obs_next, reward, is_done, info = testing_env.step(actions)
            obs = obs_next
            all_done = done_check(is_done)
            pbar.update()
            for i in range(testing_env.env_num):
                feature_label[act[i]].append(feature[i])
                act_rew[act[i]] += reward[i]
                act_count[act[i]] += 1
        avg_descent.append(mean_info(info, 'descent_seq'))
        avg_FEs += mean_info(info, 'FEs')
    avg_descent = np.array(avg_descent)
    pbar.close()
    out = np.mean(avg_descent, 0), avg_FEs / test_repeat, feature_label, act_count
    return out


def Policy_train(agent,
                training_env,
                logger,
                total_steps,
                train_steps,
                k_epoch,
                max_steps,
                epoch,
                bid,
                b_num,
                ):
    def done_check(done):
        return np.sum(done) >= done.shape[0]

    obs = training_env.reset()
    all_done = False
    memory = {'states': [],
              'logprobs': [],
              'actions': [],
              'rewards': [],
              'bl_val': [],
              'bl_val_detached': [],
              }
    memory['states'].append(obs)
    bs = training_env.env_num
    step_count = np.zeros(bs, dtype=np.int32)
    pbar = tqdm.tqdm(total=max_steps,
                     desc=f'training epoch {epoch} batch {bid + 1} / {b_num}',
                     bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    while not all_done:

        # log_obs(logger, obs, total_steps)
        probs, log_probs, act = agent.actor_sample(agent.actor_forward(obs))
        actions = [{'action': act[i].cpu(), 'qvalue': probs[i].detach().cpu()} for i in range(training_env.env_num)]
        baseline_val_detached, baseline_val = agent.critic_forward(obs)
        obs_next, rewards, is_done, info = training_env.step(actions)
        obs = obs_next
        all_done = done_check(is_done)
        memory['states'].append(obs)
        memory['logprobs'].append(log_probs)
        memory['actions'].append(act)
        memory['rewards'].append(np.array(rewards, dtype=np.float32))
        memory['bl_val'].append(baseline_val)
        memory['bl_val_detached'].append(baseline_val_detached)
        step_count[is_done == False] += 1

        avg_probs = probs.cpu().mean(0)
        act_count = np.zeros(probs.shape[1])
        for i in range(training_env.env_num):
            act_count[act[i]] += 1
        logger.write_together('train/action', total_steps, {f'action{i}': act_count[i] / training_env.env_num for i in range(len(act_count))})
        logger.write_together('train/probs', total_steps, {f'probs{i}': avg_probs[i].cpu() for i in range(avg_probs.shape[0])})

        if all_done:
            logger.write('train/FEs', total_steps, {'train/FEs': mean_info(info, 'FEs')})
            logger.write('train/descent', total_steps, {'train/descent': mean_info(info, 'descent')})
            step_count += 1
            memory['rewards'] = np.array(memory['rewards'])
            # for i in range(bs):
            #     memory['rewards'][:, i] *= info[i]['info'].get()['descent']
            award = max_steps / step_count
            #for i in range(len(memory['rewards'])):
                #memory['rewards'][i] *= award

        total_steps += 1
        pbar.update()
    pbar.close()

    agent.learn(memory, k_epoch, logger, train_steps)

    return total_steps

def Policy_train_v2(agent,
                training_env,
                logger,
                total_steps,
                train_steps,
                n_step,
                k_epoch,
                max_steps,
                epoch,
                bid,
                b_num,
                ):
    def done_check(done):
        return np.sum(done) > 0

    def init_memory():
        memory = {'states': [],
                  'logprobs': [],
                  'actions': [],
                  'rewards': [],
                  'bl_val': [],
                  'bl_val_detached': [],
                  }
        return memory

    obs = training_env.reset()
    all_done = False
    memory = init_memory()
    memory['states'].append(obs)
    bs = training_env.env_num
    step = 0
    pbar = tqdm.tqdm(total=max_steps,
                     desc=f'training epoch {epoch} batch {bid + 1} / {b_num}',
                     bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    while not all_done:

        # log_obs(logger, obs, total_steps)
        probs, log_probs, act = agent.actor_sample(agent.actor_forward(obs))
        actions = [{'action': act[i].cpu(), 'qvalue': probs[i].detach().cpu()} for i in range(bs)]
        baseline_val_detached, baseline_val = agent.critic_forward(obs)
        obs_next, rewards, is_done, info = training_env.step(actions)
        obs = obs_next
        all_done = done_check(is_done)
        memory['states'].append(obs)
        memory['logprobs'].append(log_probs)
        memory['actions'].append(act)
        memory['rewards'].append(np.array(rewards, dtype=np.float32))
        memory['bl_val'].append(baseline_val)
        memory['bl_val_detached'].append(baseline_val_detached)

        avg_probs = probs.cpu().mean(0)
        act_count = np.zeros(probs.shape[1])
        for i in range(bs):
            act_count[act[i]] += 1
        logger.write_together('train/action', total_steps, {f'action{i}': act_count[i] / bs for i in range(len(act_count))})
        logger.write_together('train/probs', total_steps, {f'probs{i}': avg_probs[i].cpu() for i in range(avg_probs.shape[0])})
        step += 1
        if step % n_step == 0 or all_done:
            agent.learn(memory, k_epoch, logger, train_steps)
            memory = init_memory()
            memory['states'].append(obs)
        if all_done:
            logger.write('train/FEs', total_steps, {'train/FEs': mean_info(info, 'FEs')})
            logger.write('train/descent', total_steps, {'train/descent': mean_info(info, 'descent')})
        total_steps += 1
        pbar.update()
    pbar.close()

    return total_steps

def Policy_train_v3(agent,
                training_env,
                logger,
                total_steps,
                train_steps,
                k_epoch,
                max_steps,
                epoch,
                bid,
                b_num,
                ):
    def done_check(done):
        return np.sum(done) >= done.shape[0]

    obs = training_env.reset()
    all_done = False
    memory = {'states': [],
              'logprobs': [],
              'actions': [],
              'rewards': [],
              'bl_val': [],
              'bl_val_detached': [],
              }
    memory['states'].append(obs)
    bs = training_env.env_num
    step_count = np.zeros(bs, dtype=np.int32)
    pbar = tqdm.tqdm(total=max_steps,
                     desc=f'training epoch {epoch} batch {bid + 1} / {b_num}',
                     bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    while not all_done:

        # log_obs(logger, obs, total_steps)
        logits, feature = agent.actor_forward(obs, to_critic=True)
        probs, log_probs, act = agent.actor_sample(logits)
        actions = [{'action': act[i].cpu(), 'qvalue': probs[i].detach().cpu()} for i in range(bs)]
        baseline_val_detached, baseline_val = agent.critic_forward(feature)
        obs_next, rewards, is_done, info = training_env.step(actions)
        obs = obs_next
        all_done = done_check(is_done)
        memory['states'].append(obs)
        memory['logprobs'].append(log_probs)
        memory['actions'].append(act)
        memory['rewards'].append(np.array(rewards, dtype=np.float32))
        memory['bl_val'].append(baseline_val)
        memory['bl_val_detached'].append(baseline_val_detached)
        step_count[is_done == False] += 1

        avg_probs = probs.cpu().mean(0)
        act_count = np.zeros(probs.shape[1])
        for i in range(training_env.env_num):
            act_count[act[i]] += 1
        logger.write_together('train/action', total_steps, {f'action{i}': act_count[i] / training_env.env_num for i in range(len(act_count))})
        logger.write_together('train/probs', total_steps, {f'probs{i}': avg_probs[i].cpu() for i in range(avg_probs.shape[0])})

        if all_done:
            logger.write('train/FEs', total_steps, {'train/FEs': mean_info(info, 'FEs')})
            logger.write('train/descent', total_steps, {'train/descent': mean_info(info, 'descent')})
            step_count += 1
            memory['rewards'] = np.array(memory['rewards'])
            # for i in range(bs):
            #     memory['rewards'][:, i] *= info[i]['info'].get()['descent']
            award = max_steps / step_count
            for i in range(len(memory['rewards'])):
                memory['rewards'][i] *= award

        total_steps += 1
        pbar.update()
    pbar.close()

    agent.learn(memory, k_epoch, logger, train_steps)

    return total_steps




def Policy_test(agent,
           testing_env,
           test_steps,
           test_repeat,
           max_steps,
           bid,
           b_num,
           ):
    def done_check(done):
        return np.sum(done) >= done.shape[0]

    avg_descent = []
    avg_FEs = 0
    pbar = tqdm.tqdm(total=max_steps * test_repeat,
                     desc=f'testing {test_steps} batch {bid + 1} / {b_num}',
                     bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    n_action = testing_env.action_space[0].n
    feature_label = [[] for _ in range(n_action)]
    act_count = np.zeros(n_action)

    for repeat in range(test_repeat):
        obs = testing_env.reset()
        all_done = False
        descent = []
        info = None
        while not all_done:
            feature, logits = agent.actor_forward_without_grad(obs, test=True)
            probs, log_probs, act = agent.actor_sample(logits)
            probs = probs.detach().cpu()
            actions = [{'action': act[i].cpu(), 'qvalue': probs[i]} for i in range(testing_env.env_num)]
            obs_next, rewards, is_done, info = testing_env.step(actions)
            obs = obs_next
            all_done = done_check(is_done)
            descent.append(mean_info(info, 'descent'))
            pbar.update()
            for i in range(testing_env.env_num):
                feature_label[act[i]].append(feature[i])
                act_count[act[i]] += 1
        avg_descent.append(mean_info(info, 'descent_seq'))
        avg_FEs += mean_info(info, 'FEs')
    avg_descent = np.array(avg_descent)
    pbar.close()
    return np.mean(avg_descent, 0), avg_FEs / test_repeat, feature_label, act_count


def baseline_test(baselines,
                  base_test_env,
                  test_repeat,
                  MaxFEs,
                  period,
                  bid,
                  b_num,
                  ):
    avg_bases = {}
    bases_FEs = {}
    avg_gbest = 0
    for i, baseline in enumerate(baselines):
        FEs = 0
        avg_descent = np.zeros(MaxFEs // period)

        pbar = tqdm.tqdm(total=test_repeat,
                         desc=f'Baseline algorithm {baseline.__class__.__name__} batch {bid + 1} / {b_num}',
                         bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        for repeat in range(test_repeat):
            base_test_env.reset()
            _, _, _, info = base_test_env.step([{'action': i, 'qvalue': [0, 0]} for _ in range(base_test_env.env_num)])
            avg_descent += mean_info(info, 'descent_seq')
            FEs += mean_info(info, 'FEs')
            avg_gbest += mean_info(info, 'best_cost')
            pbar.update()
        pbar.close()
        avg_bases[baseline.__class__.__name__] = avg_descent / test_repeat
        bases_FEs[baseline.__class__.__name__] = FEs / test_repeat
    return avg_bases, bases_FEs, avg_gbest / test_repeat / len(baselines)

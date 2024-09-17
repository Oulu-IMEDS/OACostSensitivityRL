import pickle
import random
import torch
import pandas as pd
import numpy as np
import os
import hydra
from collections import deque
from common.model import QNetwork_forPatient, QNetwork_forKnee, QNetwork_forPatient_NoImage
import matplotlib.pyplot as plt
from dqn_agent import Agent
from tqdm import tqdm
from custom_env import CostOASensitivityEnv
from data.create_dataframe import GenerateDataFrameOAI, create_df_KL_probs_from_semixup
from data.preprocess_data import preprocess
from torch.optim import Adam
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, \
    recall_score, average_precision_score, precision_score, mean_absolute_error, hamming_loss, pairwise, \
    accuracy_score
import cv2


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def train_loop(env, env_val, agent, n_episodes=5000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    final_dict = {}
    scores = []  # list containing scores from each episode
    scores_window = []  # last 100 scores
    cumulative_score = []
    cumulative_qol = []
    cumulative_screening_cost = []
    list_actions_train = []
    cumulative_score_val = []
    cumulative_screening_cost_val = []
    cumulative_qol_val = []
    list_hd_val = []
    eps = eps_start  # initialize epsilon
    best_score = -100000
    best_qol = -100000
    best_cost = 100000
    best_bacc = -1000
    epoch_screening_cost = 0
    epoch_qol = 0
    iter = 0
    for i_episode in tqdm(range(n_episodes)):
        state = env.reset()
        # state = state[0]
        agent.state_idx = 0
        score = 0
        for t in range(env.n_steps):
            action, probs  = agent.act(state, mode=env.cfg.policy.train, eps=eps)
            next_state, reward, done, qol = env.step(action)
            # print(f'ep {i_episode}, step {t}: action {action}, reward {reward}, fp {env.ref_tp}, probs {probs}')
            if action == 1:
                epoch_screening_cost += env.cfg.cost.hospital_cost
            elif action == 0:
                epoch_screening_cost += 0
            epoch_qol += qol
            agent.step(state, 0 , action, reward, next_state, 0 , done)
            state = next_state
            score += reward
            list_actions_train.append(action)
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        u = i_episode % env.cfg.update_every
        if u == 0:
            agent.update()

        if (i_episode + 1) % env.len == 0:
            iter += 1
            print(f'eps:{eps}')
            eps = max(eps_end, eps_decay * eps)  # eps decrease epoch
            print(f'\n Average QoL SF12 of train set: {epoch_qol/env.len}')
            print(f'\n Cumulative screening cost of train set: {epoch_screening_cost}')
            print(f'\n Cumulative score of train set: {np.mean(scores_window)}')
            action_in_no_train = {i: list_actions_train.count(i) for i in list_actions_train}
            print(f'TRAIN Number of action in type: {action_in_no_train}')

            if iter % env.cfg.eval_every == 0:
                val_score, seq_actions, seq_targets, list_actions_val, screening_cost_val, qol_val = eval_loop(env_val,
                                                                                                               agent)
                # agent.update()

                print(f'Cumulative QoL SF12 of validation set: {qol_val}')
                print(f'Cumulative screening cost of validation set: {screening_cost_val}')
                print(f'Cumulative score of val set: {val_score}')
                action_in_no_val = {i: list_actions_val.count(i) for i in list_actions_val}
                print(f'VAL Number of action in type: {action_in_no_val}')

                list_actions_val = []
                avg_bacc = 0
                for m in range(env_val.n_steps):
                    _ba = balanced_accuracy_score(np.array(seq_targets)[:, m], np.array(seq_actions)[:, m])
                    _pr = precision_score(np.array(seq_targets)[:, m], np.array(seq_actions)[:, m])
                    _rc = recall_score(np.array(seq_targets)[:, m], np.array(seq_actions)[:, m])
                    print(f'Balance accuracy estimate progression after {m + 1} year(s): {_ba}')
                    print(f'Precision score progression after {m + 1} year(s): {_pr}')
                    print(f'Recall score progression after {m + 1} year(s): {_rc}')
                    avg_bacc += _ba
                avg_bacc = avg_bacc / env_val.n_steps
                print(f'Average balanced accuracy: {avg_bacc}')
                hamming_distance = hamming_loss(seq_targets, seq_actions)
                print(f'Hamming distance: {hamming_distance}')
                l1 = np.mean(pairwise.paired_manhattan_distances(np.array(seq_actions), np.array(seq_targets)))
                print(f'Mean paired l1 distance: {l1}')

                env_val.close()

                if val_score > best_score:
                    print(
                        f'\n Improved score from {best_score} to {val_score}. Saved to model...')
                    torch.save(agent.qnetwork_local.state_dict(), f'checkpoint_score.pth')
                    best_score = val_score

                if screening_cost_val < best_cost:
                    print(
                        f'\n Improved cost from {best_cost} to {screening_cost_val}. Saved to model...')
                    torch.save(agent.qnetwork_local.state_dict(), f'checkpoint_cost.pth')
                    best_cost = screening_cost_val

                if qol_val > best_qol:
                    print(
                        f'\n Improved QoL from {best_qol} to {qol_val}. Saved to model...')
                    torch.save(agent.qnetwork_local.state_dict(), f'checkpoint_qol.pth')
                    best_qol = qol_val

                if avg_bacc > best_bacc:
                    print(
                        f'\n Improved BA from {best_bacc} to {avg_bacc}. Saved to model...')
                    torch.save(agent.qnetwork_local.state_dict(), f'checkpoint_bacc.pth')
                    best_bacc = avg_bacc

                if iter % env.cfg.save_model_every == 0:
                    if env.cfg.data_level == 'patient':
                        model_filename = f'checkpoint_at_{iter}_seed{env.cfg.seed}.pth'
                    elif env.cfg.data_level == 'knee':
                        model_filename = f'checkpoint_side{env.cfg.knee_side}_at_{iter}_seed{env.cfg.seed}.pth'
                    torch.save({
                        'model_state_dict': agent.qnetwork_local.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict()}, model_filename)

                cumulative_score_val.append(val_score)

                cumulative_qol_val.append(qol_val)
                cumulative_screening_cost_val.append(screening_cost_val)
                list_hd_val.append(hamming_distance)

            cumulative_score.append(np.mean(scores_window))
            cumulative_qol.append(epoch_qol)
            cumulative_screening_cost.append(epoch_screening_cost)



            list_actions_train = []
            scores_window = []
            epoch_screening_cost = 0
            epoch_qol = 0

    final_dict['train_score'] = cumulative_score
    final_dict['train_qol'] = cumulative_qol
    final_dict['train_cost'] = cumulative_screening_cost
    final_dict['val_score'] = cumulative_score_val
    final_dict['val_qol'] = cumulative_qol_val
    final_dict['val_cost'] = cumulative_screening_cost_val

    return final_dict


def eval_loop(env_val, agent):
    scores_window_val = []
    list_actions_val = []
    seq_actions = []
    seq_targets = []
    targets = []
    screening_cost = 0
    cumm_qol = 0
    env_val.idx = -1
    for j_episode in range(env_val.df.shape[0]):
        actions_j = []
        targets_j = []
        state_val = env_val.reset(mode='val')
        agent.state_idx = 0
        score_val = 0

        for j_t in range(env_val.n_steps):
            # print(agent.state_idx)
            action, action_probs = agent.act(state_val, mode=env_val.cfg.policy.val)
            actions_j.append(action)
            if j_t != env_val.t_p - 1 or env_val.t_p > env_val.n_steps:
                targets_j.append(0)
            else:
                targets_j.append(1)
            next_state_val, reward, done, qol = env_val.step(action)
            if action == 1:
                screening_cost += env_val.cfg.cost.hospital_cost
            elif action == 0:
                screening_cost += 0
            cumm_qol += qol
            # print(f'VAL ep {j_episode}, step {j_t}: action {action}, reward {reward}, fp {env_val.ref_tp}, JSW {cumm_qol}')
            state_val = next_state_val
            score_val += reward
            agent.state_idx += 1
            if done:
                if env_val.cfg.early_stop:
                    for a in range(env_val.n_steps-j_t-1):
                        targets_j.append(0)
                        actions_j.append(0)
                break

        p = env_val.data.Progress
        t_p = [0] * 4
        if len(p) == 0:
            seq_targets.append(t_p)
        else:
            for j_t in p:
                t_p[j_t - 1] = 1
            seq_targets.append(t_p)

        seq_actions.append(actions_j)
        list_actions_val.extend(actions_j)
        scores_window_val.append(score_val)
    val_score = np.mean(scores_window_val)
    avg_medical_r = cumm_qol/env_val.len
    return val_score, seq_actions, seq_targets, list_actions_val, screening_cost, avg_medical_r


def convert_action_to_binary_seq(action, n_steps):
    n_segments = n_steps
    seq = [0] * n_segments
    if action[0] == 0:
        return seq
    else:
        loc = 0
        for idx, a in enumerate(action):
            loc = loc + a
            seq[loc - 1] = 1
        return seq

def convert_target_to_binary_seq(target, n_steps):
    n_segments = n_steps
    seq = [0] * n_segments
    if target == n_steps+1:
        return seq
    else:
        seq[target - 2] = 1
        return seq

@hydra.main(config_path=os.pardir, config_name="config.yaml")
def main(cfg):

    # Fixed Randomness
    pd.options.mode.chained_assignment = None
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if cfg.multi_progression:
        progression_filename = f'Multi_{cfg.data_filename}_verK'
    else:
        progression_filename = f'{cfg.data_filename}_verK'

    data_path = os.path.join(os.path.join(cfg.root_path,cfg.dataframe_path),f'{progression_filename}.pkl')
    if not os.path.exists(data_path):
        OAI_data = GenerateDataFrameOAI(cfg)
        if cfg.progress_feature == 'KL':
            if cfg.multi_progression:
                raw_data = OAI_data.multi_KL_progression()
            else:
                raw_data = OAI_data.KL_progression()
        elif cfg.progress_feature == 'JSW':
            if cfg.multi_progression:
                raw_data = OAI_data.multi_JSW_progression()
            else:
                raw_data = OAI_data.JSW_progression()
        elif cfg.progress_feature == 'KL-JSW' or cfg.progress_feature == 'JSW-KL':
            raw_data = OAI_data.KL_JSW_merge_dataframe(progression_filename)
        elif cfg.progress_feature == 'SF12':
            raw_data = OAI_data.SF12_progression()
        print(raw_data)
    else:
        raw_data = pd.read_pickle(data_path)


    n_steps = cfg.time_interval
    n_actions = 2
    n_epochs = cfg.n_epochs
    cfg.method = 'RL'

    test_site = cfg.test_site

    post_processed_data = preprocess(cfg, raw_data)

    env = CostOASensitivityEnv(cfg, post_processed_data, n_actions=n_actions, n_steps=n_steps, mode='train', test_site=test_site)
    env_val = CostOASensitivityEnv(cfg, post_processed_data, n_actions=n_actions, n_steps=n_steps, mode='test', test_site=test_site)

    # env.seed(0)
    print(f'Number of data: {env.df.shape[0]}')
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    print(f'Number of test data: {env_val.df.shape[0]}')
    n_features = env.observation_space.shape[0]

    lr = cfg.lr
    wd = cfg.wd

    if cfg.data_level == 'patient':
        if cfg.progressor_type == 'noImage':
            model = QNetwork_forPatient_NoImage(cfg, n_features, n_actions)
        else:
            model = QNetwork_forPatient(cfg, n_features, n_actions)
    if cfg.data_level == 'knee':
        model = QNetwork_forKnee(cfg, n_features, n_actions)

    model.to(cfg.device)
    optimizer = Adam(params=model.parameters(), lr=lr)
    agent = Agent(cfg, model, optimizer, state_size=n_features, action_size=n_actions)

    n_eps = n_epochs * env.df.shape[0]
    metrics = train_loop(env, env_val, agent, n_episodes=n_eps, max_t=100, eps_start=cfg.eps.eps_start, eps_end=cfg.eps.eps_end, eps_decay=cfg.eps.eps_decay)

    avg_score = metrics['train_score']
    avg_score_val = metrics['val_score']


    with open(f'metrics_seed{cfg.seed}.pkl', 'wb') as handle:
        pickle.dump(metrics, handle, protocol=4)

    # plot the scores
    fig = plt.figure()
    train = plt.scatter(np.arange(len(avg_score)), avg_score, s=3)
    val = plt.scatter(np.arange(len(avg_score_val)), avg_score_val, s=3)
    plt.legend((train, val),
               ('Train', 'Validation'),
               scatterpoints=1,
               loc='lower left',
               fontsize=8)
 
    plt.ylabel('Average score')
    plt.xlabel('Iteration #')
    plt.title(f'Randomly sample {cfg.sample_level}')
    plt.tight_layout()
    plt.savefig(f'Training_th{cfg.cost.JSW_cost_th}_cost{cfg.cost.hospital_cost}_scale{cfg.cost.cost_convert_r}_lr{cfg.lr}.png')
    plt.show()

if __name__ == "__main__":
    main()



import torch
import pandas as pd
import numpy as np
import os
import hydra
from collections import deque
import seaborn as sns
import random
import matplotlib.pyplot as plt
from common.model import QNetwork, QNetworkGRU, QNetwork_forKnee
from data.create_dataframe import GenerateDataFrameOAI
from dqn.run_agent import convert_target_to_binary_seq
from dqn_agent import Agent
from torch.optim import Adam
from data.preprocess_data import preprocess
from tqdm import tqdm
from custom_env import CostOASensitivityEnv
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, hamming_loss, pairwise,\
    recall_score, average_precision_score, precision_score, mean_absolute_error, confusion_matrix, accuracy_score


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
    # if 'JSW' not in cfg.data_filename:
    #     if 'MJSW250' in cfg.set_features:
    #         raise ValueError('Please change input features, JSW is not exist in this data')
    # if 'KL' not in cfg.data_filename:
    #     if 'KL' in cfg.set_features:
    #         raise ValueError('Please change input features, KL is not exist in this data')

    test_site = cfg.test_site
    cfg.data_level = 'patient'
    cfg.choose_oneside = False
    post_processed_data = preprocess(cfg, raw_data)

    # df_side_0 = post_processed_data[post_processed_data['SIDE'] == 0].reset_index(drop=True)
    # df_side_1 = post_processed_data[post_processed_data['SIDE'] == 1].reset_index(drop=True)

    # watch an untrained agent
    env_test = CostOASensitivityEnv(cfg, post_processed_data, n_actions=n_actions, n_steps=n_steps, mode='test', test_site=test_site)
    print(f'Number of test data: {env_test.df.shape[0]}')

    print(f'Number of data: {env_test.df.shape[0]}')
    print('State shape: ', env_test.observation_space.shape)
    print('Number of actions: ', env_test.action_space.n)
    headers = env_test.prefix_header.copy()
    headers_side0 = [r for r in headers if '_1' not in r]
    headers_side0 = sorted(headers_side0)
    headers_side0.extend(cfg.fix_features.copy())
    headers_side0.extend(['Info_step','Current_step'])

    headers_side1 = [r for r in headers if '_0' not in r]
    headers_side1 = sorted(headers_side1)
    headers_side1.extend(cfg.fix_features.copy())
    headers_side1.extend(['Info_step','Current_step'])
    n_features = len(headers_side0)

    lr = cfg.lr
    wd = cfg.wd
    cfg.data_level = 'knee'

    model_side0 = QNetwork_forKnee(cfg, n_features, n_actions)
    model_side1 = QNetwork_forKnee(cfg, n_features, n_actions)
    cfg.data_level = 'patient'
    model_side0.to(cfg.device)
    model_side1.to(cfg.device)
    metrics = {}
    score = []
    jsw_gained = []
    cost = []
    data = []
    # for i in tqdm(range(1000,1001,100)):
    for i in [18,54,68,96,119,153,186,207,325,363]:
        cfg.seed = i
        row_dict = {}
        row_dict['seed'] = i
        path = f'/home/khanhnguyen/workspace/data_from_exp/models/Knee_prob_sigmoid'
        model_path_side0 = os.path.join(f'{path}', f'checkpoint_side0_at_1000_seed{i}.pth')
        model_path_side1 = os.path.join(f'{path}', f'checkpoint_side1_at_1000_seed{i}.pth')
        # f'=/home/khanhnguyen/workspace/OACostSentivityRL/dqn/outputs/saved_models/2023-01-31/Pretrain_RNN_100epoch'
        checkpoint_side0 = torch.load(model_path_side0)
        checkpoint_side1 = torch.load(model_path_side1)
        model_side0.load_state_dict(checkpoint_side0['model_state_dict'])
        model_side1.load_state_dict(checkpoint_side1['model_state_dict'])
        optimizer_side0 = Adam(params=model_side0.parameters(), lr=lr, weight_decay=wd)
        optimizer_side0.load_state_dict(checkpoint_side0['optimizer_state_dict'])
        optimizer_side1 = Adam(params=model_side1.parameters(), lr=lr, weight_decay=wd)
        optimizer_side1.load_state_dict(checkpoint_side1['optimizer_state_dict'])

        agent_side0 = Agent(cfg, model_side0, optimizer_side0, state_size=n_features, action_size=n_actions)
        agent_side1 = Agent(cfg, model_side1, optimizer_side1, state_size=n_features, action_size=n_actions)
        scores_window_val = []
        list_actions_val = []
        seq_actions = []
        seq_targets = []
        seq_actions_probs = []
        list_targets = []
        screening_cost = 0
        cumm_qol = 0
        env_test.idx = -1
        for j_episode in range(env_test.df.shape[0]):
            actions_j = []
            actions_probs_j = []
            targets_j = []
            state = env_test.reset(mode='val')
            agent_side0.state_idx = 0
            agent_side1.state_idx = 0
            score_val = 0

            for j_t in range(env_test.n_steps):
                # print(agent.state_idx)
                state_side0 = state[headers_side0]
                state_side0.rename({'KL_0': 'KL', 'INJ_0': 'INJ', 'SURG_0': 'SURG', 'WOMAC_0': 'WOMAC'}, inplace=True)
                state_side1 = state[headers_side1]
                state_side1.rename({'KL_1': 'KL', 'INJ_1': 'INJ', 'SURG_1': 'SURG', 'WOMAC_1': 'WOMAC'}, inplace=True)
                action_side0, action_probs_side0 = agent_side0.act(state_side0, mode='test')
                action_side1, action_probs_side1 = agent_side1.act(state_side1, mode='test')
                if action_side0==1 or  action_side1 == 1:
                    action = 1
                else:
                    action = 0
                actions_j.append(action)
                action_probs = (action_probs_side0 + action_probs_side1)/2
                actions_probs_j.append(action_probs.squeeze().detach().numpy()[1])
                if j_t != env_test.t_p - 1 or env_test.t_p > env_test.n_steps:
                    targets_j.append(0)
                else:
                    targets_j.append(1)
                next_state, reward, done, qol = env_test.step(action)
                if action == 1:
                    screening_cost += env_test.cfg.cost.hospital_cost
                elif action == 0:
                    screening_cost += 0
                cumm_qol += qol
                # print(f'VAL ep {j_episode}, step {j_t}: action {action}, reward {reward}, fp {env_test.ref_tp}')
                state = next_state
                score_val += reward
                agent_side0.state_idx += 1
                agent_side1.state_idx += 1
                if done:
                    if env_test.cfg.early_stop:
                        for a in range(env_test.n_steps - j_t - 1):
                            targets_j.append(0)
                            actions_j.append(0)
                    break

            seq_targets.append(targets_j)
            seq_actions.append(actions_j)
            seq_actions_probs.append(actions_probs_j)
            list_actions_val.extend(actions_j)
            list_targets.extend(targets_j)
            scores_window_val.append(score_val)


        val_score = np.mean(scores_window_val)
        avg_medical_r = cumm_qol/env_test.len
        avg_cost = screening_cost/env_test.len

        score.append(val_score)
        row_dict['reward'] = val_score
        jsw_gained.append(avg_medical_r)
        row_dict['jsw_gained'] = avg_medical_r
        cost.append(avg_cost)
        row_dict['cost'] = avg_cost
        print(f'Benefit in money: {val_score}')
        print(f'QALY gained: {cumm_qol}')

        action_in_no_val = {i: list_actions_val.count(i) for i in list_actions_val}
        print(f'VAL Number of action in type: {action_in_no_val}')
        # print(f'Number of dismiss actions: {action_in_no_val[0]}')
        avg_bacc = 0
        avg_pr = 0
        avg_rc = 0
        avg_roc_auc = 0
        total_TN = 0
        total_FN = 0
        total_TP = 0
        total_FP = 0
        for m in range(env_test.n_steps):
            _ba = balanced_accuracy_score(np.array(seq_targets)[:, m], np.array(seq_actions)[:, m])
            _pr = precision_score(np.array(seq_targets)[:, m], np.array(seq_actions)[:, m])
            _rc = recall_score(np.array(seq_targets)[:, m], np.array(seq_actions)[:, m])
            _roc_auc = roc_auc_score(np.array(seq_targets)[:, m], np.array(seq_actions_probs)[:, m])
            cf_matrix = confusion_matrix(np.array(seq_targets)[:, m], np.array(seq_actions)[:, m])
            total_TN += cf_matrix[0][0]
            total_FN += cf_matrix[1][0]
            total_TP += cf_matrix[1][1]
            total_FP += cf_matrix[0][1]

            row_dict[f'BA@{m * 12}'] = _ba
            row_dict[f'Precision@{m * 12}'] = _pr
            row_dict[f'Recall@{m * 12}'] = _rc
            row_dict[f'ROC_AUC@{m * 12}'] = _roc_auc

            print(f'Balance accuracy estimate progression after {m + 1} year(s): {_ba}')
            print(f'Precision score progression after {m + 1} year(s): {_pr}')
            print(f'Recall score progression after {m + 1} year(s): {_rc}')
            print(f'ROC-AUC score progression after {m + 1} year(s): {_roc_auc}')
            avg_bacc += _ba
            avg_pr += _pr
            avg_rc += _rc
            avg_roc_auc += _roc_auc
        avg_bacc = avg_bacc / env_test.n_steps
        print(f'Average balanced accuracy: {avg_bacc}')
        row_dict['Average BA'] = avg_bacc
        avg_pr = avg_pr / env_test.n_steps
        print(f'Average precision: {avg_pr}')
        row_dict['Average PR'] = avg_pr
        avg_rc = avg_rc / env_test.n_steps
        print(f'Average recall: {avg_rc}')
        row_dict['Average RC'] = avg_rc
        avg_roc_auc = avg_roc_auc / env_test.n_steps
        print(f'Average ROC AUC: {avg_roc_auc}')
        row_dict['Average ROC AUC'] = avg_roc_auc
        hamming_distance = hamming_loss(seq_targets, seq_actions)
        print(f'Hamming distance: {hamming_distance}')
        row_dict['Hamming_distance'] = hamming_distance
        l1 = np.mean(pairwise.paired_manhattan_distances(np.array(seq_actions), np.array(seq_targets)))
        row_dict['l1'] = l1
        print(f'Mean paired l1 distance: {l1}')
        print(f'No of TN : {total_TN}')
        row_dict['TN'] = total_TN
        print(f'No of FN : {total_FN}')
        row_dict['FN'] = total_FN
        print(f'No of TP : {total_TP}')
        row_dict['TP'] = total_TP
        print(f'No of FP : {total_FP}')
        row_dict['FP'] = total_FP
        data.append(row_dict)

    metrics['val_score'] = score
    metrics['val_qol'] = jsw_gained
    metrics['val_cost'] = cost
    print(metrics['val_score'])
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(path,f'Knee-Level_states_ad{cfg.cost.true_dm_coef}.csv'), index=False)
    print(df)
    print(f'Reward: {df.reward.mean().round(2)} : {(df.reward.std()/np.sqrt(df.reward.size)).round(2)}')
    print(f'Cost: {df.cost.mean().round(2)} : {(df.cost.std()/np.sqrt(df.cost.size)).round(2)}')
    print(f'BA: {(df["Average BA"].mean()*100).round(2)} : {(df["Average BA"].std()/np.sqrt(df["Average BA"].size)*100).round(2)}')
    print(f'Recall: {(df["Average RC"].mean()*100).round(2)} : {((df["Average RC"]*100).std() / np.sqrt(df["Average RC"].size)).round(2)}')

if __name__ == "__main__":
    main()



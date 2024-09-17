import torch
import pandas as pd
import numpy as np
import os
import hydra
from collections import deque
import seaborn as sns
import random
import matplotlib.pyplot as plt
from common.model import QNetwork, QNetworkGRU, QNetwork_forPatient, QNetwork_forPatient_NoImage
from data.create_dataframe import GenerateDataFrameOAI
from data.preprocess_data import merge_personal_data_keep_side
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
    post_processed_data = preprocess(cfg, raw_data)

    # watch an untrained agent
    env_test = CostOASensitivityEnv(cfg, post_processed_data, n_actions=n_actions, n_steps=n_steps, mode='test', test_site=test_site)
    raw_data_test_site = raw_data[raw_data['Site']==f'{cfg.test_site}']
    if not os.path.exists(os.path.join(os.path.join(cfg.root_path,cfg.dataframe_path),f'Merge_patient_level_data_site{cfg.test_site}.pkl')):
        patient_level_raw_data = merge_personal_data_keep_side(cfg, raw_data_test_site, f'Merge_patient_level_data_site{cfg.test_site}')
    else:
        patient_level_raw_data = pd.read_pickle(os.path.join(os.path.join(cfg.root_path,cfg.dataframe_path),f'Merge_patient_level_data_site{cfg.test_site}.pkl'))
    patient_level_raw_data.drop(columns=['Progress'], inplace=True)
    print(f'Number of test data: {env_test.df.shape[0]}')

    print(f'Number of data: {env_test.df.shape[0]}')
    print('State shape: ', env_test.observation_space.shape)
    print('Number of actions: ', env_test.action_space.n)
    n_features = env_test.observation_space.shape[0]

    lr = cfg.lr
    wd = cfg.wd

    if cfg.data_level == 'patient':
        if cfg.progressor_type == 'noImage':
            model = QNetwork_forPatient_NoImage(cfg, n_features, n_actions)
        else:
            model = QNetwork_forPatient(cfg, n_features, n_actions)

    model.to(cfg.device)
    metrics = {}
    score = []
    jsw_gained = []
    cost = []
    data = []
    final_df_KL_visit = pd.DataFrame()
    # for i in tqdm(range(1000,1001,100)):
    # [18, 54, 68, 96, 119, 153, 186, 207, 325, 363]
    for i in [18, 54, 68, 96, 119, 153, 186, 207, 325, 363]:
        df_test = env_test.df
        cfg.seed = i
        row_dict = {}
        row_dict['seed'] = i
        path = cfg.model_test_path
        model_path = os.path.join(f'{path}', f'checkpoint_at_1000_seed{i}.pth')
        # f'=/home/khanhnguyen/workspace/OACostSentivityRL/dqn/outputs/saved_models/2023-01-31/Pretrain_RNN_100epoch'
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = Adam(params=model.parameters(), lr=lr, weight_decay=wd)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        agent = Agent(cfg, model, optimizer, state_size=n_features, action_size=n_actions)
        scores_window_val = []
        list_actions_val = []
        seq_actions = []
        sum_actions = []
        seq_targets = []
        num_progression = []
        seq_actions_probs = []
        list_targets = []
        screening_cost = 0
        cumm_qol = 0
        env_test.idx = -1
        for j_episode in range(env_test.df.shape[0]):
            actions_j = []
            actions_probs_j = []
            targets_j = []
            state_val = env_test.reset(mode='val')
            agent.state_idx = 0
            score_val = 0

            for j_t in range(env_test.n_steps):
                # print(agent.state_idx)
                action, action_probs = agent.act(state_val, mode='test')
                actions_j.append(action)
                actions_probs_j.append(action_probs.squeeze().detach().numpy()[1])
                if j_t != env_test.t_p - 1 or env_test.t_p > env_test.n_steps:
                    targets_j.append(0)
                else:
                    targets_j.append(1)
                next_state_val, reward, done, qol = env_test.step(action)
                if action == 1:
                    screening_cost += env_test.cfg.cost.hospital_cost
                elif action == 0:
                    screening_cost += 0
                cumm_qol += qol
                # print(f'VAL ep {j_episode}, step {j_t}: action {action}, reward {reward}, fp {env_test.ref_tp}')
                state_val = next_state_val
                score_val += reward
                agent.state_idx += 1
                if done:
                    if env_test.cfg.early_stop:
                        for a in range(env_test.n_steps - j_t - 1):
                            targets_j.append(0)
                            actions_j.append(0)
                    break
            p = env_test.data.Progress
            t_p = [0] * 4
            if len(p) == 0:
                seq_targets.append(t_p)
            else:
                for j_t in p:
                    t_p[j_t - 1] = 1
                seq_targets.append(t_p)
            # seq_targets.append(targets_j)
            if cfg.multi_progression:
                num_progression.append(len(env_test.data['Progress']))
            else:
                num_progression.append(1)
            seq_actions.append(actions_j)
            sum_actions.append(np.sum(actions_j))
            seq_actions_probs.append(actions_probs_j)
            list_actions_val.extend(actions_j)
            list_targets.extend(targets_j)
            scores_window_val.append(score_val)

        df_test['Actions'] = seq_actions
        df_test['TotalVisit'] = sum_actions
        df_test['Targets'] = seq_targets
        df_test['NoProgression'] = num_progression
        df_test_after_eval = df_test[['ID','Progress','Actions','Targets','TotalVisit','NoProgression']]
        seed_eval_df = pd.merge(patient_level_raw_data, df_test_after_eval, on='ID', how='inner')
        # seed_eval_df['TotalKL@0'] = sorted(seed_eval_df['KL_0@0'].values, seed_eval_df['KL_1@0'].values)
        seed_eval_df['TotalKL@0'] = [str(tuple(sorted((seed_eval_df['KL_0@0'].values[x].astype(int), seed_eval_df['KL_1@0'].values[x].astype(int))))) for x in range(0, len(seed_eval_df['KL_0@0'].values))]
        df_KL_visits_seed = seed_eval_df[['ID','Progress','Actions','Targets','TotalVisit','NoProgression','TotalKL@0']]
        df_KL_visits_seed['seed'] = i
        final_df_KL_visit = pd.concat([final_df_KL_visit, df_KL_visits_seed])

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
        print(f'Cost: {avg_cost}')

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
    df.to_csv(os.path.join(path,f'Patient_{cfg.test_site}_ad{cfg.cost.true_dm_coef}.csv'), index=False)
    print(df)
    print(f'Reward: {df.reward.mean().round(2)} : {(df.reward.std()/np.sqrt(df.reward.size)).round(2)}')
    print(f'Cost: {df.cost.mean().round(2)} : {(df.cost.std()/np.sqrt(df.cost.size)).round(2)}')
    print(f'BA: {(df["Average BA"].mean()*100).round(2)} : {(df["Average BA"].std()/np.sqrt(df["Average BA"].size)*100).round(2)}')
    print(f'Recall: {(df["Average RC"].mean()*100).round(2)} : {((df["Average RC"]*100).std() / np.sqrt(df["Average RC"].size)).round(2)}')
    print(f'Recall@0: {(df["Recall@0"].mean()*100).round(2)} : {(df["Recall@0"].std()/np.sqrt(df["Recall@0"].size)*100).round(2)}')
    print(f'Recall@12: {(df["Recall@12"].mean()*100).round(2)} : {(df["Recall@12"].std()/np.sqrt(df["Recall@12"].size)*100).round(2)}')
    print(f'Recall@24: {(df["Recall@24"].mean()*100).round(2)} : {(df["Recall@24"].std()/np.sqrt(df["Recall@24"].size)*100).round(2)}')
    print(f'Recall@36: {(df["Recall@36"].mean()*100).round(2)} : {(df["Recall@36"].std()/np.sqrt(df["Recall@36"].size)*100).round(2)}')
    print(f'TN: {df.TN.mean().round(2)} : {(df.TN.std()/np.sqrt(df.TN.size)).round(2)}')
    print(f'FN: {df.FN.mean().round(2)} : {(df.FN.std()/np.sqrt(df.FN.size)).round(2)}')
    print(f'TP: {df.TP.mean().round(2)} : {(df.TP.std()/np.sqrt(df.TP.size)).round(2)}')
    print(f'FP: {df.FP.mean().round(2)} : {(df.FP.std()/np.sqrt(df.FP.size)).round(2)}')

    final_df_KL_visit.to_pickle(os.path.join(path,f'TotalKL_TotalVisit.pkl'), protocol=4)

if __name__ == "__main__":
    main()



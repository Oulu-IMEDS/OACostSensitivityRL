import numpy as np
import torch
import pandas as pd
import hydra
import os
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss, BCEWithLogitsLoss
from torch.optim import Adam

from RNN.loader import DataFrameDatasetforRNN
from RNN.RNN_loop import RNNPipeline, RNNPipelinePredictProgressionBaselineInput
from Supervised_Learning.data_loader import DataFrameDataset
from common.model import QNetwork, LogisticRegression, QNetworkGRU, QNetworkGRU_forPretrain, GRU_predictProgression, \
    LSTM_predictProgression
from common.oai_utils import SIDE_FEATURES
from dqn.custom_env import CostOASensitivityEnv
from data.preprocess_data import preprocess, split_train_val_test
from torch.utils.data import DataLoader, WeightedRandomSampler
from data.create_dataframe import GenerateDataFrameOAI
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, confusion_matrix

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
    batch_size = cfg.batch_size
    n_epochs = cfg.n_epochs
    device = cfg.device
    n_steps = cfg.time_interval
    n_actions = 2
    current_time = 0
    target_time = 12
    cfg.method = 'SL'

    progression_filename = f'{cfg.data_filename}_verK'

    data_path = os.path.join(os.path.join(cfg.root_path, cfg.dataframe_path), f'{progression_filename}.pkl')
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
            raw_data = OAI_data.KL_JSW_merge_dataframe()
        elif cfg.progress_feature == 'SF12':
            raw_data = OAI_data.SF12_progression()
        print(raw_data)
    else:
        raw_data = pd.read_pickle(data_path)

    # unchanged_features_header = cfg.fix_features
    # side_features = set(SIDE_FEATURES) & set(cfg.set_features)
    # list_side_features = [x + '_0' for x in side_features] + [x + '_1' for x in side_features]
    # list_patient_features = list(set(cfg.set_features) - set(SIDE_FEATURES))
    # prefix_header = sorted(list_patient_features + list_side_features)
    #
    # features_headers = [s + f'@{current_time}' for s in prefix_header]
    # features_headers.extend(cfg.fix_features)

    post_processed_data = preprocess(cfg, raw_data)
    binary_progression_data = []
    for i, row in post_processed_data.iterrows():
        row_dict = {}
        row_dict['ID'] = row['ID']
        progress = row['Progress']
        # row_dict['Progress'] = row['Progress']
        if len(progress) == 0:
            row_dict['Progress@12'] = 0
            row_dict['Progress@24'] = 0
            row_dict['Progress@36'] = 0
            row_dict['Progress@48'] = 0
        else:
            encode_visit = [12, 24, 36, 48]
            p_index = 0
            for t in range(cfg.time_interval):
                if t == progress[p_index] - 1:
                    row_dict[f'Progress@{encode_visit[t]}'] = 1
                    if p_index == len(progress) - 1:
                        continue
                    else:
                        p_index += 1
                else:
                    row_dict[f'Progress@{encode_visit[t]}'] = 0
        binary_progression_data.append(row_dict)
    binary_progress_df = pd.DataFrame(binary_progression_data)
    post_processed_data = pd.merge(post_processed_data, binary_progress_df, on=['ID'], how='inner')
    # post_processed_data = post_processed_data.drop(columns=['Progress'])


    _, _, df_test = split_train_val_test(cfg, post_processed_data, f'Train_val_indx_seed{cfg.seed}.pkl')



    feature_headers = [t for t in df_test.columns if f'@0' in t]
    target_features = [f for f in feature_headers if f'{cfg.reward_feature[0]}' in f]
    state_size = len(feature_headers) + len(cfg.fix_features) - len(target_features) + 2
    print(state_size)

    test_ds = DataFrameDatasetforRNN(cfg, '', df_test)

    test_loader = torch.utils.data.DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, num_workers=4,drop_last=False)

    lr = 1e-3
    wd = cfg.wd
    # model = LogisticRegression(state_size, 1)
    # model = GRU_predictProgression(cfg, state_size, action_size=n_actions)
    model = LSTM_predictProgression(cfg, state_size, action_size=n_actions)
    optimizer = Adam(params=model.parameters(), lr=lr, weight_decay=wd)
    # model_path = f'/home/khanhnguyen/workspace/OACostSentivityRL/RNN/outputs/saved_models/2023-07-06/Pretrain_RNN/pretrain_GRU_best_ba.pth'
    # model_path = f'/home/khanhnguyen/workspace/OACostSentivityRL/outputs/saved_models/2023-04-14/RNN/pretrain_GRU_best_ba.pth'
    # model_path = f'/home/khanhnguyen/workspace/OACostSentivityRL/RNN/outputs/saved_models/2023-05-22/Pretrain_RNN/pretrain_LSTM_best_ba.pth'
    # model_path = f'/home/khanhnguyen/workspace/OACostSentivityRL/RNN/outputs/saved_models/2023-09-11/Pretrain_RNN/pretrain_LSTM_best_ba.pth'
    model_path = f'/home/khanhnguyen/workspace/OACostSentivityRL/RNN/outputs/saved_models/2023-09-12/Pretrain_RNN/pretrain_GRU_best_ba.pth'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    

    model.to(device)
    loss_func = {}
    loss_func['MSE'] = MSELoss()
    loss_func['CE'] = CrossEntropyLoss()
    loss_func['BCE'] = BCELoss()
    loss_func['BCEWithLogits'] = BCEWithLogitsLoss()
    pipeline = RNNPipelinePredictProgressionBaselineInput(cfg, model,loss_func,optimizer,device)

    action_seq = pipeline.test_loop(test_loader)
    encode_visit = [12,24,36,48]
    for i in range(cfg.time_interval):
        df_test[f'Action@{encode_visit[i]}'] = action_seq[i]
    print(df_test)

    env_test = CostOASensitivityEnv(cfg, df_test, n_actions=n_actions, n_steps=n_steps, mode='test-action',
                                    test_site=cfg.test_site)
    scores_window_val = []
    seq_targets = []
    seq_actions = []
    data = []
    screening_cost = 0
    cumm_qol = 0
    env_test.idx = -1
    for j_episode in range(env_test.df.shape[0]):
        actions_j = []
        actions_probs_j = []
        targets_j = []
        state_val = env_test.reset(mode='val')
        score_val = 0

        for j_t in range(env_test.n_steps):
            # print(agent.state_idx)
            action = env_test.data[f'Action@{encode_visit[j_t]}']
            actions_j.append(action)
            if j_t != env_test.t_p - 1 or env_test.t_p > env_test.n_steps:
                targets_j.append(0)
            else:
                targets_j.append(1)
            _, reward, done, qol = env_test.step(action)
            if action == 1:
                screening_cost += env_test.cfg.cost.hospital_cost
            elif action == 0:
                screening_cost += 0
            cumm_qol += qol
            # print(f'VAL ep {j_episode}, step {j_t}: action {action}, reward {reward}, fp {env_test.ref_tp}')
            score_val += reward
        
        p = env_test.data.Progress
        t_p = [0] * 4
        if len(p) == 0:
            seq_targets.append(t_p)
        else:
            for j_t in p:
                t_p[j_t - 1] = 1
            seq_targets.append(t_p)
        seq_actions.append(actions_j)

        scores_window_val.append(score_val)

    val_score = np.mean(scores_window_val)
    avg_medical_r = cumm_qol / env_test.len
    avg_cost = screening_cost / env_test.len

    row_dict['reward'] = val_score
    row_dict['jsw_gained'] = avg_medical_r
    row_dict['cost'] = avg_cost
    print(f'Benefit in money: {val_score}')
    print(f'QALY gained: {cumm_qol}')
    print(f'Cost: {avg_cost}')

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
        cf_matrix = confusion_matrix(np.array(seq_targets)[:, m], np.array(seq_actions)[:, m])
        print(cf_matrix)
        total_TN += cf_matrix[0][0]
        total_FN += cf_matrix[1][0]
        total_TP += cf_matrix[1][1]
        total_FP += cf_matrix[0][1]

        row_dict[f'BA@{m * 12}'] = _ba
        row_dict[f'Precision@{m * 12}'] = _pr
        row_dict[f'Recall@{m * 12}'] = _rc

        print(f'Balance accuracy estimate progression after {m + 1} year(s): {_ba}')
        print(f'Precision score progression after {m + 1} year(s): {_pr}')
        print(f'Recall score progression after {m + 1} year(s): {_rc}')
        # print(f'ROC-AUC score progression after {m + 1} year(s): {_roc_auc}')
        avg_bacc += _ba
        avg_pr += _pr
        avg_rc += _rc
        # avg_roc_auc += _roc_auc
    avg_bacc = avg_bacc / env_test.n_steps
    print(f'Average balanced accuracy: {avg_bacc}')
    row_dict['Average BA'] = avg_bacc
    avg_pr = avg_pr / env_test.n_steps
    print(f'Average precision: {avg_pr}')
    row_dict['Average PR'] = avg_pr
    avg_rc = avg_rc / env_test.n_steps
    print(f'Average recall: {avg_rc}')
    row_dict['Average RC'] = avg_rc
    print(f'No of TN : {total_TN}')
    row_dict['TN'] = total_TN
    print(f'No of FN : {total_FN}')
    row_dict['FN'] = total_FN
    print(f'No of TP : {total_TP}')
    row_dict['TP'] = total_TP
    print(f'No of FP : {total_FP}')
    row_dict['FP'] = total_FP
    data.append(row_dict)

    df = pd.DataFrame(data)
    print(df)

    # main_process(model,
    #              train_loader,
    #              eval_loader,
    #              device=device,
    #              n_epochs=n_epochs)
    # pretrain_model_path = os.getcwd()
    # model_filename = f'oai_best_roc_auc_{cfg.set_features}{cfg.fix_features}.pth'
    # model_path = os.path.join(pretrain_model_path, model_filename)
    #
    # checkpoint = torch.load(model_path)
    # pipeline.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    # optimizer = Adam(params=model.parameters(), lr=lr, weight_decay=wd)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])



if __name__ == "__main__":
    main()



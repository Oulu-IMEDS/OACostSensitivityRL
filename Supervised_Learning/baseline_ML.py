import torch
import pandas as pd
import hydra
import os
import numpy as np
from Supervised_Learning.data_loader import DataFrameDataset
from Supervised_Learning.utils import convert_predictions_to_cost
from data.preprocess_data import preprocess
from data.create_dataframe import GenerateDataFrameOAI
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, recall_score, precision_score, confusion_matrix, \
    accuracy_score, hamming_loss
import seaborn as sns
import matplotlib.pyplot as plt
from common.oai_utils import SIDE_FEATURES


@hydra.main(config_path=os.pardir, config_name="config.yaml")
def main(cfg):
    batch_size = cfg.batch_size
    n_epochs = cfg.n_epochs
    device = cfg.device
    test_site = cfg.test_site
    n_steps = cfg.time_interval
    n_actions = n_steps + 1
    seed = 18
    cfg.method = 'SL'

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

    encode_visit = ['0', '12', '24', '36', '48', '72', '96']
    post_processed_data = preprocess(cfg, raw_data)
    if cfg.progressor_type == 'prob':
        post_processed_data = convert_KL_prob_to_num(post_processed_data)
    df_train = post_processed_data[post_processed_data['Site'] != f'{cfg.test_site}']
    df_test = post_processed_data[post_processed_data['Site'] == f'{cfg.test_site}']
    df_train = convert_progression_to_binary(df_train)
    df_test = convert_progression_to_binary(df_test)

    side_features = set(SIDE_FEATURES) & set(cfg.set_features) - set(cfg.reward_feature)
    list_side_features = [x + '_0' for x in side_features] + [x + '_1' for x in side_features]
    list_patient_features = list(set(cfg.set_features) - set(SIDE_FEATURES))

    prefix_header = sorted(list_patient_features + list_side_features)
    for i in range(cfg.time_interval):
        step_features = [x + f'@{encode_visit[i]}' for x in prefix_header]
        step_features.extend(cfg.fix_features)
        x = df_train[step_features].to_numpy()
        y = df_train[f'Progress@{encode_visit[i+1]}'].to_numpy()
        if i==0:
            X_train = x
            y_train = y
        else:
            X_train = np.concatenate([X_train, x], axis=0)
            y_train = np.concatenate([y_train, y], axis=0)

    model = LogisticRegression(class_weight='balanced')
    # model = svm.SVC(class_weight='balanced')
    # model = LogisticRegression()

    # model.fit(X_train, y_train)

    avg_ba = 0
    avg_rc = 0
    avg_pr = 0
    total_TN = 0
    total_FN = 0
    total_TP = 0
    total_FP = 0
    for i in range(0, cfg.time_interval):
        step_features = [x + f'@{encode_visit[i]}' for x in prefix_header]
        step_features.extend(cfg.fix_features)
        X_train = df_train[step_features].to_numpy()
        y_train = df_train[f'Progress@{encode_visit[i+1]}'].to_numpy()
        model.fit(X_train, y_train)

        x_test = df_test[step_features].to_numpy()
        y_test = df_test[f'Progress@{encode_visit[i+1]}'].to_numpy()
        y_preds = model.predict(x_test)
        if i == 0:
            action_seq = y_preds
            target_seq = y_test
        else:
            action_seq = np.column_stack((action_seq, y_preds))
            target_seq = np.column_stack((target_seq, y_test))
        _ba = balanced_accuracy_score(y_test, y_preds)
        _rc = recall_score(y_test, y_preds)
        _pr = precision_score(y_test, y_preds)
        cf_matrix = confusion_matrix(y_test, y_preds)
        total_TN += cf_matrix[0][0]
        total_FN += cf_matrix[1][0]
        total_TP += cf_matrix[1][1]
        total_FP += cf_matrix[0][1]

        avg_ba += _ba
        avg_rc += _rc
        avg_pr += _pr
        print(f'Balanced Accuracy at  year {encode_visit[i]}: {_ba} ')
        print(f'Precision at  year {encode_visit[i]}: {_pr} ')
        print(f'Recall at  year {encode_visit[i]}: {_rc} ')

    hamming_distance = hamming_loss(target_seq, action_seq)
    print(f'Hamming distance: {hamming_distance}')
    print(f'Average BA: {avg_ba/cfg.time_interval} ')
    print(f'Average RC: {avg_rc / cfg.time_interval} ')
    print(f'Average PR: {avg_pr / cfg.time_interval} ')
    print(f'Total TN: {total_TN} ')
    print(f'Total FN: {total_FN} ')
    print(f'Total TP: {total_TP} ')
    print(f'Total FP: {total_FP} ')
    print(f'Average cost: {((total_FP+total_TP)*cfg.cost.hospital_cost)}')




def convert_progression_to_binary(df):
    encode_visit = ['0', '12', '24', '36', '48', '72', '96']
    data = []
    for i in range(df.shape[0]):
        row_data = df.iloc[i]
        row_dict = {}
        row_dict['ID'] = row_data['ID']
        row_dict['Progress@12'] = 0
        row_dict['Progress@24'] = 0
        row_dict['Progress@36'] = 0
        row_dict['Progress@48'] = 0
        if len(row_data['Progress']) == 0:
            pass
        elif len(row_data['Progress']) == 1:
            tp = row_data['Progress'][0]
            row_dict[f'Progress@{encode_visit[tp]}'] = 1
        elif len(df['Progress']) > 1:
            for j in range(len(row_data['Progress'])):
                tp = row_data['Progress'][j]
                row_dict[f'Progress@{encode_visit[tp]}'] = 1
        data.append(row_dict)
    new_df = pd.DataFrame(data)
    final_df = pd.merge(df, new_df, on='ID', how='inner')
    return final_df

def convert_KL_prob_to_num(df):
    encode_visit = ['0', '12', '24', '36', '48', '72', '96']
    data = []
    for i in range(df.shape[0]):
        row_data = df.iloc[i]
        row_dict = {}
        row_dict['ID'] = row_data['ID']
        for j in range(4):
            row_dict[f'KL_0@{encode_visit[j]}'] = np.argmax(row_data[f'KL_0@{encode_visit[j]}'])
            row_dict[f'KL_1@{encode_visit[j]}'] = np.argmax(row_data[f'KL_1@{encode_visit[j]}'])
        data.append(row_dict)
    new_df = pd.DataFrame(data)
    KL_cols = [x for x in df.columns if 'KL' in x]
    df.drop(columns=KL_cols, inplace=True)
    final_df = pd.merge(df, new_df, on='ID', how='inner')
    return final_df

def convert_preds_to_QoL(cfg, pred, label, current_qol, next_qol):
    qol = []
    cost = []
    defined_cost = (1/cfg.cost_coef)*cfg.screening_cost
    TN = 0
    FN = 0
    FP = 0
    TP = 0
    for i in range(len(pred)):
        action = pred[i]
        if action == 0:
            if label[i] != 0:
                r = -(max(0, current_qol[i] - next_qol[i]))
            else:
                r = 0
            score = r
            if score < 0:
                FN += 1
            else:
                TN += 1
            qol.append(r)
            cost.append(0)
        elif action == 1:
            r = (max(0, current_qol[i] - next_qol[i]))
            score = r - defined_cost
            if score < 0:
                FP += 1
            else:
                TP += 1
            qol.append(r)
            cost.append(cfg.screening_cost)
    count_preds = {'TP':TP,'TN':TN,'FP':FP,'FN':FN}
    return qol, cost, count_preds





if __name__ == "__main__":
    main()



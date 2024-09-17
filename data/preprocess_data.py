import os
import pickle

import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
from tqdm import tqdm
from common.oai_utils import *
from data.create_dataframe import merge_df_KL_preds_to_processed_data, create_image_path_dataframe


def preprocess(cfg, total_df):
    input_feature = cfg.input_feature
    n_steps = cfg.time_interval

    # progress_cols = [c for c in total_df.columns if 'Progress' in c]
    # if len(progress_cols) > 1:
    #     progress_var = cfg.progress_feature.split('-')[-1]
    #     for col in progress_cols:
    #         if progress_var in col:
    #             total_df.rename(columns={f'Progress_{progress_var}': 'Progress'}, inplace=True)
    #         else:
    #             continue

    dataframe_path = os.path.join(cfg.root_path, cfg.dataframe_path)
    if cfg.data_level == 'patient':
        if cfg.multi_progression:
            prename = f'Multi'
        else:
            prename = f''

        merge_data_fielname = f'{prename}{cfg.progress_feature}_clinical_vars_merged_personally'
        merge_data_path = os.path.join(dataframe_path,f'{merge_data_fielname}.pkl')
        if not os.path.exists(merge_data_path):
            raw_input_data = merge_personal_data_keep_side(cfg, total_df, merge_data_fielname)
        else:
            raw_input_data = pd.read_pickle(merge_data_path)
        cfg.choose_oneside = False
    elif cfg.data_level == 'knee':
        raw_input_data = total_df.rename(columns={"MJSW250@0": "JSW@0", "MJSW250@12": "JSW@12", "MJSW250@24": "JSW@24",
                                                  "MJSW250@36": "JSW@36", "MJSW250@48": "JSW@48", "MJSW250@72": "JSW@72",
                                                  "MJSW250@96": "JSW@96"})
        pass
        # if cfg.multi_progression:
        #     prename = f'Multi'
        # else:
        #     prename = f''
        #
        # merge_data_fielname = f'{prename}{cfg.progress_feature}_clinical_vars_merged_personally'
        # merge_data_path = os.path.join(dataframe_path,f'{merge_data_fielname}.pkl')
        # if not os.path.exists(merge_data_path):
        #     raw_input_data = merge_personal_data_keep_side(cfg, total_df, merge_data_fielname)
        # else:
        #     raw_input_data = pd.read_pickle(merge_data_path)

    if input_feature is not None:
        if input_feature == 'KL':
            raw_input_data = raw_input_data[(raw_input_data['KL_0@0'] >= cfg.KL_lower_th) & (raw_input_data['KL_0@0'] <= cfg.input_th)]
            raw_input_data = raw_input_data[(raw_input_data['KL_1@0'] >= cfg.KL_lower_th) & (raw_input_data['KL_1@0'] <= cfg.input_th)].reset_index(drop=True)
            common_part_filename = f'_Input{cfg.input_feature}_' \
                                   f'{cfg.KL_lower_th}-{cfg.input_th}_Target{cfg.progress_feature}' \
                                   f'_Time{cfg.time_interval}_site{cfg.test_site}'
        elif input_feature == 'JSW':
            if cfg.include_one_knee:
                raw_input_data = raw_input_data[(raw_input_data['JSW_0@0'] >= cfg.input_th) | (raw_input_data['JSW_0@0'] < 0)].reset_index(drop=True)
                raw_input_data = raw_input_data[(raw_input_data['JSW_1@0'] >= cfg.input_th) | (raw_input_data['JSW_1@0'] < 0)].reset_index(drop=True)
            else:
                if 'JSW_0@0' in raw_input_data.columns:
                    raw_input_data = raw_input_data[
                        (raw_input_data['JSW_0@0'] >= cfg.input_th) & (raw_input_data['JSW_1@0'] >= cfg.input_th)].reset_index(
                        drop=True)
                elif 'JSW@0' in raw_input_data.columns:
                    raw_input_data = raw_input_data[(raw_input_data['JSW@0'] >= cfg.input_th)].reset_index(drop=True)
            common_part_filename = f'_Input{cfg.input_feature}_{cfg.input_th}' \
                                   f'Target{cfg.progress_feature}' \
                                   f'_Time{cfg.time_interval}_site{cfg.test_site}'
        else:
            raise ValueError(f'Only support filter healthy for Kl or JSW')

    #Remove row with increasing KL, KL grade 4 or min medial JSW smaller than 1.6
    if cfg.multi_progression:
        pass
    else:
        raw_input_data = raw_input_data[raw_input_data['Progress'] >= 0]

    total_col_headers = raw_input_data.columns
    features = {'time_changed': cfg.set_features.copy(), 'unchanged': cfg.fix_features.copy()}
    selected_headers = select_col_headers(features, total_col_headers, n_steps, progressor_type=cfg.progressor_type)

    df_selected_features = raw_input_data[selected_headers]
    df_selected_features = df_selected_features.dropna().reset_index(drop=True)

    if cfg.data_level == 'patient':
        pass
    elif cfg.data_level == 'knee':
        list_ID_only_one_knee = [r for r in df_selected_features.ID.unique() if
                                 len(df_selected_features[df_selected_features['ID'] == r]) == 1]
        df_selected_features = df_selected_features[~df_selected_features['ID'].isin(list_ID_only_one_knee)].reset_index(drop=True)

    #mean normalize data
    if cfg.normalize == 'mean':
        df_selected_features, dict_features = mean_normalization(df_selected_features, features['time_changed'])
    elif cfg.normalize == 'min':
        df_selected_features, dict_features = min_normalization(df_selected_features, features['time_changed'])
    else:
        raise ValueError(f'Only support mean or min normalization')

    cfg.dict_features = dict_features
    with open(os.path.join(dataframe_path,'dict_features.pkl'), 'wb') as handle:
        pickle.dump(dict_features, handle, protocol=4)


    #modify target column
    df_selected_features = modify_progress_column(cfg, df_selected_features)


    if cfg.progressor_type in ['num','noImage']:
        KL_pred_ID = pd.read_pickle(os.path.join(dataframe_path, f'KL_predictions_ID.pkl'))
        df_selected_features = pd.merge(df_selected_features, KL_pred_ID, on='ID', how='inner')
        if not os.path.exists(os.path.join(dataframe_path,f'Post_Processing_{cfg.progress_feature}_{cfg.progressor_type}_{cfg.data_level}_level.pkl')):
            df_selected_features.to_csv(os.path.join(dataframe_path,f'Post_Processing_{cfg.progress_feature}_{cfg.progressor_type}_{cfg.data_level}_level.csv'), index=False)
            df_selected_features.to_pickle(os.path.join(dataframe_path, f'Post_Processing_{cfg.progress_feature}_{cfg.progressor_type}_{cfg.data_level}_level.pkl'), protocol=4)
    else:
        df_selected_features = merge_df_KL_preds_to_processed_data(cfg, df_selected_features)
        if not os.path.exists(os.path.join(dataframe_path, f'Post_Processing_{cfg.progress_feature}_with_KL_{cfg.progressor_type}_{cfg.data_level}_level.pkl')):
            df_selected_features.to_csv(os.path.join(dataframe_path, f'Post_Processing_{cfg.progress_feature}_with_KL_{cfg.progressor_type}_{cfg.data_level}_level.csv'),index=False)
            df_selected_features.to_pickle(os.path.join(dataframe_path, f'Post_Processing_{cfg.progress_feature}_with_KL_{cfg.progressor_type}_{cfg.data_level}_level.pkl'),protocol=4)


    # Choose one side leg
    if cfg.choose_oneside:
        df_selected_features = df_selected_features[df_selected_features['SIDE'] == cfg.knee_side].reset_index(drop=True)

    return df_selected_features

# def split_train_test_df(cfg, df):
#     # if cfg.include_one_knee:
#     #     ids_filename= f'inds_train_val_{common_part_filename}_seed{cfg.seed}_include_one_knee_only.pkl'
#     # else:
#     #     ids_filename = f'inds_train_val_{common_part_filename}_seed{cfg.seed}.pkl'
#     df_train, df_val, df_test = split_train_val_test(cfg, df_selected_features, ids_filename)
#
#
#     return df_train, df_val, df_test
def include_img_path(cfg, df_progression):
    df_img = create_image_path_dataframe(cfg)
    if cfg.data_level == 'patient':
        df = pd.merge(df_progression, df_img, on=['ID'], how='inner')
    elif cfg.data_level == 'knee':
        df = pd.merge(df_progression, df_img, on=['ID','SIDE'], how='inner')
    return df

def modify_progress_column(cfg, df):
    if cfg.multi_progression:
        for i in range(len(df)):
            progress = df.loc[i, 'Progress']
            if len(progress) == 0:
                continue
            else:
                new_prog = sorted([t for t in progress if t <= cfg.time_interval])
                df.at[i, "Progress"] = new_prog
    else:
        df['Progress'] = df['Progress'].astype(np.int32)
        if cfg.progression_within:
            df.loc[
                (df["Progress"] > 0) & (df["Progress"] <= cfg.time_interval), "Progress"] = 1

        # df_selected_features.drop(df_selected_features[df_selected_features.Progress > n_steps].index, inplace=True)

        df.loc[(df["Progress"] == 0) | (df["Progress"] > cfg.time_interval), "Progress"] = 0
        # df_selected_features.loc[(df_selected_features["Progress"] == 0) | (df_selected_features["Progress"] > n_steps), "Progress"] = n_steps
    return df

def merge_personal_data_keep_side(cfg, df, filename):
    personal_time_changed_features = ['AGE', 'BMI', 'KOOSQoL', 'PAS', 'SF12']
    personal_fixed_features = ['SEX']
    group_headers = ['ID', 'Site']
    cols = df.columns
    encode_visit = ['0', '12', '24', '36', '48', '72', '96']
    features_headers = []
    for f in personal_time_changed_features:
        personal_time_changed_features = [col for col in cols if f'{f}@' in col]
        features_headers.extend(personal_time_changed_features)
    features_headers.extend(personal_fixed_features)
    features_headers.extend(group_headers)

    total_df = df[features_headers].drop_duplicates(subset=['ID'])
    total_df.reset_index(drop=True, inplace=True)

    # merge progression for 2 side of knee
    list_ID_unique = df.ID.unique()
    data = []
    for i in tqdm(list_ID_unique, desc=f"Merging data from knee level into person level"):
        row_dict = {}
        row_data = df[df['ID'] == i].reset_index(drop=True)

        row_dict['ID'] = i
        if len(row_data) > 1:
            prog_side0 = row_data[row_data['SIDE'] == 0].iloc[0]['Progress']
            prog_side1 = row_data[row_data['SIDE'] == 1].iloc[0]['Progress']
            if cfg.multi_progression:
                both_side_prog = sorted(list(set(prog_side0 + prog_side1)))
                row_dict['Progress'] = both_side_prog
            else:
                if prog_side0 > 0 and prog_side1 > 0:
                    if prog_side0 <= prog_side1:
                        row_dict['Progress'] = prog_side0
                    else:
                        row_dict['Progress'] = prog_side1
                elif prog_side0 > 0 and prog_side1 == 0:
                    row_dict['Progress'] = prog_side0
                elif prog_side0 == 0 and prog_side1 > 0:
                    row_dict['Progress'] = prog_side1
                elif prog_side0 == 0 and prog_side1 == 0:
                    row_dict['Progress'] = 0

                if prog_side0 < 0 or prog_side1 < 0:
                    row_dict['Progress'] = -1

            for j in range(7):
                if 'KL@0' in row_data.columns:
                    kl_side0 = row_data[row_data['SIDE'] == 0].iloc[0][f'KL@{encode_visit[j]}']
                    kl_side1 = row_data[row_data['SIDE'] == 1].iloc[0][f'KL@{encode_visit[j]}']
                    row_dict[f'KL_0@{encode_visit[j]}'] = kl_side0
                    row_dict[f'KL_1@{encode_visit[j]}'] = kl_side1

                if 'MJSW250@0' in row_data.columns:
                    jsw_side0 = row_data[row_data['SIDE'] == 0].iloc[0][f'MJSW250@{encode_visit[j]}']
                    jsw_side1 = row_data[row_data['SIDE'] == 1].iloc[0][f'MJSW250@{encode_visit[j]}']
                    row_dict[f'JSW_0@{encode_visit[j]}'] = jsw_side0
                    row_dict[f'JSW_1@{encode_visit[j]}'] = jsw_side1

                if 'KL_Probs@0' in row_data.columns:
                    kl_prob_side0 = row_data[row_data['SIDE'] == 0].iloc[0][f'KL_Probs@{encode_visit[j]}']
                    kl_prob_side1 = row_data[row_data['SIDE'] == 1].iloc[0][f'KL_Probs@{encode_visit[j]}']
                    row_dict[f'KL_Prob_0@{encode_visit[j]}'] = kl_prob_side0
                    row_dict[f'KL_Prob_1@{encode_visit[j]}'] = kl_prob_side1

                if 'WOMAC@0' in row_data.columns:
                    womac_side0 = row_data[row_data['SIDE'] == 0].iloc[0][f'WOMAC@{encode_visit[j]}']
                    womac_side1 = row_data[row_data['SIDE'] == 1].iloc[0][f'WOMAC@{encode_visit[j]}']
                    row_dict[f'WOMAC_0@{encode_visit[j]}'] = womac_side0
                    row_dict[f'WOMAC_1@{encode_visit[j]}'] = womac_side1

                if 'INJ@0' in row_data.columns:
                    inj_side0 = row_data[row_data['SIDE'] == 0].iloc[0][f'INJ@{encode_visit[j]}']
                    inj_side1 = row_data[row_data['SIDE'] == 1].iloc[0][f'INJ@{encode_visit[j]}']
                    row_dict[f'INJ_0@{encode_visit[j]}'] = inj_side0
                    row_dict[f'INJ_1@{encode_visit[j]}'] = inj_side1

                if 'SURG@0' in row_data.columns:
                    surg_side0 = row_data[row_data['SIDE'] == 0].iloc[0][f'SURG@{encode_visit[j]}']
                    surg_side1 = row_data[row_data['SIDE'] == 1].iloc[0][f'SURG@{encode_visit[j]}']
                    row_dict[f'SURG_0@{encode_visit[j]}'] = surg_side0
                    row_dict[f'SURG_1@{encode_visit[j]}'] = surg_side1


        else:
            row_dict['Progress'] = row_data.iloc[0]['Progress']
            side = row_data.iloc[0]['SIDE']
            if side == 0:
                other_side = 1
            else:
                other_side = 0
            for j in range(7):
                if 'KL@0' in row_data.columns:
                    row_dict[f'KL_{side}@{encode_visit[j]}'] = row_data.iloc[0][f'KL@{encode_visit[j]}']
                    row_dict[f'KL_{other_side}@{encode_visit[j]}'] = -1

                if 'MJSW250@0' in row_data.columns:
                    row_dict[f'JSW_{side}@{encode_visit[j]}'] = row_data.iloc[0][f'MJSW250@{encode_visit[j]}']
                    row_dict[f'JSW_{other_side}@{encode_visit[j]}'] = -1


                if 'WOMAC@0' in row_data.columns:
                    row_dict[f'WOMAC_{side}@{encode_visit[j]}'] = row_data.iloc[0][f'WOMAC@{encode_visit[j]}']
                    row_dict[f'WOMAC_{other_side}@{encode_visit[j]}'] = -1

                if 'INJ@0' in row_data.columns:
                    row_dict[f'INJ_{side}@{encode_visit[j]}'] = row_data.iloc[0][f'INJ@{encode_visit[j]}']
                    row_dict[f'INJ_{other_side}@{encode_visit[j]}'] = -1

                if 'SURG@0' in row_data.columns:
                    row_dict[f'SURG_{side}@{encode_visit[j]}'] = row_data.iloc[0][f'SURG@{encode_visit[j]}']
                    row_dict[f'SURG_{other_side}@{encode_visit[j]}'] = -1

        data.append(row_dict)
    new_prog_df = pd.DataFrame(data=data)
    personal_prog_df = pd.merge(total_df, new_prog_df, on=['ID'], how='inner')
    save_path = os.path.join(cfg.root_path, cfg.dataframe_path)
    personal_prog_df.to_csv(os.path.join(save_path, f'{filename}.csv'), index=False)
    personal_prog_df.to_pickle(os.path.join(save_path, f'{filename}.pkl'), protocol=4)

    return personal_prog_df

def merge_personal_data(cfg, df):
    personal_time_changed_features = ['AGE', 'BMI', 'KOOSQoL', 'PAS', 'SF12']
    personal_fixed_features = ['SEX']
    group_headers = ['ID', 'Site']
    cols = df.columns
    encode_visit = ['0', '12', '24', '36', '48', '72', '96']
    features_headers = []
    for f in personal_time_changed_features:
        personal_time_changed_features = [col for col in cols if f'{f}@' in col]
        features_headers.extend(personal_time_changed_features)
    features_headers.extend(personal_fixed_features)
    features_headers.extend(group_headers)

    total_df = df[features_headers].drop_duplicates(subset=['ID'])
    total_df.reset_index(drop=True, inplace=True)

    # merge progression for 2 side of knee
    list_ID_unique = df.ID.unique()
    data = []
    for i in list_ID_unique:
        row_dict = {}
        row_data = df[df['ID'] == i].reset_index(drop=True)

        row_dict['ID'] = i
        if len(row_data) > 1:
            prog_side0 = row_data[row_data['SIDE'] == 0].iloc[0]['Progress']
            prog_side1 = row_data[row_data['SIDE'] == 1].iloc[0]['Progress']
            if prog_side0 > 0 and prog_side1 > 0:
                if prog_side0 <= prog_side1:
                    row_dict['Progress'] = prog_side0
                else:
                    row_dict['Progress'] = prog_side1
            elif prog_side0 > 0 and prog_side1 == 0:
                row_dict['Progress'] = prog_side0
            elif prog_side0 == 0 and prog_side1 > 0:
                row_dict['Progress'] = prog_side1
            elif prog_side0 == 0 and prog_side1 == 0:
                row_dict['Progress'] = 0

            if prog_side0 < 0 or prog_side1 < 0:
                row_dict['Progress'] = -1

            for j in range(7):
                kl_side0 = row_data[row_data['SIDE'] == 0].iloc[0][f'KL@{encode_visit[j]}']
                kl_side1 = row_data[row_data['SIDE'] == 1].iloc[0][f'KL@{encode_visit[j]}']
                if kl_side0 > kl_side1:
                    row_dict[f'KL@{encode_visit[j]}'] = kl_side0
                else:
                    row_dict[f'KL@{encode_visit[j]}'] = kl_side1
                # row_dict[f'KL@{encode_visit[j]}'] = kl_side0 + kl_side1

                womac_side0 = row_data[row_data['SIDE'] == 0].iloc[0][f'WOMAC@{encode_visit[j]}']
                womac_side1 = row_data[row_data['SIDE'] == 1].iloc[0][f'WOMAC@{encode_visit[j]}']
                # if womac_side0 > womac_side1:
                #     row_dict[f'WOMAC@{encode_visit[j]}'] = womac_side0
                # else:
                #     row_dict[f'WOMAC@{encode_visit[j]}'] = womac_side1
                row_dict[f'WOMAC@{encode_visit[j]}'] = womac_side0 + womac_side1

                inj_side0 = row_data[row_data['SIDE'] == 0].iloc[0][f'INJ@{encode_visit[j]}']
                inj_side1 = row_data[row_data['SIDE'] == 1].iloc[0][f'INJ@{encode_visit[j]}']
                if inj_side0 > inj_side1:
                    row_dict[f'INJ@{encode_visit[j]}'] = inj_side0
                else:
                    row_dict[f'INJ@{encode_visit[j]}'] = inj_side1
                # row_dict[f'INJ@{encode_visit[j]}'] = inj_side0 + inj_side1

                surg_side0 = row_data[row_data['SIDE'] == 0].iloc[0][f'SURG@{encode_visit[j]}']
                surg_side1 = row_data[row_data['SIDE'] == 1].iloc[0][f'SURG@{encode_visit[j]}']
                if surg_side0 > surg_side1:
                    row_dict[f'SURG@{encode_visit[j]}'] = surg_side0
                else:
                    row_dict[f'SURG@{encode_visit[j]}'] = surg_side1
                # row_dict[f'SURG@{encode_visit[j]}'] = surg_side0 + surg_side1


        else:
            row_dict['Progress'] = row_data.iloc[0]['Progress']
            for j in range(7):
                row_dict[f'KL@{encode_visit[j]}'] = row_data.iloc[0][f'KL@{encode_visit[j]}']
                row_dict[f'WOMAC@{encode_visit[j]}'] = row_data.iloc[0][f'WOMAC@{encode_visit[j]}']
                row_dict[f'INJ@{encode_visit[j]}'] = row_data.iloc[0][f'INJ@{encode_visit[j]}']
                row_dict[f'SURG@{encode_visit[j]}'] = row_data.iloc[0][f'SURG@{encode_visit[j]}']

        data.append(row_dict)
    new_prog_df = pd.DataFrame(data=data)
    personal_prog_df = pd.merge(total_df, new_prog_df, on=['ID'], how='inner')
    save_path = os.path.join(cfg.root_path, cfg.dataframe_path)
    data_filename = f'{cfg.progress_feature}_clinical_vars_merged_personally'
    personal_prog_df.to_csv(os.path.join(save_path, f'{data_filename}.csv'), index=False)
    personal_prog_df.to_pickle(os.path.join(save_path, f'{data_filename}.pkl'), protocol=4)

    return personal_prog_df

def select_col_headers(features, cols, n_steps, progressor_type='num'):
    time_features = features['time_changed']
    if progressor_type in ['prob','logit','feature']:
        time_features.remove('KL')
    elif progressor_type in ['num','noImage']:
        pass
    else:
        raise ValueError(f'Not support KL type {progressor_type}')
    unchanged_features = features['unchanged']
    encode_visit = ['0', '12', '24', '36', '48', '72', '96']
    features_headers = []
    for f in time_features:
        timeline_features = [col for col in cols if (f'{f}@' in col) or (f'{f}_' in col)]
        features_headers.extend(timeline_features)
    if 'minJSW@0' in features_headers:
        features_headers.remove('minJSW@0')
    progressed_header = ['Progress']

    selected_headers = []
    for i in range(n_steps + 1):
        follow_ups = f'@{encode_visit[i]}'
        selected_headers.extend([col for col in features_headers if follow_ups in col])
    selected_headers.extend(unchanged_features)
    selected_headers.extend(progressed_header)
    group_headers = ['Site', 'ID']
    if 'SIDE' in cols:
        group_headers.append('SIDE')
    selected_headers.extend(group_headers)

    return selected_headers


def mean_normalization(df, set_used_features):
    numeric_features = set(NUM_FEATURES) & set(set_used_features)
    long_df = pd.DataFrame()
    for f in numeric_features:
        list_headers = [col for col in df.columns if f in col]
        long_df[f'{f}'] = df[list_headers].melt(value_name=f'{f}')[f'{f}']
        if f == 'SF12':
            sf12_max_min = long_df[f].max() - long_df[f].min()
        df[list_headers] = (df[list_headers] - long_df[f].mean()) / (
                long_df[f].max() - long_df[f].min())
    if 'SEX' in df.columns:
        df['SEX'] = df['SEX'] - 1
    return df, sf12_max_min

def min_normalization(df, set_used_features):
    numeric_features = set(NUM_FEATURES) & set(set_used_features)
    # long_df = pd.DataFrame()
    dict = {}
    for f in numeric_features:
        list_headers = [col for col in df.columns if f in col]
        long_df = df[list_headers].melt(value_name=f'{f}')[f'{f}']
        min = long_df.min()
        max = long_df.max()
        dict[f'{f}_min'] = float(min)
        dict[f'{f}_max'] = float(max)
        dict[f'{f}_interval'] = float(max - min)

        df[list_headers] = (df[list_headers] - min) / (max - min)
    if 'SEX' in df.columns:
        df['SEX'] = df['SEX'] - 1
    return df, dict
def beam_data(entry, metadata):
    n_segments = 4
    mean_bmi = 28
    mean_age = 60
    mean_womac = 8.78
    bmi_ranges = [0, 18.5, 25, 30, float("Inf")]

    input = {}

    # AGE
    age_stats = [45, 60, 90 + 1]
    d_age = (age_stats[-1] - age_stats[0]) / n_segments
    age_ranges = [age_stats[0] + i * d_age for i in range(n_segments + 1)]
    age_header = [a for a in metadata if 'AGE' in a]
    if len(age_header) > 0:
        age_header = age_header[0]
        age_level = None
        # input["AGE_mask"] = torch.tensor(True)
        age_ori = float(entry[f'{age_header}'])
        for i in range(len(age_ranges) - 1):
            if age_ranges[i] <= age_ori < age_ranges[i + 1]:
                age_level = torch.tensor(i)
                break

        age = [0.0] * n_segments
        age[age_level] = 1.0
        input["AGE"] = torch.tensor(age)

    # SEX
    if "SEX" in metadata:
        # input["SEX_mask"] = torch.tensor(True)
        if (isinstance(entry['SEX'], str) and "Female" in entry['SEX']) or (
                isinstance(entry['SEX'], int) and entry['SEX'] == 0):
            sex = [0.0, 1.0]
        else:
            sex = [1.0, 0.0]
        input["SEX"] = torch.tensor(sex)

    # BMI
    bmi_header = [a for a in metadata if 'BMI' in a]
    if len(bmi_header) > 0:
        bmi_header = bmi_header[0]
        bmi_ori = float(entry[f'{bmi_header}'])
        bmi_level = None
        for bmi_i in range(len(bmi_ranges) - 1):
            if bmi_ranges[bmi_i] <= bmi_ori < bmi_ranges[bmi_i + 1]:
                bmi_level = torch.tensor(bmi_i)
                break
        if bmi_level is None:
            # input["BMI_mask"] = torch.tensor(False)
            input["BMI"] = torch.tensor([0.0] * 4, dtype=torch.float32)
        else:
            # input["BMI_mask"] = torch.tensor(True)
            bmi = [0.0] * 4
            bmi[bmi_level] = 1.0
            input["BMI"] = torch.tensor(bmi)

    # INJURY
    inj_header = [a for a in metadata if 'INJ' in a]
    if len(inj_header) > 0:
        inj_header = inj_header[0]
        # input['INJ_mask'] = torch.tensor(True)
        inj = [1.0, 0.0] if (isinstance(entry[f'{inj_header}'], str) and "0" in entry[f'{inj_header}']) or entry[f'{inj_header}'] == 0 else [0.0, 1.0]
        input["INJ"] = torch.tensor(inj)

    # SURGERY
    surg_header = [a for a in metadata if 'SURG' in a]
    if len(surg_header) > 0:
        surg_header = surg_header[0]
        # input["SURG_mask"] = torch.tensor(True)
        surg = [1.0, 0.0] if (isinstance(entry[f'{surg_header}'], str) and "0" in entry[f'{surg_header}']) or entry[f'{surg_header}'] == 0 else [
            0.0, 1.0]
        input["SURG"] = torch.tensor(surg)

    # WOMAC
    womac_stats = [0, 9, 85 + 1]
    d_womac = (womac_stats[-1] - womac_stats[0]) / n_segments
    womac_ranges = [womac_stats[0] + i * d_womac for i in range(n_segments + 1)]
    womac_header = [a for a in metadata if 'WOMAC' in a]
    if len(womac_header) > 0:
        womac_header = womac_header[0]
        womac_level = 0
        # input['WOMAC_mask'] = torch.tensor(True)
        womac_ori = float(entry[f'{womac_header}'])
        for i in range(len(womac_ranges) - 1):
            if womac_ranges[i] <= womac_ori < womac_ranges[i + 1]:
                womac_level = i
                break
        if womac_level is None:
            raise ValueError(f'Cannot find WOMAC level of {womac_ori}.')
        womac = [0.0] * n_segments
        womac[womac_level] = 1.0
        input["WOMAC"] = torch.tensor(womac)

    # KL
    kl_header = [a for a in metadata if 'KL' in a]
    if len(kl_header) > 0:
        kl_header = kl_header[0]
        # input["KL_mask"] = torch.tensor(True)
        kl_ori = entry[f'{kl_header}']
        kl_level = torch.tensor(kl_ori - 1, dtype=torch.int32)

        kl = [0.0] * n_segments
        kl[kl_level] = 1.0
        input["KL"] = torch.tensor(kl)

    return input

def split_train_val_test(cfg, df, filename):
    unchanged_features_header = cfg.fix_features
    n_steps = cfg.time_interval
    test_site = cfg.test_site
    progressed_header = ['Progress']

    group_headers = ['Site', 'ID']


    df_train_temp = df[df['Site']!=test_site]
    df_test = df[df['Site']==test_site]

    if not os.path.exists(os.path.join(os.path.join(cfg.root_path, cfg.dataframe_path),filename)):
        splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state=cfg.seed)
        split = splitter.split(df_train_temp, df_train_temp['Progress'], groups=df_train_temp['ID'])
        train_inds, val_inds = next(split)
        data_pickle = {}
        data_pickle['train'] = train_inds
        data_pickle['validation'] = val_inds
        save_path = os.path.join(cfg.root_path, cfg.dataframe_path)
        with open(os.path.join(save_path,filename), 'wb') as f:
            pickle.dump(data_pickle, f)
    else:
        data_pickle = pd.read_pickle(os.path.join(os.path.join(cfg.root_path, cfg.dataframe_path),filename))
        train_inds = data_pickle['train']
        val_inds = data_pickle['validation']



    df_train = df_train_temp.iloc[train_inds]
    df_val = df_train_temp.iloc[val_inds]

    df_train = df_train.drop(columns=group_headers).reset_index(drop=True)
    df_val = df_val.drop(columns=group_headers).reset_index(drop=True)
    df_test = df_test.drop(columns=group_headers).reset_index(drop=True)

    if cfg.balance_data_by_cancel_out:
        cfg.balance_data_by_weight_sample = False


        df_train_dismiss= df_train[df_train['Progress'] == 0].sample(frac=0.1)
        df_train_follow = df_train[df_train['Progress'] != 0]
        df_train =  pd.concat([df_train_dismiss, df_train_follow], axis=0)
        df_train = df_train.sample(frac=1, random_state=1).reset_index(drop=True)


    print(f'Number of trained data: {len(df_train)}')
    print(f'Number of validation data: {len(df_val)}')
    print(f'Number of test data: {len(df_test)}')

    # train_data_filename = f'train_{filename}.pkl'
    # val_data_filename = f'val_{filename}.pkl'
    # test_data_filename = f'test_{filename}.pkl'
    #
    # df_train.to_pickle(f'{cfg.root_path}{train_data_filename}', protocol=4)
    # df_val.to_pickle(f'{cfg.root_path}{val_data_filename}', protocol=4)


    # df_test.to_pickle(f'{cfg.root_path}{test_data_filename}', protocol=4)

    return df_train, df_val, df_test

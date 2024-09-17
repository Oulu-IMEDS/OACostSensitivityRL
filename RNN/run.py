import torch
import pandas as pd
import hydra
import os
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss, BCEWithLogitsLoss
from torch.optim import Adam
import numpy as np
import random

from RNN.loader import DataFrameDatasetforRNN
from RNN.RNN_loop import RNNPipeline, RNNPipelinePredictProgressionBaselineInput
from common.model import QNetwork, LogisticRegression, QNetworkGRU, QNetworkGRU_forPretrain, GRU_predictProgression, LSTM_predictProgression
from data.preprocess_data import preprocess, split_train_val_test
from torch.utils.data import DataLoader, WeightedRandomSampler
from data.create_dataframe import GenerateDataFrameOAI

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

    batch_size = cfg.batch_size
    n_epochs = cfg.n_epochs
    device = cfg.device
    n_actions = 2
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


    df_train, df_val, df_test = split_train_val_test(cfg, post_processed_data, f'Train_val_indx_seed{cfg.seed}.pkl')



    feature_headers = [t for t in df_train.columns if f'@0' in t]
    target_features = [f for f in feature_headers if f'{cfg.reward_feature[0]}' in f]
    state_size = len(feature_headers) + len(cfg.fix_features) - len(target_features) + 2
    print(state_size)

    train_ds = DataFrameDatasetforRNN(cfg, '', df_train)
    val_ds = DataFrameDatasetforRNN(cfg, '', df_val)




    train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=8,drop_last=True )

    eval_loader = torch.utils.data.DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False)

    # model = LogisticRegression(state_size, 1)
    if cfg.RNN_method == 'GRU':
        model = GRU_predictProgression(cfg, state_size, action_size=n_actions)
    elif cfg.RNN_method == 'LSTM':
        model = LSTM_predictProgression(cfg, state_size, action_size=n_actions)

    model.to(device)
    lr = 1e-3
    wd = cfg.wd
    loss_func = {}
    loss_func['MSE'] = MSELoss()
    loss_func['CE'] = CrossEntropyLoss()
    loss_func['BCE'] = BCELoss()
    loss_func['BCEWithLogits'] = BCEWithLogitsLoss()
    optimizer = Adam(params=model.parameters(), lr=lr, weight_decay=wd)
    pipeline = RNNPipelinePredictProgressionBaselineInput(cfg, model,loss_func,optimizer,device)
    for epoch_id in range(n_epochs):
        pipeline.train_loop(train_loader, epoch_id)
        pipeline.eval_loop(eval_loader, epoch_id)

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



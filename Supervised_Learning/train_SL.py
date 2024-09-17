import torch
import pandas as pd
import hydra
import os
from torch.nn import CrossEntropyLoss, BCELoss
from torch.optim import Adam
from Supervised_Learning.main_loop import SupervisedLearningPipeline
from Supervised_Learning.data_loader import DataFrameDataset
from common.model import QNetwork, LogisticRegression
from data.preprocess_data import preprocess
from torch.utils.data import DataLoader, WeightedRandomSampler
from data.create_dataframe import GenerateDataFrameOAI


def main_process(model, train_loader, eval_loader, device='cpu', n_epochs=10):
    model.to(device)
    lr = 1e-4
    wd = 1e-4
    loss_func = CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=lr, weight_decay=wd)
    pipeline = SupervisedLearningPipeline(model,loss_func,optimizer,device)
    for epoch_id in range(n_epochs):
        pipeline.train_loop(train_loader, epoch_id)
        pipeline.eval_loop(eval_loader, epoch_id)

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

    # _path = f'{cfg.root_path}'
    data_path = os.path.join(f'{cfg.root_path}{cfg.dataframe_path}',cfg.data_filename)
    if not os.path.exists(data_path):
        OAI_data = GenerateDataFrameOAI(cfg, progression_var='KL')
        if cfg.progress_feature == 'KL':
            raw_data = OAI_data.KL_progression()
        elif cfg.progress_feature == 'JSW':
            raw_data = OAI_data.JSW_progression()
        elif cfg.progress_feature == 'SF12':
            raw_data = OAI_data.SF12_progression()
        print(raw_data)
    else:
        raw_data = pd.read_pickle(data_path)

    features_headers = [s + f'@{cfg.time_baseline}' for s in cfg.set_features]
    features_headers.extend(cfg.fix_features)

    df_train, df_val, df_test = preprocess(cfg, raw_data)


    state_size = len(features_headers)

    train_ds = DataFrameDataset(cfg, '', df_train, feature_headers=features_headers)
    val_ds = DataFrameDataset(cfg, '', df_val, feature_headers=features_headers)
    test_ds = DataFrameDataset(cfg, '', df_test, feature_headers=features_headers)


    if cfg.balance_data_by_weight_sample:
        print("Using frequency-based sampler for training subset")
        map_freqs = df_train['Progress'].value_counts(normalize=True).to_dict()
        sample_weights = [1.0 / map_freqs[e] for e in df_train['Progress'].tolist()]

        sampler_train = WeightedRandomSampler(weights=sample_weights,
                                              num_samples=len(sample_weights),
                                              replacement=True)

        train_loader = torch.utils.data.DataLoader(dataset=train_ds, sampler = sampler_train,
                                                   batch_size=batch_size, num_workers=0)
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    eval_loader = torch.utils.data.DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = LogisticRegression(state_size, 1)
    # model = QNetwork(cfg, state_size, action_size=n_actions, seed = seed)

    model.to(device)
    lr = 1e-3
    wd = cfg.wd
    loss_func = BCELoss()
    optimizer = Adam(params=model.parameters(), lr=lr, weight_decay=wd)
    pipeline = SupervisedLearningPipeline(cfg, model,loss_func,optimizer,device)
    for epoch_id in range(n_epochs):
        pipeline.train_loop(train_loader, epoch_id)
        pipeline.eval_loop(eval_loader, epoch_id)

    # main_process(model,
    #              train_loader,
    #              eval_loader,
    #              device=device,
    #              n_epochs=n_epochs)
    pretrain_model_path = os.getcwd()
    model_filename = f'oai_best_roc_auc_{cfg.set_features}{cfg.fix_features}.pth'
    model_path = os.path.join(pretrain_model_path, model_filename)

    checkpoint = torch.load(model_path)
    pipeline.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    optimizer = Adam(params=model.parameters(), lr=lr, weight_decay=wd)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    pipeline.test_loop(test_loader)

if __name__ == "__main__":
    main()



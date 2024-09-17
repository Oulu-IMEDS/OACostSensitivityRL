import torch
import pandas as pd
import hydra
import os
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from Supervised_Learning.main_loop import SupervisedLearningPipeline
from Supervised_Learning.data_loader import DataFrameDataset
from common.model import QNetwork, LogisticRegression
from data.preprocess_data import preprocess
from torch.utils.data import DataLoader, WeightedRandomSampler
from data.create_dataframe import GenerateDataFrameOAI



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

    _path = f'{cfg.root_path}'
    data_path = os.path.join(f'{cfg.root_path}{cfg.dataframe_path}',cfg.data_filename)
    if not os.path.exists(data_path):
        OAI_data = GenerateDataFrameOAI(cfg, progression_var='KL')
        if cfg.progress_feature == 'KL':
            raw_data = OAI_data.KL_progression()
        elif cfg.progress_feature == 'JSW':
            raw_data = OAI_data.JSW_progression()
        print(raw_data)
    else:
        raw_data = pd.read_pickle(data_path)

    features_headers = [s + '@0' for s in cfg.set_features]
    features_headers.extend(cfg.fix_features)

    df_train, df_val, df_test= preprocess(cfg, raw_data)


    state_size = len(features_headers)

    test_ds = DataFrameDataset(cfg, '', df_test, feature_headers=features_headers)


    test_loader = torch.utils.data.DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = LogisticRegression(state_size, 1)
    # model = QNetwork(cfg, state_size, action_size=n_actions, seed = seed)
    pretrain_model_path = os.getcwd()
    model_filename = f'oai_best_bacc_{cfg.set_features}{cfg.fix_features}.pth'
    model_path = os.path.join(pretrain_model_path, model_filename)

    lr = cfg.lr
    wd = cfg.wd
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    optimizer = Adam(params=model.parameters(), lr=lr, weight_decay=wd)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    model.to(device)
    # model.load_state_dict(torch.load(model_path), strict=True )

    loss_func = CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=lr, weight_decay=wd)
    pipeline = SupervisedLearningPipeline(cfg, model,loss_func,optimizer,device)


    pipeline.test_loop(test_loader)


if __name__ == "__main__":
    main()



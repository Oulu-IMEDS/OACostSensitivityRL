import torch
import torch.nn as nn
import torch.nn.functional as F
from common.oai_utils import *
import numpy as np
import pandas as pd

class QNetwork(nn.Module):

    def __init__(self, cfg, state_size, action_size):
        super(QNetwork, self).__init__()
        self.cfg = cfg
        if self.cfg.data_level == 'patient':
            n_KL = 2
        elif self.cfg.data_level == 'knee':
            n_KL = 1

        if cfg.progressor_type == 'feature':
            n_input_KL_layer = 2048
            self.layer_KL = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(n_input_KL_layer, 512),
                nn.Linear(512, self.cfg.arch.l_KL_out)
            )
        elif cfg.progressor_type == 'noImage':
            n_KL = 0
        else:
            n_input_KL_layer = 5
            self.layer_KL = nn.Identity()

        n_exclude_KL = state_size - n_KL
        n_other_var_out = n_exclude_KL
        self.layer_other_var = nn.Identity()

        n_input_combine = n_other_var_out + self.cfg.arch.l_KL_out * n_KL
        self.layer_pre_FCN = nn.Sequential(
            nn.BatchNorm1d(n_input_combine),
            nn.Linear(n_input_combine, n_input_combine),
            nn.Dropout(self.cfg.arch.drop_out),
            nn.Sigmoid()
        )

        self.predict_q = nn.Sequential(
            nn.Linear(n_input_combine, action_size)
        )

class QNetwork_forPatient(QNetwork):
    def __init__(self, cfg, state_size, action_size):
        super(QNetwork_forPatient, self).__init__(cfg, state_size, action_size)

    def forward(self, input):
        """Build a network that maps state -> action values."""
        list_keys = list(input.keys())
        KL_0_key = f'KL_0'
        KL_1_key = f'KL_1'
        f_kl_0 = self.layer_KL(input[f'{KL_0_key}'])
        f_kl_1 = self.layer_KL(input[f'{KL_1_key}'])

        headers = [e for e in list_keys if 'KL' not in e]
        tensor_exclude_kl = torch.concat([input[i] for i in headers], dim=-1).float()
        out_tensor_clinical_var_and_steps = self.layer_other_var(tensor_exclude_kl)
        pre_tensor = torch.concat([out_tensor_clinical_var_and_steps, f_kl_0, f_kl_1], dim=-1)
        input_tensor = self.layer_pre_FCN(pre_tensor)

        outputs = self.predict_q(input_tensor)
        return outputs

class QNetwork_forPatient_NoImage(QNetwork):
    def __init__(self, cfg, state_size, action_size):
        super(QNetwork_forPatient_NoImage, self).__init__(cfg, state_size, action_size)

    def forward(self, input):
        """Build a network that maps state -> action values."""
        list_keys = list(input.keys())

        headers = [e for e in list_keys if 'KL' not in e]
        tensor_exclude_kl = torch.concat([input[i] for i in headers], dim=-1).float()
        out_tensor_clinical_var_and_steps = self.layer_other_var(tensor_exclude_kl)
        input_tensor = self.layer_pre_FCN(out_tensor_clinical_var_and_steps)

        outputs = self.predict_q(input_tensor)
        return outputs

class QNetwork_forKnee(QNetwork):
    def __init__(self, cfg, state_size, action_size):
        super(QNetwork_forKnee, self).__init__(cfg, state_size, action_size)

    def forward(self, input):
        """Build a network that maps state -> action values."""
        list_keys = list(input.keys())
        KL_key = f'KL'
        f_kl = self.layer_KL(input[f'{KL_key}'])

        headers = [e for e in list_keys if 'KL' not in e]
        tensor_exclude_kl = torch.concat([input[i] for i in headers], dim=-1).float()
        out_tensor_clinical_var_and_steps = self.layer_other_var(tensor_exclude_kl)
        pre_tensor = torch.concat([out_tensor_clinical_var_and_steps, f_kl], dim=-1)
        if len(input['SF12'].shape) > 0:
            input_tensor = self.layer_pre_FCN(pre_tensor)
        else:
            input_tensor = self.layer_pre_FCN(pre_tensor.unsqueeze(0))

        outputs = self.predict_q(input_tensor)
        return outputs

class GRU_predictProgression(nn.Module):

    def __init__(self, cfg, state_size, action_size, fc1_units=64, fc2_units=64):
        super(GRU_predictProgression, self).__init__()
        self.cfg = cfg
        self.n_meta_out_features = 0
        self.n_metadata = 0
        self.input_data = []
        self.input_data.extend(self.cfg.fix_features)
        self.input_data.extend(self.cfg.set_features)
        self.n_meta_features = cfg.n_meta_features
        self.name = 'GRU'

        self.layer_KL = nn.Linear(5, 5)

        self.layer_other_var = nn.Linear(state_size - 2, 12)

        self.layer_pre_RNN = nn.Sequential(
            nn.BatchNorm1d(22),
            nn.Linear(22, 22),
            nn.Dropout(0.2)
        )

        self.RNN = nn.GRU(input_size=22, hidden_size=cfg.RNN.hidden_size,
                          num_layers=cfg.RNN.n_layers, bias=True, batch_first=True, dropout=cfg.RNN.drop_out,
                          bidirectional=True)

        self.predict_p = nn.Sequential(
            nn.Linear(2*cfg.RNN.hidden_size, fc1_units),
            nn.Linear(fc1_units, action_size)
        )

    def categorize_KL(self, KL_num):
        n_segments = 5

        if type(KL_num) == np.ndarray:
            kl_level = KL_num.astype(int)
            kl_level = np.stack((np.arange(len(KL_num)), kl_level), axis=0)
            kl = np.zeros((len(KL_num), n_segments))
            kl[kl_level[0, :], kl_level[1, :]] = 1.0
        else:
            kl_level = KL_num.astype(int)
            kl = np.zeros((1,n_segments))
            kl[:, kl_level] = 1.0
        out_kl = torch.tensor(kl, device=self.cfg.device).squeeze(0).float()
        return out_kl



    def forward(self, input, h, c=None):
        """Build a network that maps state -> action values."""
        list_keys = list(input.keys())
        KL_0_key = f'KL_0'
        KL_1_key = f'KL_1'
        f_kl_0 = self.layer_KL(input[f'{KL_0_key}'])
        f_kl_1 = self.layer_KL(input[f'{KL_1_key}'])

        headers = [e for e in list_keys if ('KL' not in e) and ('Progress' not in e)]
        tensor_exclude_kl = torch.stack([input[i] for i in headers], dim=-1).float()
        out_tensor_clinical_var_and_steps = self.layer_other_var(tensor_exclude_kl)
        pre_tensor = torch.concat([out_tensor_clinical_var_and_steps, f_kl_0, f_kl_1], dim=-1)
        input_tensor = self.layer_pre_RNN(pre_tensor).unsqueeze(1)
        out, h_out = self.RNN(input_tensor, h)
        action_logit = self.predict_p(out)

        return action_logit, h_out

class LSTM_predictProgression(nn.Module):

    def __init__(self, cfg, state_size, action_size, fc1_units=64, fc2_units=64):
        super(LSTM_predictProgression, self).__init__()
        self.cfg = cfg
        self.n_meta_out_features = 0
        self.n_metadata = 0
        self.input_data = []
        self.input_data.extend(self.cfg.fix_features)
        self.input_data.extend(self.cfg.set_features)
        self.n_meta_features = cfg.n_meta_features
        self.name = 'LSTM'

        self.layer_KL = nn.Linear(5, 5)

        self.layer_other_var = nn.Linear(state_size - 2, 12)

        self.layer_pre_RNN = nn.Sequential(
            nn.BatchNorm1d(22),
            nn.Linear(22, 22),
            nn.Dropout(0.2)
        )

        self.RNN = nn.LSTM(input_size=22, hidden_size=cfg.RNN.hidden_size,
                          num_layers=cfg.RNN.n_layers, bias=True, batch_first=True, dropout=cfg.RNN.drop_out,
                          bidirectional=True)

        self.predict_p = nn.Sequential(
            nn.Linear(2*cfg.RNN.hidden_size, fc1_units),
            nn.Linear(fc1_units, action_size)
        )

    def categorize_KL(self, KL_num):
        n_segments = 5

        if type(KL_num) == np.ndarray:
            kl_level = KL_num.astype(int)
            kl_level = np.stack((np.arange(len(KL_num)), kl_level), axis=0)
            kl = np.zeros((len(KL_num), n_segments))
            kl[kl_level[0, :], kl_level[1, :]] = 1.0
        else:
            kl_level = KL_num.astype(int)
            kl = np.zeros((1,n_segments))
            kl[:, kl_level] = 1.0
        out_kl = torch.tensor(kl, device=self.cfg.device).squeeze(0).float()
        return out_kl



    def forward(self, input, hc):
        """Build a network that maps state -> action values."""
        list_keys = list(input.keys())
        KL_0_key = f'KL_0'
        KL_1_key = f'KL_1'
        f_kl_0 = self.layer_KL(input[f'{KL_0_key}'])
        f_kl_1 = self.layer_KL(input[f'{KL_1_key}'])

        headers = [e for e in list_keys if ('KL' not in e) and ('Progress' not in e)]
        tensor_exclude_kl = torch.stack([input[i] for i in headers], dim=-1).float()
        out_tensor_clinical_var_and_steps = self.layer_other_var(tensor_exclude_kl)
        pre_tensor = torch.concat([out_tensor_clinical_var_and_steps, f_kl_0, f_kl_1], dim=-1)

        input_tensor = self.layer_pre_RNN(pre_tensor).unsqueeze(1)
        out, (h_out,c_out) = self.RNN(input_tensor, hc)
        action_logit = self.predict_p(out)

        return action_logit, (h_out,c_out)


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        logits = self.linear(x)
        probs = torch.sigmoid(logits)
        outputs = probs.squeeze()

        return outputs
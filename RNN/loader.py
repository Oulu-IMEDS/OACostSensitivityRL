import pandas as pd
import numpy as np

import cv2
import os
from torch.utils.data import Dataset
import torch
from data.preprocess_data import beam_data


class DataFrameDatasetforRNN(Dataset):

    def __init__(self, cfg, root, data):
        if not isinstance(root, str):
            raise TypeError("`root` must be `str`")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("`meta_data` must be `pandas.DataFrame`, but found {}".format(type(data)))
        self.cfg = cfg
        self.root = root
        self.data = data

    # def standardize_image_size(self, img):
    #     h = img.shape[0]
    #     w = img.shape[1]
    #     min_scale = min(self.std_size[1 ] /w, self.std_size[0 ] /h)
    #     new_w = int(np.round(min_scale * w))
    #     new_h = int(np.round(min_scale * h))
    #     new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    #     return new_img
    #
    # def parse_item(self, root, entry, transform):
    #     # Read image from path
    #     img_fullname = os.path.join(root, entry['path'])
    #     img = cv2.imread(img_fullname, cv2.IMREAD_COLOR)
    #     img = self.standardize_image_size(img)
    #
    #     stats = {'mean': self.mean, 'std': self.std}
    #     # Apply transformations into image
    #     trf_output = transform({'image': img}, return_torch=True, normalize=True, **stats)
    #     return {'data': trf_output['image'], 'target': entry['target'], 'body_type': entry['body_type']}
    def categorize_KL(self, KL_num):
        n_segments = 5
        kl_level = KL_num.astype(int)
        kl = np.zeros((n_segments))
        kl[kl_level] = 1.0
        return kl

    def parse_item(self, root, entry):
        out = {}
        encode_visit = ['0', '12', '24', '36', '48', '72', '96']
        for i in range(self.cfg.time_interval + 1):
            target_features = [f for f in entry.keys() if f'{self.cfg.reward_feature[0]}' in f]
            feature_headers = [t for t in entry.keys() if (f'@{encode_visit[i]}' in t) and ('Progress' not in t)]
            progress_headers = [p for p in entry.keys() if f'Progress@{encode_visit[i+1]}' in p]
            feature_headers = set(feature_headers) - set(target_features)
            feature_headers = sorted(feature_headers)
            feature_headers.extend(self.cfg.fix_features)
            feature_headers.extend(progress_headers)
            data = entry[feature_headers]
            data.rename({key: key.split(f'@{encode_visit[i]}')[0] for key in data.keys()}, inplace=True)
            data.rename({key: key.split(f'@{encode_visit[i+1]}')[0] for key in data.keys()}, inplace=True)
            if self.cfg.progressor_type == 'num':
                data[f'KL_0'] = self.categorize_KL(data[f'KL_0'])
                data[f'KL_1'] = self.categorize_KL(data[f'KL_1'])
            else:
                pass
            data['Info_step'] = float(i)
            data['Current_step'] = float(i)
            # data = np.append(data, i)
            # data = np.append(data, i)
            # out[f'visit{encode_visit[i]}'] = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(self.cfg.device)
            out[f'visit{encode_visit[i]}'] = data.to_dict()
        # target = torch.tensor(entry['Progress']).type(torch.LongTensor)
        # target = torch.tensor(entry['Progress'], dtype=torch.float32)

        # trf_output = transform({'image': img}, return_torch=True, normalize=True, **stats)
        return out

    def __getitem__(self, index):
        """Get ``index``-th parsed item of :attr:`meta_data`.

        Parameters
        ----------
        index : int
            Index of row.

        Returns
        -------
        entry : dict
            Dictionary of `index`-th parsed item.
        """
        entry = self.data.iloc[index]
        entry = self.parse_item(self.root, entry)
        if not isinstance(entry, dict):
            raise TypeError("Output of `parse_item_cb` must be `dict`, but found {}".format(type(entry)))
        return entry

    def __len__(self):
        """Get length of `meta_data`.
        """
        return len(self.data.index)

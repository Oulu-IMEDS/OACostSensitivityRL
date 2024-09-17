import pandas as pd
import numpy as np

import cv2
import os
from torch.utils.data import Dataset
import torch
from data.preprocess_data import beam_data




class DataFrameDataset(Dataset):
    """Dataset based on ``pandas.DataFrame``.

    Parameters
    ----------
    root : str
        Path to root directory of input data.
    meta_data : pandas.DataFrame
        Meta data of data and labels.
    transform : callable, optional
        Transformation applied to row of :attr:`meta_data` (the default is None, which means to do nothing).
    std_size: tuple
        The size that input images are standardized to (Default: (224, 224))
    mean: tuple
        Mean to normalize image intensities (Default: (0.485, 0.456, 0.406))
    std: tuple
        Standard deviation to normalize image intensities (Default: (0.229, 0.224, 0.225))

    Raises
    ------
    TypeError
        `root` must be `str`.
    TypeError
        `meta_data` must be `pandas.DataFrame`.

    """

    def __init__(self, cfg, root, meta_data, feature_headers, transform=None, std_size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        if not isinstance(root, str):
            raise TypeError("`root` must be `str`")
        if not isinstance(meta_data, pd.DataFrame):
            raise TypeError("`meta_data` must be `pandas.DataFrame`, but found {}".format(type(meta_data)))
        self.cfg = cfg
        self.root = root
        self.meta_data = meta_data
        self.feature_headers = feature_headers
        self.transform = transform
        self.std_size = std_size
        self.mean = mean
        self.std = std

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

    def parse_item(self, root, entry, transform):
        # norm_entry = beam_data(entry, self.feature_headers)
        if self.cfg.beam_data:
            input = beam_data(entry, self.feature_headers)
        else:
            input = torch.tensor(entry[self.feature_headers], dtype=torch.float32)

        # target = torch.tensor(entry['Progress']).type(torch.LongTensor)
        target = torch.tensor(entry['Progress'], dtype=torch.float32)

        # trf_output = transform({'image': img}, return_torch=True, normalize=True, **stats)
        return {'data': input, 'target': target }

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
        entry = self.meta_data.iloc[index]
        entry = self.parse_item(self.root, entry, self.transform)
        if not isinstance(entry, dict):
            raise TypeError("Output of `parse_item_cb` must be `dict`, but found {}".format(type(entry)))
        return entry

    def __len__(self):
        """Get length of `meta_data`.
        """
        return len(self.meta_data.index)
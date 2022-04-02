import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PretrainDataset(Dataset):
    """
    Pretraining Pytorch Dataset.
    The original version of step_size is 1 by default. In order to speed up training, step_size is added,
    which is equivalent to the distance between different time windows

    self.df : DataFrame
    """
    def __init__(self, data_des, window_size, col_leave, step_size=1):
        self.window_size = window_size
        self.df = pd.read_csv(data_des)
        if col_leave:
            col_list = [col for col in self.df.columns if col in col_leave]
            self.df = self.df[col_list]
        self.dataset_size = int((len(self.df) - self.window_size)/step_size) + 1
        self.step_size = step_size
        self.std = self.df.std()

    def __getitem__(self, i):
        """
        :param i:
        :return:  (time_step. feature_size)
        """
        start = i * self.step_size
        end = start + self.window_size
        x = torch.from_numpy(self.df.iloc[start:end].values).float().to(device)
        return x

    def __len__(self):
        return self.dataset_size


class FTDataset(Dataset):
    """
    A Pytorch Dataset class to be used in PyTorch DataLoader to create batches
    """
    def __init__(self, data_x_de, data_y_de, col_leave, step=4, dur=60):
        self.step = step
        self.df = pd.read_csv(data_x_de)
        self.df_y = pd.read_csv(data_y_de)
        col_list = [col for col in self.df.columns if col in col_leave]
        col_sparse = [col for col in self.df.columns if "bin" in col]
        self.df_fm = self.df[col_sparse]
        self.df = self.df[col_list]
        self.dataset_size = int(len(self.df)-self.step+1)
        if dur != None:
            self.dataset_size = int(len(self.df)/dur)
        self.dur = dur

    def __getitem__(self, i):
        """
        :param i:
        :return:  (time_step. feature_size)
        """
        if self.dur:
            start = i * self.dur
            end = self.dur + i * self.dur
        else:
            start = i
            end = i + self.step
        x_ori = torch.from_numpy(self.df.iloc[start:end].values).float().to(device)
        x_fm = torch.from_numpy(self.df_fm.iloc[start:end].values).float().to(device)
        y_ori = torch.Tensor([self.df_y.iloc[i, -1]]).to(device)
        return x_ori[-self.step:], x_fm[-1].flatten().long(), y_ori

    def __len__(self):
        return self.dataset_size


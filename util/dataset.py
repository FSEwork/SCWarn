import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import minmax_scale
from util.preprocess import impute_missing, fill_nan
from datetime import datetime, timedelta



# faults = [
#     ['2020-04-03 03:00', '2020-04-03 23:23', '2020-04-04 00:02'],
#     ['2020-04-04 06:00', '2020-04-07 10:57', '2020-04-07 11:20'],
#     ['2020-04-08 00:00', '2020-04-08 22:23', '2020-04-08 22:38'],
#     ['2020-04-08 00:00', '2020-04-09 00:48', '2020-04-09 01:00'],
#     ['2020-04-09 12:00', '2020-04-13 23:31', '2020-04-13 23:52'],
#     ['2020-04-09 12:00', '2020-04-14 00:14', '2020-04-14 01:00'],
#     ['2020-04-14 03:59', '2020-04-16 15:15', '2020-04-16 15:40']
#
# ]
# used_fault = faults[4]



class SequenceDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = labels
        self.feature_len = seqs.shape[-1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.seqs[i], self.labels[i]


def apply_sliding_window(data, seq_len=10, flatten=False):
    """
    Parameters
    ----------
    data: sequence data
    seq_len: the length of sliding window

    Returns
    -------
    the first: data after being applied sliding window to
    the second: the ground truth; for example the values from t-w to t are the input so the value at t+1 is the ground
    truth.
    """
    seq_ls = []
    label_ls = []
    for i in range(seq_len, len(data)):
        if not flatten:
            seq_ls.append(data[i - seq_len: i])
        else:
            seq_ls.append(data[i - seq_len: i].flatten())
        label_ls.append(data[i])

    return np.array(seq_ls, dtype=np.float32), np.array(label_ls, dtype=np.float32)


def split_train_test(data, test_portion=0.4):
    train_len = round(len(data) * (1 - test_portion))
    return data[:train_len], data[train_len:]



def load_ts_data(filepath):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    df = df.set_index('timestamp')
    return df


def use_mini_batch(data, labels, batch_size):
    """
    Returns
    -------
    datalodaer is an iterable dataset. In each iteration, it will return a tuple, the first item is the data and the
    second item is the label. So this object is usually used in training a model.
    You can use len() to know the batch count of the dataset
    """
    seq_dataset = SequenceDataset(data, labels)
    dataloader = DataLoader(seq_dataset, batch_size=batch_size, drop_last=True)

    return dataloader


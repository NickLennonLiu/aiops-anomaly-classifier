import os
import pickle
from os.path import exists
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as stats


def get_cmdb_dt(dt, cmdb_id):
    return {key[1]: dt[key] for key in dt.keys() if key[0] == cmdb_id}

def get_kpi_dt(dt, kpi_name):
    return {key[0]: dt[key] for key in dt.keys() if key[1] == kpi_name}

def get_kpi_at_time(data, cmdb_idx, t, window=1):
    t = int(t+0.5)
    return data[t:t+window, cmdb_idx]


def get_time_range(df):
    ranges = []
    start = None
    end = None
    for timestamp, value in zip(df.index, df.values.squeeze()):
        if not np.isnan(value):
            if start is None:
                start = timestamp
            end = timestamp
        elif start:
            ranges.append(pd.Interval(left=pd.Timestamp(start), right=pd.Timestamp(end)))
            start = None
            end = None
    if start:
        ranges.append(pd.Interval(start, end))
    return pd.arrays.IntervalArray(ranges)


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, x):
        """
        :param x:  (time, cmdb_id)
        :return:   anomaly type in range(8)
        """
        return np.random.randint(0, self.label_num)

class Basic(Model):
    def __init__(self, args, data, cmdb_kpi):
        super().__init__(args)

        self.data = F.pad(torch.nan_to_num(data), pad=(0,1)).to(torch.float32)
        self.cmdb_kpi = cmdb_kpi
        self.metric_num = len(cmdb_kpi)
        self.cmdb_idx = self.get_cmdb_idx()
        self.kpi_set = set([ck[1] for ck in self.cmdb_kpi])
        self.kpi_num = len(self.kpi_set)

        self.class_num = args.class_num
        self.workdir = args.workdir
        self.window = args.window

        self.stats = self.get_describe() # std mean median mode

        print(f"metric_num: {self.metric_num}, kpi_num: {self.kpi_num}, {self.data.shape}")

    def get_cmdb_idx(self):
        cmdb_set = list(set([ck[0] for ck in self.cmdb_kpi]))
        kpi_set = list(set([ck[1] for ck in self.cmdb_kpi]))

        cmdb_idx = {}
        for cmdb in cmdb_set:
            a = []
            for idx, kpi in enumerate(kpi_set):
                try:
                    found = self.cmdb_kpi.index((cmdb,kpi))
                    a.append(found)
                except ValueError:
                    a.append(self.metric_num)
            cmdb_idx[cmdb] = a
        return cmdb_idx

    def get_describe(self):
        filename = os.path.join(self.workdir, "data_stats.pkl")
        if exists(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            std, mean = torch.std_mean(self.data, dim=0)
            median = torch.median(self.data, dim=0)[0]
            mode = torch.mode(self.data,dim=0)[0]
            des = torch.stack([std, mean, median, mode])
            with open(filename, 'wb') as f:
                pickle.dump(des, f)
            return des

class SingleCMDB_MLP(Basic):
    def __init__(self, args, data, kpi_name):
        super().__init__(args, data, kpi_name)

        self.fc1 = nn.Linear(self.kpi_num, 50)
        self.fc2 = nn.Linear(50, self.class_num)

    def forward(self, x):
        cmdb_id, timestamp = x
        cmdb_idx = self.cmdb_idx[cmdb_id]
        x = get_kpi_at_time(self.data, cmdb_idx, timestamp, window=self.window) # time_len x metrics
        y = np.divide(np.abs(np.subtract(x, self.stats[1, cmdb_idx])), self.stats[0, cmdb_idx]+1e-10).unsqueeze(dim=0)
        x = torch.mean(y, dim=1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        return x

class KStest(Basic):
    def __init__(self, args, data, kpi_name):
        super().__init__(args, data, kpi_name)
        self.fc1 = nn.Linear(self.kpi_num, self.class_num) # self.data.shape[1]
        #self.fc2 = nn.Linear(200, 100)
        #self.fc3 = nn.Linear(100, self.class_num)

    def forward(self, x):
        cmdb_id, timestamp = x
        cmdb_idx = self.cmdb_idx[cmdb_id]
        x = get_kpi_at_time(self.data, cmdb_idx, timestamp, window=self.window) # time_len x metrics
        y = get_kpi_at_time(self.data, cmdb_idx, timestamp-self.window, window=self.window*2)
        # x = torch.tensor(np.argsort([stats.ks_2samp(x[:,i], y[:,i])[1] for i in range(self.data.shape[1])])[:500]).to(torch.float32)
        x = torch.tensor([stats.ks_2samp(x[:,i], y[:,i])[1] for i in range(self.kpi_num)]).to(torch.float32)
        # y = np.divide(np.abs(np.subtract(x[-1,:], self.stats[1, :])), self.stats[0, :]).float().unsqueeze(dim=0)
        x = x.unsqueeze(dim=0)
        x = F.leaky_relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        return x



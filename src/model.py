import os
import pickle
from os.path import exists

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src import stats


def get_cmdb_dt(dt, cmdb_id):
    return {key[1]: dt[key] for key in dt.keys() if key[0] == cmdb_id}

def get_kpi_dt(dt, kpi_name):
    return {key[0]: dt[key] for key in dt.keys() if key[1] == kpi_name}

def get_kpi_at_time(dt, datetime):
    return np.array([data.loc[datetime].value if (datetime in data.index)
                else np.NaN
                for data in dt.values()])

def get_time_range(df):
    ranges = []
    start = None
    end = None
    for timestamp, value in zip(df.index, df.value):
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


class Statistic(Model):
    def __init__(self, args, data):
        super().__init__(args)
        self.data = data
        self.kpi = len(self.data)
        self.class_num = args.class_num
        self.workdir = args.workdir

        self.time_range = self.get_time_range_dt()
        self.stats = self.get_describe()

        self.fc = nn.Linear(len(self.data), self.class_num)

    def get_time_range_dt(self):
        filename = os.path.join(self.workdir, "time_range.pkl")
        if exists(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            time_range = {}
            for key in self.data.keys():
                time_range[key] = get_time_range(self.data[key])
            with open(filename, 'wb') as f:
                pickle.dump(time_range, f)
            return time_range

    def get_describe(self):
        filename = os.path.join(self.workdir, "data_stats.pkl")
        if exists(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            des = {}
            for key in self.data.keys():
                des[key] = self.data[key].describe()
            with open(filename, 'wb') as f:
                pickle.dump(des, f)
            return des

    def forward(self, x):
        cmdb_id, timestamp = x
        x = get_kpi_at_time(self.data, timestamp)

        return 0




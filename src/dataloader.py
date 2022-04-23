import os
import pickle
import re

import numpy as np
import openpyxl
from itertools import islice

import pandas
import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import Dataset
from os.path import exists

from src.net import get_kpi_at_time
from src.params import get_args
from src.preprocess import preprocess_dt, preprocess_gt

cause_id = {"网络丢包": 0,
             "网络延迟": 1,
             "JVM CPU负载高": 2,
             "CPU使用率高": 3,
             "磁盘空间使用率过高": 4,
             "内存使用率过高": 5,
             "JVM OOM Heap": 6,
             "磁盘IO读使用率过高": 7}

def get_root_cause(args):
    gt_path = args.gt_path
    wb = openpyxl.load_workbook(gt_path)
    ws = wb.active
    data = ws.values
    cols = next(data)[1:]
    data = list(data)
    idx = [r[0] for r in data]
    data = (islice(r, 1, None) for r in data)
    df = DataFrame(data, index=idx, columns=cols)
    df["故障内容"] = df["故障内容"].map(lambda x: cause_id[x])
    df['time'] = pd.to_datetime(df.time)
    df['time'] = df['time'] - pd.Timedelta('08:00:00')
    root_cause = {i: set() for i in range(8)}
    rt_list = list(zip(df.根因, df.故障内容))
    for rt, label in rt_list:
        for key_metric in rt.split(';\n'):
            root_cause[label].add(key_metric)
    return root_cause

def get_cmdb(args):
    data_filename = os.path.join(args.workdir, "data_pre.pkl")
    # 如果有预处理过的数据，那么直接读取
    if exists(data_filename):
        print(f"Loading preprocessed data from {data_filename}")
        data, kpi_name = load_preprocessed(data_filename)
    else:
        print(f"Loading raw data from {args.dt_raw}")
        data = load_dt_raw(args)
        data, kpi_name = preprocess_dt(data, args.start_time)
        print(f"Saving preprocessed data to {data_filename}")
        with open(data_filename, 'wb') as f:
            pickle.dump([data, kpi_name], f)
    return data, kpi_name

def load_gt(args):
    gt_path = args.gt_path
    wb = openpyxl.load_workbook(gt_path)
    ws = wb.active

    data = ws.values
    cols = next(data)[1:]
    data = list(data)
    idx = [r[0] for r in data]
    data = (islice(r, 1, None) for r in data)
    df = DataFrame(data, index=idx, columns=cols).drop(["根因", "故障类别"], axis=1) # Remove '根因' for simplicity
    df["故障内容"] = df["故障内容"].map(lambda x: cause_id[x])
    df['time'] = pd.to_datetime(df.time)
    df['time'] = df['time'] - pd.Timedelta('08:00:00')
    return df

def load_preprocessed(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def load_dt_raw(args):
    """
    from dt_path, load kpi data which is not constant all the time.
    """
    dt_path = args.dt_raw
    data_dt = {}
    for file in os.listdir(dt_path):
        if not re.match(r"\w+##.*\.csv", file):
            continue
        cmdb_id, kpi_name = os.path.splitext(file)[0].split('##')
        df = pd.read_csv(os.path.join(dt_path, file), index_col=0, usecols=[1,4])
        df.index = pd.to_datetime(df.index, unit="s")
        df = df[~df.index.duplicated()]
        if df.nunique().iloc[0] > 1:
            data_dt[(cmdb_id, kpi_name)] = df
    return data_dt

def get_gt(args):
    gt_pre = os.path.join(args.workdir, "gt_pre.pkl")
    if exists(gt_pre):
        gt = load_preprocessed(gt_pre)
    else:
        gt = load_gt(args)
        gt = preprocess_gt(gt, args.start_time)
        with open(gt_pre, 'wb') as f:
            pickle.dump(gt, f)
    return gt

def get_multiIndex(args):
    dt_path = args.dt_raw
    cmdb_kpi = []
    for file in os.listdir(dt_path):
        try:
            cmdb_id, kpi_name = os.path.splitext(file)[0].split('##')
            cmdb_kpi.append((cmdb_id, kpi_name))
        except:
            continue
    print(f"{len(set([ck[0] for ck in cmdb_kpi]))}")
    print(f"{len(set([ck[1] for ck in cmdb_kpi]))}")
    print(f"Loaded with {len(cmdb_kpi)} cmdb_kpi tuples")
    return pd.MultiIndex.from_tuples(cmdb_kpi)

def get_cmdb_kpi_dict(args):
    dt_path = args.dt_preprocess
    cmdb_kpi = {}
    for file in os.listdir(dt_path):
        cmdb_id, kpi_name = os.path.splitext(file)[0].split('##')
        if cmdb_id not in cmdb_kpi:
            cmdb_kpi[cmdb_id] = [kpi_name]
        else:
            cmdb_kpi[cmdb_id].append(kpi_name)
    return cmdb_kpi

def get_cmdb_kpi(args, cmdb_id, kpi_name):
    dt_path = args.dt_path
    file = os.path.join(dt_path, f"{cmdb_id}##{kpi_name}.csv")
    return pd.read_csv(file, index_col=0)

def one_hot(ys, n):
    result = []
    for i,y in enumerate(ys):
        a = np.zeros(n)
        a[y] = 1
        result.append(a)
    return np.array(result)

class TSDataset(Dataset):
    def __init__(self, df, y):
        self.df = df
        self.y = y
    def __len__(self):
        return len(self.df)
    def __getitem__(self, item):
        return self.df.id.loc[item], self.y.loc[item]

class TensorDataset(Dataset):
    def __init__(self, x, y, class_num):
        self.x = x
        self.y = y["y"].values
        self.yy = one_hot(self.y, class_num)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, item):
        return self.x[item], self.yy[item]

class MyDataset2(Dataset):
    def __init__(self, pp):
        self.pp = pp
        self.pp["y"] = self.pp["y"].values
    def __len__(self):
        return self.pp["y"].size
    def __getitem__(self, item):
        dt = {
            "x": self.pp["x"][item],
            "ts": torch.from_numpy(self.pp["ts"][self.pp["ts"].id == item].values[:,2:]),
            "fft": torch.from_numpy(self.pp["fft"][self.pp["fft"].id == item].values[:,2:].astype(np.float32)),
            "stat": torch.from_numpy(self.pp["stat"][item,:]),
            "fft_stat": torch.from_numpy(self.pp["fft_stat"][item,:]),
        }
        return dt, one_hot(self.pp["y"][item],8)

class MyDataset(Dataset):
    def __init__(self, args):
        gt_pre = os.path.join(args.workdir, "gt_pre.pkl")
        self.class_num = args.class_num
        self.start_time = args.start_time
        if exists(gt_pre):
            print("Loaded ground truth from gt_pre.pkl")
            df = self.load(os.path.join(args.workdir, "gt_pre.pkl"))
            df = preprocess_gt(df, args.start_time)
            self.x = df.x.values
            self.y = df.y.values
        else:
            df = load_gt(args)
            self.x = list(zip(df.cmdb_id, df.time))
            self.y = df.故障内容.values

        print(np.bincount(self.y))
        self.yy = one_hot(self.y, self.class_num)


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.yy[idx]

    def load(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def dump(self, filename):
        df = DataFrame(data=list(zip(self.x, self.y)), columns=["x", "y"])
        print(df.head())
        with open(filename, 'wb') as f:
            pickle.dump(df, f)

    def remove_empty_gt(self, dt):
        new_x = []
        new_y = []
        for x, y in self:
            timestamp = x[1]
            kpi_data = get_kpi_at_time(dt, timestamp)
            if not np.isnan(kpi_data).all():
                new_x.append(x)
                new_y.append(y)
        self.x = new_x
        self.y = new_y

if __name__ == "__main__":
    _args = get_args()
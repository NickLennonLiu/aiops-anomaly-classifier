import os
import pickle
import re

import openpyxl
from itertools import islice

import pandas as pd
from pandas import DataFrame
from torch.utils.data import Dataset

from src.params import get_args
from src.preprocess import preprocess_dt

cause_id = {"网络丢包": 0,
             "网络延迟": 1,
             "JVM CPU负载高": 2,
             "CPU使用率高": 3,
             "磁盘空间使用率过高": 4,
             "内存使用率过高": 5,
             "JVM OOM Heap": 6,
             "磁盘IO读使用率过高": 7}

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
        # df = df.rename(columns={"value": kpi_name})
        if df.nunique().iloc[0] > 1:
            data_dt[(cmdb_id, kpi_name)] = df
            # if cmdb_id not in data_dt.keys():
            #     data_dt[cmdb_id] = {kpi_name: df}
            # else:
            #     data_dt[cmdb_id][kpi_name] = df
    return data_dt

def get_multiIndex(args):
    dt_path = args.dt_path
    cmdb_kpi = []
    for file in os.listdir(dt_path):
        cmdb_id, kpi_name = os.path.splitext(file)[0].split('##')
        cmdb_kpi.append((cmdb_id, kpi_name))
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


class MyDataset(Dataset):
    def __init__(self, args):
        df = load_gt(args)
        self.x = list(zip(df.cmdb_id, df.time))
        self.y = df.故障内容.values

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


if __name__ == "__main__":
    _args = get_args()
    load_dt(_args)
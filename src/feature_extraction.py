import os
import pickle

import numpy as np
import pandas as pd
from pandas import DataFrame


def get_cmdb_idx(cmdb_kpi, metric_num):
    cmdb_set = list(set([ck[0] for ck in cmdb_kpi]))
    kpi_set = list(set([ck[1] for ck in cmdb_kpi]))
    cmdb_idx = {}
    for cmdb in cmdb_set:
        a = []
        for idx, kpi in enumerate(kpi_set):
            try:
                found = cmdb_kpi.index((cmdb, kpi))
                a.append(found)
            except ValueError:
                a.append(metric_num)
        cmdb_idx[cmdb] = a
    return cmdb_idx

def get_kpi_at_time(data, cmdb_idx, t, window=1):
    t = int(t)
    return data[t:t+window, cmdb_idx]


def gen_timeseries(cmdb_idx_dt, gt, data, window, kpi_set):
    dfs_before = []
    dfs_after = []
    labels = []
    cnt = 0
    for idx, (x, y) in enumerate(zip(gt.x, gt.y)):
        cmdb_id, timestamp = x
        cmdb_idx = cmdb_idx_dt[cmdb_id]
        tensor_after = get_kpi_at_time(data, cmdb_idx, timestamp, window)
        tensor_before = get_kpi_at_time(data, cmdb_idx, timestamp - window, window)

        if np.isnan(tensor_after).all() or np.isnan(tensor_before).all():
            continue

        df_a = DataFrame(tensor_after, columns=list(kpi_set))
        df_a.reset_index(inplace=True)
        df_a.rename(columns={"index":"timestamp"}, inplace=True)

        df_b = DataFrame(tensor_before, columns=list(kpi_set))
        df_b.reset_index(inplace=True)
        df_b.rename(columns={"index": "timestamp"}, inplace=True)

        if df_a.values.shape[0] == window and df_b.values.shape[0] == window:
            df_b["id"] = pd.Series([cnt for i in range(len(df_b.index))])
            df_a["id"] = pd.Series([cnt for i in range(len(df_a.index))])
            cols = list(df_a.columns)
            cols = [cols[-1]] + cols[:-1]
            df_a = df_a[cols]
            df_b = df_b[cols]

            labels.append(y)
            dfs_after.append(df_a)
            dfs_before.append(df_b)
            cnt += 1

    ts_a = pd.concat(dfs_after)
    ts_b = pd.concat(dfs_before)

    y_df = DataFrame(labels, columns=["y"])
    return ts_a, ts_b, y_df

def get_timeseries(args, cmdb_kpi, data, gt, window):
    ts_file = os.path.join(args.workdir, "timeseries.pkl")
    if not os.path.exists(ts_file):
        cmdb_set = set([ck[0] for ck in cmdb_kpi])
        kpi_set = set([ck[1] for ck in cmdb_kpi])

        cmdb_idx_dt = get_cmdb_idx(cmdb_kpi, len(cmdb_kpi))
        data = np.pad(data, ((0, 0), (0, 1)))
        ts_a, ts_b, y_df = gen_timeseries(cmdb_idx_dt, gt, data, window, kpi_set)

        ts_a = ts_a.loc[:, (ts_a != ts_a.iloc[0]).any()]
        ts_a.dropna(how='all', inplace=True, axis=1)
        cols = list(ts_a.columns)
        ts_b = ts_b[cols]
        ts_b.id = ts_b.id.astype('int64')

        timeseries = {
            "after": ts_a,
            "before": ts_b,
            "y": y_df
        }

        with open(ts_file, 'wb') as f:
            pickle.dump(timeseries, f)
    else:
        with open(ts_file, 'rb') as f:
            timeseries = pickle.load(f)
    return timeseries

def get_fft(ts):
    ft = DataFrame(columns=ts.columns, index=ts.index, data=None)
    ft.id = ts.id
    ft.timestamp = ts.timestamp
    for i in range(ts.id.max()+1):
        if (ts.id != i).all():
            print(f"{i} missing, this should not be happening")
            continue
        t = ts[ts.id == i].loc[:, ts.columns[2]:]
        fft = abs(np.fft.fft(t, axis=0))
        ft.loc[ft.id == i, ts.columns[2]:] = fft
    return ft

def get_stats(ts):
    data_size = ts.id.max() + 1
    describe = []
    for i in range(data_size):
        if (ts.id != i).all():
            print(f"{i} missing, this should not be happening")
            continue
        val = ts[ts.id == i].loc[:, ts.columns[2]:]
        describe.append(val.describe()[1:].values.flatten(order='F'))
    describe = np.stack(describe)
    describe = np.nan_to_num(describe)
    return describe

def feature_extraction(data, cmdb_kpi, gt, window, args):
    pp_file = os.path.join(args.workdir, "preprocess_data.pkl")
    if os.path.exists(pp_file):
        with open(pp_file, 'rb') as f:
            pp = pickle.load(f)
        return pp
    else:
        ts = get_timeseries(args, cmdb_kpi, data, gt, window)
        ts_a = ts["after"]
        ts_b = ts["before"]
        y_df = ts["y"]
        ft_a = get_fft(ts_a)
        ft_b = get_fft(ts_b)
        ts["after_fft"] = ft_a
        ts["before_fft"] = ft_b
        ts["after_stat"] = get_stats(ts_a)
        ts["before_stat"] = get_stats(ts_b)
        ts["after_fft_stat"] = get_stats(ft_a)
        ts["before_fft_stat"] = get_stats(ft_b)
        with open(pp_file, 'wb') as f:
            pickle.dump(ts, f)
        return ts

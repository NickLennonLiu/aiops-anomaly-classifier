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

def get_kpi_at_time(data, cmdb_idx, t, window=(0,1)):
    t = int(t)
    return data[t+window[0]:t+window[1], cmdb_idx]


def gen_timeseries(cmdb_idx_dt, gt, data, window, kpi_set, dummy):
    dfs= []
    labels = []
    xs = []
    cnt = 0
    for idx, (x, y) in enumerate(zip(gt.x, gt.y)):
        cmdb_id, timestamp = x
        cmdb_idx = cmdb_idx_dt[cmdb_id]
        cmdb_idx_true = []
        for cidx in cmdb_idx:
            if cidx != dummy:
                cmdb_idx_true.append(cidx)
        time_window = get_kpi_at_time(data, cmdb_idx, timestamp, window)
        t = int(timestamp)
        if np.isnan(data[t + window[0]:t + window[1], cmdb_idx_true]).all():
            # print(f"skipping {idx} {x} {y}")
            continue
        df_a = DataFrame(time_window, columns=list(kpi_set))
        df_a.reset_index(inplace=True)
        df_a.rename(columns={"index":"timestamp"}, inplace=True)

        if df_a.values.shape[0] == (window[1] - window[0]):
            df_a["id"] = pd.Series([cnt for i in range(len(df_a.index))])
            cols = list(df_a.columns)
            cols = [cols[-1]] + cols[:-1]
            df_a = df_a[cols]

            # print(df_a.head())
            xs.append(x)
            labels.append(y)
            dfs.append(df_a)
            cnt += 1

    ts = pd.concat(dfs)
    y_df = DataFrame(labels, columns=["y"])
    return ts, y_df, xs

def get_timeseries(args, cmdb_kpi, data, gt, window):
    ts_file = os.path.join(args.workdir, f"{window}_timeseries.pkl")
    if not os.path.exists(ts_file):
        cmdb_set = list(set([ck[0] for ck in cmdb_kpi]))
        kpi_set = list(set([ck[1] for ck in cmdb_kpi]))
        kpi_set.sort()

        cmdb_idx_dt = get_cmdb_idx(cmdb_kpi, len(cmdb_kpi))
        data = np.pad(data, ((0, 0), (0, 1)))
        ts, y_df, xs = gen_timeseries(cmdb_idx_dt, gt, data, window, kpi_set, len(cmdb_kpi))

        # ts = ts.loc[:, (ts != ts.iloc[0]).any()]
        # ts.dropna(how='all', inplace=True, axis=1)

        const_columns = (ts != ts.iloc[0]).any()
        const_columns["timestamp"] = True
        ts = ts.loc[:, const_columns]
        ts.dropna(how='all', inplace=True, axis=1)

        timeseries = {
            "x": xs,
            "ts": ts,
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
        ft.loc[ft.id == i, ts.columns[2]:] = fft.astype(np.float)
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

def feature_extraction(data, cmdb_kpi, gt, args):
    pp_file = os.path.join(args.workdir, f"{(args.wstart, args.wstart+args.window)}_preprocess_data.pkl")
    if os.path.exists(pp_file):
        with open(pp_file, 'rb') as f:
            pp = pickle.load(f)
        return pp
    else:
        timeseries = get_timeseries(args, cmdb_kpi, data, gt, (args.wstart, args.wstart+args.window))
        ts = timeseries["ts"]
        y_df = timeseries["y"]
        ft = get_fft(ts)
        timeseries["fft"] = ft
        timeseries["stat"] = get_stats(ts).astype(np.float)
        timeseries["fft_stat"] = get_stats(ft).astype(np.float)
        with open(pp_file, 'wb') as f:
            pickle.dump(timeseries, f)
        return timeseries

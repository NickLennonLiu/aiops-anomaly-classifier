import os
import openpyxl
from itertools import islice

import pandas as pd
from pandas import DataFrame
from src.params import get_args
from src.preprocess import preprocess_dt


def load_gt(args):
    gt_path = args.gt_path
    wb = openpyxl.load_workbook(gt_path)
    ws = wb.active

    data = ws.values
    cols = next(data)[1:]
    data = list(data)
    idx = [r[0] for r in data]
    data = (islice(r, 1, None) for r in data)
    df = DataFrame(data, index=idx, columns=cols).drop("根因", axis=1) # Remove '根因' for simplicity
    return df

def load_dt(args):
    dt_path = args.dt_path
    data_dt = {}
    for file in os.listdir(dt_path):
        cmdb_id, kpi_name = os.path.splitext(file)[0].split('##')
        df = pd.read_csv(os.path.join(dt_path, file), index_col=0, usecols=[1,4])
        df.index = pd.to_datetime(df.index, unit="s")
        df = df.rename(columns={"value": kpi_name})
        if df.nunique().iloc[0] > 1:
            if cmdb_id not in data_dt.keys():
                data_dt[cmdb_id] = [df]
            else:
                data_dt[cmdb_id].append(df)
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
    dt_path = args.dt_path
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


if __name__ == "__main__":
    args = get_args()
    load_dt(args)
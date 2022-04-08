from datetime import datetime

import pandas as pd
from pandas import DataFrame

from src.params import get_args
import src.dataloader as dataloader

cause_id = {"网络丢包": 0,
             "网络延迟": 1,
             "JVM CPU负载高": 2,
             "CPU使用率高": 3,
             "磁盘空间使用率过高": 4,
             "内存使用率过高": 5,
             "JVM OOM Heap": 6,
             "磁盘IO读使用率过高": 7}

type_id = {"网络故障": 0,
           "应用故障": 1,
           "资源故障": 2, }

cause_types = ["网络故障", "应用故障", "资源故障"]

cause_to_type = [0, 0, 1, 2, 2, 2, 1, 2]

cause_literal = list(cause_id.keys())

def datetime_to_timestamp(datetime_str):
    """
    :param datetime_str:
    :return:
    example:    2021-02-26 03:55:02.253467
    """
    return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f").timestamp()

def preprocess_gt(df):
    df["故障类别"] = df["故障类别"].map(lambda x: type_id[x])
    df["故障内容"] = df["故障内容"].map(lambda x: cause_id[x])
    df['time'] = pd.to_datetime(df.time)
    return df

def preprocess_dt(df):
    df = df.drop(columns=['cmdb_id', 'kpi_name'])
    df.timestamp = pd.to_datetime(df.timestamp)
    return df


if __name__ == "__main__":
    args = get_args()
    gt = dataloader.load_gt(args)
    preprocess_gt(gt)
    print(gt.loc[1].time)

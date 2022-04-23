import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import src.dataloader as dataloader
from src.params import get_args

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

def preprocess_dt(data, start_time):
    """
    加载指标数据，
    1. 填充10Min内的空缺
    2. 时间轴平移到start_time为始的原点，单位长度min
    3. standard scaling
    最后合并到一个Dataframe
    """
    for key in data.keys():
        data[key] = data[key].resample("Min").mean()
        data[key].fillna(limit=10, method='ffill', inplace=True)

    for key, datas in data.items():
        datas.rename(columns={"value": f"{key[0]}##{key[1]}"}, inplace=True)
    data_l = list(data.values())
    key_l = list(data.keys())
    df = pd.concat(data_l, axis=1)
    df.index = [int((index.timestamp() - start_time) / 60) for index in df.index]
    # df.sort_index(inplace=True)
    df_tensor = torch.tensor(df.values)
    ss = StandardScaler()
    df_tensor = ss.fit_transform(df_tensor)
    return df_tensor, key_l

def preprocess_gt(df, start_time):
    """
    将时间轴平移到以start_time为原点，单位min
    """
    df.cmdb_id = list(zip(df.cmdb_id, df.time))
    df = df.rename(columns={"cmdb_id": "x", "故障内容": "y"})
    df.drop(['time'], axis=1, inplace=True)
    df.x = [(x[0], (x[1].timestamp() - start_time) / 60) for x in df.x]
    return df


if __name__ == "__main__":
    args = get_args()
    gt = dataloader.load_gt(args)
    print(gt.loc[1].time)

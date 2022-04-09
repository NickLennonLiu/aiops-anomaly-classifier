import os
import pickle
from os.path import exists

import openpyxl
import torch
from torch.utils.data import DataLoader, Dataset

from src import demo
from src.dataloader import load_gt, MyDataset, load_preprocessed, load_dt_raw
from src.params import get_args
from src.preprocess import preprocess_dt


def get_dataset(args):
    dataset = MyDataset(args)
    train_size,  valid_size = int(len(dataset) * 0.6), int(len(dataset) * 0.2)
    test_size = len(dataset) - train_size - valid_size
    train_set, valid_set, test_set = torch.utils.data.random_split(dataset,
                                                                   [train_size, valid_size, test_size],
                                                                   generator=torch.Generator().manual_seed(args.seed))
    return train_set, valid_set, test_set

def get_cmdb(args):
    data_filename = os.path.join(args.workdir, "data_pre.pkl")
    # 如果有预处理过的数据，那么直接读取
    if exists(data_filename):
        print(f"Loading preprocessed data from {data_filename}")
        data = load_preprocessed(data_filename)
    else:
        print(f"Loading raw data from {args.dt_raw}")
        data = load_dt_raw(args)
        preprocess_dt(data)
        print(f"Saving preprocessed data to {data_filename}")
        with open(data_filename, 'wb') as f:
            pickle.dump(data, f)
    return data

def main(args):
    train_set, valid_set, test_set = get_dataset(args)
    data = get_cmdb(args)
    if args.kpi_plot:
        demo.plot_kpi(args, data)


if __name__ == "__main__":
    _args = get_args("config/basic.yaml")
    # _args.kpi_plot = "../workdir/system_a/kpi_plot"
    main(_args)

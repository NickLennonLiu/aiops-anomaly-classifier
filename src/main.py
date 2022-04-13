import datetime
import os
import pickle
from os.path import exists

import numpy as np
import openpyxl
import sklearn.metrics
import torch
from sklearn.metrics import f1_score, accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from src import demo, model
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
        data, kpi_name = load_preprocessed(data_filename)
    else:
        print(f"Loading raw data from {args.dt_raw}")
        data = load_dt_raw(args)
        data, kpi_name = preprocess_dt(data, args.start_time)
        print(f"Saving preprocessed data to {data_filename}")
        with open(data_filename, 'wb') as f:
            pickle.dump([data, kpi_name], f)
    return data, kpi_name

def main(args):
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_set, valid_set, test_set = get_dataset(args)
    data, cmdb_kpi = get_cmdb(args)
    if args.kpi_plot:
        demo.plot_kpi(args, data)
    smodel = model.SingleCMDB_MLP(args, data, cmdb_kpi)
    train(args, smodel, train_set + valid_set)
    evaluate(args, smodel, test_set)


def evaluate(args, model, dataset):
    print(model.eval())
    result = []
    y_true = []
    for i, data in enumerate(dataset):
        inputs, labels = data
        outputs = model(inputs)
        result.append(np.argmax(outputs.detach().numpy()))
        y_true.append(np.argmax(labels))
    print(y_true)
    print(result)
    print(accuracy_score(y_true, result))
    print(f1_score(y_true, result, average=None))

def train(args, model, dataset):
    print(model.train())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)

    for epoch in range(args.epoch):
        start_time = datetime.datetime.now()
        running_loss = 0.0
        for i, data in enumerate(dataset):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, torch.tensor(labels).unsqueeze(dim=0))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"\r epoch {epoch}, running loss {running_loss}, used time {datetime.datetime.now() - start_time}", end='')
    print("Training finished")



if __name__ == "__main__":
    _args = get_args("config/system_a.yaml")
    # _args.kpi_plot = "../workdir/system_a/kpi_plot"
    main(_args)

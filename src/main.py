import datetime
import os
import pickle
from os.path import exists

import numpy as np
import openpyxl
import sklearn.metrics
import torch
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from src import demo
from src.dataloader import load_gt, MyDataset, load_preprocessed, load_dt_raw, TensorDataset
from src.params import get_args
from src.preprocess import preprocess_dt
import src.model as md


def get_dataset(args):
    stat_features = torch.load("/home/junetheriver/code/aiops/workdir/system_a/stat_feature_scaled.txt")
    y_df = pd.read_csv("/home/junetheriver/code/aiops/workdir/system_a/y.csv")
    dataset = TensorDataset(stat_features, y_df, 8)

    # dataset = MyDataset(args)

    # x = dataset.x
    # y = dataset.y
    # labels = [[] for i in range(args.class_num)]
    # for idx, label in enumerate(y):
    #     labels[label].append(idx)
    #
    # train_idx, valid_idx, test_idx = [], [], []
    # for i in range(args.class_num):
    #     np.random.shuffle(labels[i])
    #     train_idx += labels[i][:int(len(labels[i])*0.6)]
    #     valid_idx += labels[i][int(len(labels[i])*0.6):int(len(labels[i])*0.8)]
    #     test_idx += labels[i][int(len(labels[i])*0.8):]
    # np.random.shuffle(train_idx)
    # np.random.shuffle(valid_idx)
    # np.random.shuffle(test_idx)

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
    # smodel = model.SingleCMDB_MLP(args, data, cmdb_kpi)

    train(args, train_set+valid_set, test_set)
    evaluate(args, test_set, show=True)
    # evaluate(args, train_set+valid_set)


def evaluate(args, dataset, show=False):
    model.eval()
    result = []
    y_true = []
    with torch.no_grad():
        for i, data in enumerate(dataset):
            inputs, labels = data
            outputs = model(inputs)
            result.append(int(np.argmax(outputs.data)))
            y_true.append(np.argmax(labels))
    if show:
        print(y_true)
        print(result)
        print("micro: ", f1_score(y_true, result, average='micro'))
        print("macro: ", f1_score(y_true, result, average='macro'))
        print("weighted: ", f1_score(y_true, result, average='weighted'))
    print("test: accuracy ", accuracy_score(y_true, result))


def train(args, dataset, testset):
    print(np.bincount([np.argmax(data[1]) for data in dataset]))

    print(model.train())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)

    for epoch in range(args.epoch):
        running_loss = 0.0
        for i, data in enumerate(dataset):
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, torch.tensor(labels).unsqueeze(dim=0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if not epoch % 10:
            print(f"epoch {epoch}, running loss {running_loss}")
            evaluate(args, testset)
    print("Training finished")

model = md.MLP(679, 8)

if __name__ == "__main__":
    _args = get_args("config/system_a.yaml")
    # _args.kpi_plot = "../workdir/system_a/kpi_plot"
    main(_args)

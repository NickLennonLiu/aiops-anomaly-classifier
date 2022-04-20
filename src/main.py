import os
import pickle
from os.path import exists

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader

import src.dataloader as dataloader
import src.net as md
from src.feature_extraction import feature_extraction
from src.params import get_args
from src.preprocess import preprocess_gt


class Main:
    def __init__(self, args):
        torch.random.manual_seed(args.seed)
        np.random.seed(args.seed)

        self.train_set, self.test_set = self.get_dataset(args)

        self.model = md.Classifier(args)

        self.train(args)
        # (args, test_set, show=True)
        self.evaluate(args, show=True)

    def get_dataset(self, args):
        gt_pre = os.path.join(args.workdir, "gt_pre.pkl")
        if exists(gt_pre):
            gt = dataloader.load_preprocessed(gt_pre)
        else:
            gt = dataloader.load_gt(args)
            gt.cmdb_id = list(zip(gt.cmdb_id, gt.time))
            gt = gt.rename(columns={"cmdb_id": "x", "故障内容": "y"})
            gt.drop(['time'], axis=1, inplace=True)
            with open(gt_pre, 'wb') as f:
                pickle.dump(gt, f)
        gt = preprocess_gt(gt, args.start_time)
        # print(gt)
        data, cmdb_kpi = dataloader.get_cmdb(args)
        # print(len(cmdb_kpi), data.shape)
        raw_data = feature_extraction(data, cmdb_kpi, gt, args.window, args)
        dataset = dataloader.MyDataset2(raw_data)
        train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8),
                                                                      len(dataset) - int(len(dataset) * 0.8)])
        return train_set, test_set

    def evaluate(self, args, show=False):
        self.model.eval()
        result = []
        y_true = []
        with torch.no_grad():
            for i, data in enumerate(self.test_set):
                inputs, labels = data
                outputs = self.model(inputs)
                result.append(int(np.argmax(outputs.data)))
                y_true.append(np.argmax(labels))
        print("y_true :", ''.join(map(lambda x: str(x), y_true)))
        print("predict:", ''.join(map(lambda x: str(x), result)))
        if show:
            print("micro: ", f1_score(y_true, result, average='micro'))
            print("macro: ", f1_score(y_true, result, average='macro'))
            print("weighted: ", f1_score(y_true, result, average='weighted'))
        print("test: accuracy ", accuracy_score(y_true, result))

    def train(self, args):
        print(self.model.train())
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=args.lr)

        for epoch in range(args.epoch):
            running_loss = 0.0
            for i, data in enumerate(self.train_set):
                inputs, labels = data
                outputs = self.model(inputs)
                loss = criterion(outputs, torch.tensor(labels))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if not epoch % 10:
                print(f"epoch {epoch}, running loss {running_loss}")
                self.evaluate(args)
        print("Training finished")

if __name__ == "__main__":
    _args = get_args("config/system_a.yaml")
    # _args.kpi_plot = "../workdir/system_a/kpi_plot"
    main(_args)

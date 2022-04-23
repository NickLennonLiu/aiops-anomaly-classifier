import os
import pickle
from os.path import exists

import numpy as np
import sklearn.tree
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from torch import nn, optim
from torch.utils.data import DataLoader

import src.dataloader as dataloader
import src.net as md
from src.demo import plot_incident, plot_anomaly, plot_fft
from src.feature_extraction import feature_extraction
from src.params import get_args
from src.preprocess import preprocess_gt

class Demo:
    def __init__(self, args):
        self.gt = dataloader.get_gt(args)
        self.root_cause = dataloader.get_root_cause(args)
        self.data, self.cmdb_kpi = dataloader.get_cmdb(args)

        for l in range(8):
            for x, y in self.gt[self.gt.y == l].values[:1]:
                plot_fft(args, self.data, self.cmdb_kpi, self.root_cause,
                             y, x[0], x[1], "../temp/fft/")

class DT:
    def __init__(self, args):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        if not os.path.exists(os.path.join(args.workdir, args.tag)):
            os.makedirs(os.path.join(args.workdir, args.tag))

        self.gt = dataloader.get_gt(args)
        print(f"Loaded {len(self.gt)} ground truth points ")
        self.root_cause = dataloader.get_root_cause(args)
        self.data, self.cmdb_kpi = dataloader.get_cmdb(args)
        print(f"Loaded raw data with shape {self.data.shape}")
        self.preprocess_data = feature_extraction(self.data, self.cmdb_kpi, self.gt, args)
        self.train_set, self.test_set = self.get_dataset()

        self.classifier = DecisionTreeClassifier(max_depth=args.depth)

        self.train(args)
        with open(os.path.join(args.workdir, args.tag, "result.txt"), 'w') as f:
            self.eval(args, f)

    def get_dataset(self):
        dataset = dataloader.MyDataset2(self.preprocess_data)
        train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8),
                                                                      len(dataset) - int(len(dataset) * 0.8)])
        return train_set, test_set

    def train(self, args):
        X = []
        y = []
        for x, yy in self.train_set:
            X.append(torch.nan_to_num(x['stat']))
            y.append(np.argmax(yy))
        self.classifier.fit(torch.stack(X), y)

    def eval(self, args, file=None):
        X = []
        y = []
        for x, yy in self.test_set:
            # X.append(torch.nan_to_num(torch.concat([x['stat'], x['fft_stat']])))
            X.append(torch.nan_to_num(x['stat']))
            y.append(np.argmax(yy))
        result = self.classifier.predict(torch.stack(X))
        print(f1_score(result, y, average='macro'))
        print("y_true :", ''.join(map(lambda xx: str(xx), y)), file=file)
        print("predict:", ''.join(map(lambda xx: str(xx), result)), file=file)
        print("micro: ", f1_score(y, result, average='micro'), file=file)
        print("macro: ", f1_score(y, result, average='macro'), file=file)
        print("weighted: ", f1_score(y, result, average='weighted'), file=file)

class Main:
    def __init__(self, args):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        if not os.path.exists(os.path.join(args.workdir, args.tag)):
            os.makedirs(os.path.join(args.workdir, args.tag))

        self.gt = dataloader.get_gt(args)
        print(f"Loaded {len(self.gt)} ground truth points ")
        self.root_cause = dataloader.get_root_cause(args)
        self.data, self.cmdb_kpi = dataloader.get_cmdb(args)
        print(f"Loaded raw data with shape {self.data.shape}")
        self.preprocess_data = feature_extraction(self.data, self.cmdb_kpi, self.gt, args)
        print(f"Loaded preprocessed data.")
        args.ni = self.preprocess_data["ts"].shape[1] - 2

        self.train_set, self.test_set = self.get_dataset()

        if args.model == 'Classifier':
            args.nsi = self.preprocess_data["stat"].shape[1] + self.preprocess_data["fft_stat"].shape[1]
            self.model = md.Classifier(args)
        elif args.model == 'Classifier3':
            args.nsi = self.preprocess_data["stat"].shape[1]
            self.model = md.Classifier3(args)
        elif args.model == 'Stat':
            args.nsi = self.preprocess_data["stat"].shape[1]
            self.model = md.Stat(args)
        elif args.model == 'Classifier2':
            args.nsi = self.preprocess_data["stat"].shape[1]
            self.model = md.Classifier2(args)
        else:
            raise NotImplementedError

        fail = None
        if args.phase == 'full' or args.phase == 'train':
            self.train(args)
        if args.phase == 'full' or args.phase == 'eval':
            with open(os.path.join(args.workdir, args.tag, "result.txt"), 'w') as f:
                acc, fail = self.evaluate(args, show=True, file=f, load_weight=True)
        if args.show_fail and fail:
            fail_dir = os.path.join(args.workdir, args.tag, "fail_cases")
            if not os.path.exists(fail_dir):
                os.makedirs(fail_dir)
            for fail_case in fail:
                plot_incident(args, self.data, self.cmdb_kpi, self.root_cause,
                              fail_case[1], fail_case[2], fail_case[0][0], fail_case[0][1], fail_dir)

    def get_dataset(self):
        dataset = dataloader.MyDataset2(self.preprocess_data)
        train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8),
                                                                      len(dataset) - int(len(dataset) * 0.8)])
        return train_set, test_set

    def evaluate(self, args, show=False, file=None, load_weight=False):
        if load_weight and os.path.exists(os.path.join(args.workdir, args.tag, "model.pt")):
            print("Loading weight")
            self.model.load_state_dict(torch.load(os.path.join(args.workdir, args.tag, "model.pt")))
        self.model.eval()
        result = []
        y_true = []
        fail = []
        with torch.no_grad():
            for i, data in enumerate(self.test_set):
                inputs, labels = data
                outputs = self.model(inputs)
                result.append(int(np.argmax(outputs.data)))
                y_true.append(np.argmax(labels))
                if int(np.argmax(outputs.data)) != np.argmax(labels):
                    fail.append((inputs["x"], np.argmax(labels), int(np.argmax(outputs.data))))
        print("y_true :", ''.join(map(lambda x: str(x), y_true)), file=file)
        print("predict:", ''.join(map(lambda x: str(x), result)), file=file)
        if show:
            print("micro: ", f1_score(y_true, result, average='micro'), file=file)
            print("macro: ", f1_score(y_true, result, average='macro'), file=file)
            print("weighted: ", f1_score(y_true, result, average='weighted'), file=file)
        print("test: accuracy ", accuracy_score(y_true, result), file=file)
        return accuracy_score(y_true, result), fail

    def train(self, args):
        print(self.model.train())
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9)
        # optimizer = optim.Adagrad(self.model.parameters(), lr=args.lr) # ,

        best_acc = 0.0
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
                acc, fail = self.evaluate(args)
                if acc > best_acc:
                    torch.save(self.model.state_dict(), os.path.join(args.workdir, args.tag, "model.pt"))
        print("Training finished")

if __name__ == "__main__":
    _args = get_args("config/system_b.yaml")
    # _args.kpi_plot = "../workdir/system_a/kpi_plot"
    # main = Main(_args)
    Demo(_args)
    # dt = DT(_args)
    # dt.train(_args)
    # dt.eval(_args)

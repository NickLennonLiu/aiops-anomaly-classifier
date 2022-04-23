import os
import pickle
from os.path import exists
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as stats


def get_cmdb_dt(dt, cmdb_id):
    return {key[1]: dt[key] for key in dt.keys() if key[0] == cmdb_id}

def get_kpi_dt(dt, kpi_name):
    return {key[0]: dt[key] for key in dt.keys() if key[1] == kpi_name}

def get_kpi_at_time(data, cmdb_idx, t, window=1):
    t = int(t+0.5)
    return data[t:t+window, cmdb_idx]

def get_cmdb_all(data, cmdb_idx):
    return data[:, cmdb_idx]


def get_time_range(df):
    ranges = []
    start = None
    end = None
    for timestamp, value in zip(df.index, df.values.squeeze()):
        if not np.isnan(value):
            if start is None:
                start = timestamp
            end = timestamp
        elif start:
            ranges.append(pd.Interval(left=pd.Timestamp(start), right=pd.Timestamp(end)))
            start = None
            end = None
    if start:
        ranges.append(pd.Interval(start, end))
    return pd.arrays.IntervalArray(ranges)

class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.no = args.no   # class out
        self.nsi = args.nsi   # stat_in
        self.nso = args.nso   # stat_out

        self.ni = args.ni  # feature_in
        assert self.ni is not None, "You need to specify number of features inputted!"
        self.window = args.window
        self.conv_out, self.conv_kernel, \
            self.conv_stride, self.conv_pad = args.conv

        self.conv_out_size = self.conv_out * int((self.window - self.conv_kernel + self.conv_pad*2 + self.conv_stride)/ self.conv_stride)

        self.raw = nn.Sequential(
            SepConv1d(self.ni, self.conv_out, self.conv_kernel, self.conv_stride, self.conv_pad, drop=0.1), Flatten()
        )
        self.fft = nn.Sequential(
            SepConv1d(self.ni, self.conv_out, self.conv_kernel, self.conv_stride, self.conv_pad, drop=0.1), Flatten()
        )
        self.stat = nn.Sequential(
            nn.Linear(self.nsi, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, self.nso), nn.ReLU(), nn.Dropout(0.2),
        )

        self.out = nn.Sequential(
            nn.Linear(self.nso + self.conv_out_size*2, 64), nn.ReLU(),
            nn.Linear(64, 8), nn.ReLU(),
        )

    def forward(self, x):
        raw = x["ts"].unsqueeze(dim=0).permute((0,2,1)).to(torch.float32)
        fft = x["fft"].unsqueeze(dim=0).permute((0,2,1)).to(torch.float32)

        stat = torch.concat([x["stat"], x["fft_stat"]
                        ]).unsqueeze(dim=0).to(torch.float32) # x["after_fft_stat"]

        raw = torch.nan_to_num(raw)
        fft = torch.nan_to_num(fft)
        stat = torch.nan_to_num(stat)
        raw = self.raw(raw)
        fft = self.fft(fft)
        stat = self.stat(stat)
        x = torch.concat([raw, fft, stat], dim=1)
        return self.out(x)

class Classifier3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.no = args.no   # class out
        self.nsi = args.nsi   # stat_in
        self.nso = args.nso   # stat_out

        self.ni = args.ni  # feature_in
        assert self.ni is not None, "You need to specify number of features inputted!"
        self.window = args.window
        self.conv_out, self.conv_kernel, \
            self.conv_stride, self.conv_pad = args.conv

        self.conv_out_size = self.ni * int((self.window - self.conv_kernel + self.conv_pad*2 + self.conv_stride)/ self.conv_stride)

        self.raw = nn.Sequential(
            MyConv1d(self.ni, self.conv_out, self.conv_kernel, self.conv_stride, self.conv_pad, self.conv_out_size, drop=0.1), Flatten()
        )
        # self.fft = nn.Sequential(
        #     MyConv1d(self.ni, self.conv_out, self.conv_kernel, self.conv_stride, self.conv_pad, self.conv_out_size, drop=0.1), Flatten()
        # )
        self.stat = nn.Sequential(
            nn.Linear(self.nsi, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, self.nso), nn.ReLU(), nn.Dropout(0.2),
        )

        self.out = nn.Sequential(
            nn.Linear(self.nso + self.conv_out, 64), nn.ReLU(),
            nn.Linear(64, 8), nn.ReLU(),
        )

    def forward(self, x):
        raw = x["ts"].unsqueeze(dim=0).permute((0,2,1)).to(torch.float32)
        # fft = x["fft"].unsqueeze(dim=0).permute((0,2,1)).to(torch.float32)

        # stat = torch.concat([x["stat"], x["fft_stat"]
        #                 ]).unsqueeze(dim=0).to(torch.float32) # x["after_fft_stat"]
        stat = x["stat"].unsqueeze(dim=0).to(torch.float32) # x["after_fft_stat"]

        raw = torch.nan_to_num(raw)
        stat = torch.nan_to_num(stat)
        raw = self.raw(raw)
        # print(raw.shape)
        stat = self.stat(stat)
        x = torch.concat([raw, stat], dim=1)
        return self.out(x)

class Classifier2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.no = args.no   # class out
        self.nsi = args.nsi   # stat_in
        self.nso = args.nso   # stat_out

        self.ni = args.ni  # feature_in
        assert self.ni is not None, "You need to specify number of features inputted!"
        self.window = args.window
        self.conv_out, self.conv_kernel, \
            self.conv_stride, self.conv_pad = args.conv

        self.conv_out_size = self.conv_out * int((self.window - self.conv_kernel + self.conv_pad*2 + self.conv_stride)/ self.conv_stride)

        self.raw = nn.Sequential(
            SepConv1d(self.ni, self.conv_out, self.conv_kernel, self.conv_stride, self.conv_pad, drop=0.1), Flatten()
        )

        self.stat = nn.Sequential(
            nn.Linear(self.nsi, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, self.nso), nn.ReLU(), nn.Dropout(0.2),
        )

        self.out = nn.Sequential(
            nn.Linear(self.nso + self.conv_out_size, 64), nn.ReLU(),
            nn.Linear(64, 8), nn.ReLU(),
        )

    def forward(self, x):
        raw = x["ts"].unsqueeze(dim=0).permute((0, 2, 1)).to(torch.float32)
        stat = x["stat"].unsqueeze(dim=0).to(torch.float32)

        raw = torch.nan_to_num(raw)
        stat = torch.nan_to_num(stat)
        raw = self.raw(raw)
        stat = self.stat(stat)
        x = torch.concat([raw, stat], dim=1)
        return self.out(x)

class Stat(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.no = args.no   # class out
        self.nsi = args.nsi   # stat_in
        self.nso = args.nso   # stat_out

        self.stat = nn.Sequential(
            nn.Linear(self.nsi, 512), nn.ReLU(),
            nn.Linear(512, self.nso), nn.ReLU(),
        )

        self.out = nn.Sequential(
            nn.Linear(self.nso, 64), nn.ReLU(),
            nn.Linear(64, 8), nn.ReLU(),
        )

    def forward(self, x):
        stat = x["stat"].unsqueeze(dim=0).to(torch.float32) # x["after_fft_stat"]
        stat = torch.nan_to_num(stat)
        x = self.stat(stat)
        x = self.out(x)
        return x


class MLP8(nn.Module):
    def __init__(self, ni):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(ni, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512,160), nn.ReLU(), nn.Dropout(0.1)
        )
        self.l1 = nn.Sequential(
            nn.Linear(40, 2),
        )
        self.l2 = nn.Sequential(
            nn.Linear(80, 4),
        )
        self.l3 = nn.Sequential(
            nn.Linear(40, 2)
        )

    def forward(self, x):
        x = self.fc1(x)
        x1 = self.l1(x[:,:40])
        x2 = self.l2(x[:,40:120])
        x3 = self.l3(x[:,120:])
        x = torch.concat([x1, x2, x3], dim=1)
        x = x[:,[0,1,6,2,3,4,7,5]]
        return x

class _SepConv1d(nn.Module):
    """A simple separable convolution implementation.
    The separable convlution is a method to reduce number of the parameters
    in the deep learning network for slight decrease in predictions quality.
    """
    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()
        self.depthwise = nn.Conv1d(ni, ni, kernel, stride, padding=pad, groups=ni)
        self.pointwise = nn.Conv1d(ni, no, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class _MyConv1d(nn.Module):
    def __init__(self, ni, no, kernel, stride, pad, conv_out):
        super().__init__()
        self.depthwise = nn.Conv1d(ni, ni, kernel, stride, padding=pad, groups=ni)
        self.pointwise = nn.Linear(conv_out, no)

    def forward(self, x):
        return self.pointwise(self.depthwise(x).flatten().unsqueeze(dim=0))

class MyConv1d(nn.Module):
    def __init__(self, ni, no, kernel, stride, pad, conv_out, drop=None,
                 activ=lambda: nn.ReLU(inplace=True)):

        super().__init__()
        assert drop is None or (0.0 < drop < 1.0)
        layers = [_MyConv1d(ni, no, kernel, stride, pad, conv_out)]
        if activ:
            layers.append(activ())
        if drop is not None:
            layers.append(nn.Dropout(drop))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        return self.layers(x)

class SepConv1d(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.
    The module adds (optionally) activation function and dropout layers right after
    a separable convolution layer.
    """
    def __init__(self, ni, no, kernel, stride, pad, drop=None,
                 activ=lambda: nn.ReLU(inplace=True)):

        super().__init__()
        assert drop is None or (0.0 < drop < 1.0)
        layers = [_SepConv1d(ni, no, kernel, stride, pad)]
        if activ:
            layers.append(activ())
        if drop is not None:
            layers.append(nn.Dropout(drop))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        return self.layers(x)

class Flatten(nn.Module):
    """Converts N-dimensional tensor into 'flat' one."""

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.view(x.size(0), -1)
        return x.view(-1)

class Basic(nn.Module):
    def __init__(self, args, data, cmdb_kpi):
        super().__init__()

        self.data = F.pad(torch.nan_to_num(data), pad=(0,1)).to(torch.float32)
        self.cmdb_kpi = cmdb_kpi
        self.metric_num = len(cmdb_kpi)
        self.cmdb_idx = self.get_cmdb_idx()
        self.kpi_set = set([ck[1] for ck in self.cmdb_kpi])
        self.kpi_num = len(self.kpi_set)

        self.class_num = args.class_num
        self.workdir = args.workdir
        self.window = args.window

        self.stats = self.get_describe() # std mean median mode

        print(f"metric_num: {self.metric_num}, kpi_num: {self.kpi_num}, {self.data.shape}")

    def get_cmdb_idx(self):
        cmdb_set = list(set([ck[0] for ck in self.cmdb_kpi]))
        kpi_set = list(set([ck[1] for ck in self.cmdb_kpi]))

        cmdb_idx = {}
        for cmdb in cmdb_set:
            a = []
            for idx, kpi in enumerate(kpi_set):
                try:
                    found = self.cmdb_kpi.index((cmdb,kpi))
                    a.append(found)
                except ValueError:
                    a.append(self.metric_num)
            cmdb_idx[cmdb] = a
        return cmdb_idx

    def get_describe(self):
        filename = os.path.join(self.workdir, "data_stats.pkl")
        if exists(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            std, mean = torch.std_mean(self.data, dim=0)
            median = torch.median(self.data, dim=0)[0]
            mode = torch.mode(self.data,dim=0)[0]
            des = torch.stack([std, mean, median, mode])
            with open(filename, 'wb') as f:
                pickle.dump(des, f)
            return des

class SingleCMDB_MLP(Basic):
    def __init__(self, args, data, kpi_name):
        super().__init__(args, data, kpi_name)

        self.dropout = nn.Dropout(p=0.1)

        drop=.1

        self.raw = nn.Sequential(
            SepConv1d(self.kpi_num, 32, 8, 2, 3, drop=drop),
            SepConv1d(32, 64, 8, 4, 2, drop=drop),
            # SepConv1d(64, 128, 8, 4, 2, drop=drop),
            # SepConv1d(128, 256, 8, 4, 2),
            Flatten(),
            # nn.Dropout(drop), nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear(64, 64), nn.ReLU(inplace=True))

        self.out = nn.Sequential(
            nn.Linear(64, self.class_num), nn.ReLU(inplace=True)) #nn.Linear(128, 64),

    def forward(self, x):
        cmdb_id, timestamp = x
        cmdb_idx = self.cmdb_idx[cmdb_id]
        x = get_kpi_at_time(self.data, cmdb_idx, timestamp, window=self.window) # time_len x metrics
        x = torch.t(x).unsqueeze(dim=0)
        x = F.normalize(x, dim=1)
        x = self.raw(x)
        x = self.out(x)
        return x

class KStest(Basic):
    def __init__(self, args, data, kpi_name):
        super().__init__(args, data, kpi_name)
        self.fc1 = nn.Linear(self.kpi_num, self.class_num) # self.data.shape[1]
        #self.fc2 = nn.Linear(200, 100)
        #self.fc3 = nn.Linear(100, self.class_num)

    def forward(self, x):
        cmdb_id, timestamp = x
        cmdb_idx = self.cmdb_idx[cmdb_id]
        x = get_kpi_at_time(self.data, cmdb_idx, timestamp, window=self.window) # time_len x metrics
        y = get_kpi_at_time(self.data, cmdb_idx, timestamp-self.window, window=self.window*2)
        # x = torch.tensor(np.argsort([stats.ks_2samp(x[:,i], y[:,i])[1] for i in range(self.data.shape[1])])[:500]).to(torch.float32)
        x = torch.tensor([stats.ks_2samp(x[:,i], y[:,i])[1] for i in range(self.kpi_num)]).to(torch.float32)
        # y = np.divide(np.abs(np.subtract(x[-1,:], self.stats[1, :])), self.stats[0, :]).float().unsqueeze(dim=0)
        x = x.unsqueeze(dim=0)
        x = F.leaky_relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        return x



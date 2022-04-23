import os

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame

from src.dataloader import get_cmdb_kpi_dict, get_cmdb_kpi
from src.net import get_kpi_at_time
from src.params import get_args

cause_id = {"网络丢包": 0,
             "网络延迟": 1,
             "JVM CPU负载高": 2,
             "CPU使用率高": 3,
             "磁盘空间使用率过高": 4,
             "内存使用率过高": 5,
             "JVM OOM Heap": 6,
             "磁盘IO读使用率过高": 7}
cause_eng = {
             "网络丢包": "Network PackLost",
             "网络延迟": "Network Delay",
             "JVM CPU负载高": "JVM CPU Overload",
             "CPU使用率高": "CPU Usage",
             "磁盘空间使用率过高": "Disk Space Usage",
             "内存使用率过高": "Memory Usage",
             "JVM OOM Heap": "JVM OOM Heap",
             "磁盘IO读使用率过高": "Disk IO Usage"
}
cause_literal = list(cause_id.keys())

def show_time_series(df: DataFrame, filename=None):
    plt.plot(df)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

def plot_kpi(args, data):
    assert(os.path.isdir(args.kpi_plot))
    if not os.path.exists(args.kpi_plot):
        os.makedirs(args.kpi_plot)
    for key in data.keys():
        filename = os.path.join(args.kpi_plot, f"{key[0]}##{key[1]}.png")
        show_time_series(data[key], filename)

def plot_fft(args, data, cmdb_kpi, root_cause, label, cmdb, occur_time, save_dir):
    tag = f"{label}_{cmdb}_{int(occur_time)}"

    legends = []
    stds = []
    tss = []
    for cause in root_cause[label]:
        for filename in os.listdir(args.dt_raw):
            if cause in filename and cmdb in filename:
                kpi = '.'.join(filename.split('##')[1].split('.')[:-1])
                if kpi in legends:
                    continue
                try:
                    idx = cmdb_kpi.index((cmdb, kpi))
                except ValueError:
                    continue
                ts = get_kpi_at_time(data, idx, occur_time - 10, 20)
                if ts.std() < 1e-2:
                    continue
                stds.append(ts.std())
                tss.append(ts)
                legends.append(kpi)
    max5 = np.argmax(stds)
    new_legend = []

    plt.plot(tss[max5])
    # plt.legend(new_legend)
    # plt.title(f"{cmdb}_{cause_eng[cause_literal[label]]}")
    plt.show()
    plt.plot(np.abs(np.fft.fft(tss[max5])))
    plt.show()

def plot_anomaly(args, data, cmdb_kpi, root_cause, label, cmdb, occur_time, save_dir):
    tag = f"{label}_{cmdb}_{int(occur_time)}"

    legends = []
    stds = []
    tss = []
    for cause in root_cause[label]:
        for filename in os.listdir(args.dt_raw):
            if cause in filename and cmdb in filename:
                kpi = '.'.join(filename.split('##')[1].split('.')[:-1])
                if kpi in legends:
                    continue
                try:
                    idx = cmdb_kpi.index((cmdb, kpi))
                except ValueError:
                    continue
                ts = get_kpi_at_time(data, idx, occur_time - 10, 20)
                if ts.std() < 1e-2:
                    continue
                stds.append(-1 * ts.std())
                tss.append(ts)
                legends.append(kpi)
    max5 = np.argpartition(stds, kth=min(5,len(stds)-1))[:min(5,len(stds))]
    new_legend = []
    for idx, ts in enumerate(tss):
        if idx in max5:
            plt.plot(ts)
            new_legend.append(legends[idx])
    plt.legend(new_legend)
    plt.title(f"{cmdb}_{cause_eng[cause_literal[label]]}")

    if save_dir:
        plt.savefig(os.path.join(save_dir, tag+".png"))
    else:
        plt.show()
    # plt.show()
    plt.close()

def plot_incident(args, data, cmdb_kpi, root_cause, label, predict, cmdb, occur_time, save_dir):
    tag = f"{label}_{cmdb}_{int(occur_time)}_{predict}"

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,8))

    for aidx, l in enumerate([label, predict]):
        legends = []
        stds = []
        tss = []
        for cause in root_cause[l]:
            for filename in os.listdir(args.dt_raw):
                if cause in filename and cmdb in filename:
                    kpi = '.'.join(filename.split('##')[1].split('.')[:-1])
                    if kpi in legends:
                        continue
                    try:
                        idx = cmdb_kpi.index((cmdb, kpi))
                    except ValueError:
                        continue
                    ts = get_kpi_at_time(data, idx, occur_time - 10, 20)
                    if ts.std() < 1e-2:
                        continue
                    stds.append(ts.std())
                    tss.append(ts)
                    legends.append(kpi)
        max5 = np.argpartition(stds, kth=5)[:5]
        new_l = []
        for idx, ts in enumerate(tss):
            if idx in max5:
                axes[aidx].plot(ts)
                new_l.append(legends[idx])
        axes[aidx].legend(new_l)
        axes[aidx].set_title(cause_eng[cause_literal[l]])

    if save_dir:
        fig.savefig(os.path.join(save_dir, tag+".png"))
    else:
        fig.show()
    # plt.show()
    plt.close(fig)

if __name__ == "__main__":
    args = get_args()
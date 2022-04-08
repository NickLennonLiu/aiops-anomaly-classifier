import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Timestamp

import src.dataloader as dataloader
from src import params, preprocess


def time_span(dt_path):
    timestamp_set = set()
    for file in os.listdir(dt_path):
        df = pd.read_csv(os.path.join(dt_path, file))
        for t in df.timestamp.values:
            timestamp_set.add(t)
    timestamp_list = list(timestamp_set)
    timestamp_list.sort()
    return timestamp_list

def timestamp_to_pixel(start, timestamp):
    return int((timestamp - start) / 60)

def draw_timestamp_distrib(dt_path, cmdb_kpi, start, size, save=None):
    grid = np.zeros(size)
    row_id = 0
    colors = [100, 200, 255]
    color_idx = 0
    print(len(cmdb_kpi.keys()))
    for cmdb_id in cmdb_kpi.keys():
        print(cmdb_id)
        for kpi_name in cmdb_kpi[cmdb_id]:
            file = os.path.join(dt_path, f"{cmdb_id}##{kpi_name}.csv")
            ts = list(map(lambda x : timestamp_to_pixel(start, x), pd.read_csv(file).timestamp.values))
            for t in ts:
                grid[row_id][t] = colors[color_idx]
            row_id += 1
        color_idx = (color_idx+1) % 3
    np.save("../workdir/grid.npy", grid)
    plt.matshow(grid)

def test(gt, start):
    grid = np.load("../workdir/grid.npy")
    # plt.imshow(grid, cmap="gray", vmin=0, vmax=255, aspect=10)
    # plt.show()
    ts = list(map(lambda x: timestamp_to_pixel(start, x), pd.to_numeric(gt.time) // 10 ** 9))
    print(ts)
    plt.imshow(grid, cmap="gray", vmin=0, vmax=255, aspect=10)
    for t in ts:
        plt.axline((t, 0), (t,1), alpha=0.5)
    plt.xlabel("t - t0 (s)")
    plt.ylabel("cmdb_kpi")
    plt.show()


def describe(dt):
    """
    :param dt: {cmdb_id: [kpi_df]}
    :return: {cmdb_id: np.array(len(kpi), 7)}
    (mean, std, min, 25%, 50%, 75%, max)
    """
    dt = dt.copy()
    for cmdb_id in list(dt.keys()):
        statistical = np.stack([df.describe()[1:] for df in dt[cmdb_id]], axis=2).squeeze().transpose()
        dt[cmdb_id] = statistical
    return dt


if __name__ == "__main__":
    # tl = time_span("../data/system-a")
    # print(len(tl), min(tl), max(tl))
    # print((max(tl) - min(tl)) / 60)
    args = params.get_args()
    gt = dataloader.load_gt(args)
    gt = preprocess.preprocess_gt(gt)
    start = 1614268800
    test(gt, start)
    # dt_path = "../data/system-a"
    # cmdb_kpi = dataloader.get_cmdb_kpi_tuple(args)
    # draw_timestamp_distrib(dt_path, cmdb_kpi, start, (3300, 38900))
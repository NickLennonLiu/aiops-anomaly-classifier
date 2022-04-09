import os

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame

from src.dataloader import get_cmdb_kpi_dict, get_cmdb_kpi
from src.params import get_args


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


if __name__ == "__main__":
    args = get_args()
    cmdb_kpi = get_cmdb_kpi_dict(args)
    cmdb = list(cmdb_kpi.keys())[0]
    kpi = cmdb_kpi[cmdb][0]
    df = get_cmdb_kpi(args, cmdb, kpi)
    print(df.head())
    show_time_series(df)
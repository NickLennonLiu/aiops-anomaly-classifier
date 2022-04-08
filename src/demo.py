import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from src.dataloader import get_cmdb_kpi_dict, get_cmdb_kpi
from src.params import get_args


def show_time_series(df):
    plt.plot(df.timestamp, df.value)
    plt.show()


if __name__ == "__main__":
    args = get_args()
    cmdb_kpi = get_cmdb_kpi_dict(args)
    cmdb = list(cmdb_kpi.keys())[0]
    kpi = cmdb_kpi[cmdb][0]
    df = get_cmdb_kpi(args, cmdb, kpi)
    print(df.head())
    show_time_series(df)
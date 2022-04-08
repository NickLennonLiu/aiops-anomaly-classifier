import numpy as np
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.label_num = args.label_num
        self.instance_num = args.instance_num
        self.kpi_num = args.kpi_num

    def forward(self, x):
        """
        :param x:  (time, cmdb_id)
        :return:   anomaly type in range(8)
        """
        return np.random.randint(0, self.label_num)


class Statistic(Model):
    def __init__(self, args):
        super().__init__(args)




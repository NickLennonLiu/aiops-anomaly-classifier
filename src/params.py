import argparse
import yaml

def get_default_args():
    parser = get_parser()
    p, unknown = parser.parse_known_args([])

    if p.config is not None:
        # load config file
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        # update parser from config file
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('Unknown Arguments: {}'.format(k))
                assert k in key
        parser.set_defaults(**default_arg)
    return parser.parse_args([])

def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

def get_parser():
    parser = argparse.ArgumentParser(description='Anomaly Classifier')
    parser.add_argument('--model', default="Classifier")
    parser.add_argument('--phase', default='full', choices=['full', 'train', 'eval'])

    parser.add_argument('--config', default=None)
    parser.add_argument('--seed', default=5)
    parser.add_argument('--gt_path', default='/home/junetheriver/code/aiops/data/groundtruth_a.xlsx')
    parser.add_argument('--dt_raw', default='/home/junetheriver/code/aiops/data/system-a/')
    parser.add_argument('--kpi_plot', default=None)
    parser.add_argument('--class_num', default=8)
    parser.add_argument('--epoch', default=200)
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--start_time')
    parser.add_argument('--time_range')

    parser.add_argument('--wstart', default=0)
    parser.add_argument('--window', default=10)

    parser.add_argument('--no', default=8)
    parser.add_argument('--nsi', default=None)
    parser.add_argument('--nso', default=None)
    parser.add_argument('--ni', default=None)
    parser.add_argument('--conv', type=tuple_type, default=None)

    parser.add_argument('--workdir', default='/home/junetheriver/code/aiops/workdir/system_a')
    parser.add_argument('--tag', default="")
    parser.add_argument('--show_fail', type=str, default=None)

    parser.add_argument('--depth', default=20)
    return parser


def get_args(config_file=None):
    parser = get_parser()

    # load arg form config file
    # p = parser.parse_args()
    p, unknown = parser.parse_known_args()
    if p.config is not None or config_file is not None:
        # load config file
        if config_file is not None:
            p.config = config_file
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        # update parser from config file
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('Unknown Arguments: {}'.format(k))
                assert k in key
        parser.set_defaults(**default_arg)
    p, unknown = parser.parse_known_args()
    return p

if __name__ == "__main__":
    get_parser().print_help()
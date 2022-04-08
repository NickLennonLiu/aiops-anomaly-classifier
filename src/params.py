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

def get_parser():
    parser = argparse.ArgumentParser(description='Anomaly Classifier')

    parser.add_argument('--config', default=None)
    parser.add_argument('--gt_path', default='/home/junetheriver/code/aiops/data/groundtruth_a.xlsx')
    parser.add_argument('--dt_path', default='/home/junetheriver/code/aiops/data/system-a/')

    return parser


def get_args():
    parser = get_parser()

    # load arg form config file
    # p = parser.parse_args()
    p, unknown = parser.parse_known_args()
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
    p, unknown = parser.parse_known_args()
    return p

if __name__ == "__main__":
    get_parser().print_help()
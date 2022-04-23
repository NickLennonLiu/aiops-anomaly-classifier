from src.main import Main, DT
from src.params import get_args

def test1(args):
    args.wstart = -5
    args.window = 10
    args.tag = "w=(-5,10)"
    Main(args)

def test2(args):
    args.wstart = 0
    args.window = 5
    args.tag = "w=(0,5)"
    Main(args)

def test3(args):
    args.wstart = 0
    args.window = 10
    args.model = "Stat"
    args.tag = "w=(0,10),stat"
    Main(args)

def test4(args):
    args.wstart = 0
    args.window = 10
    args.model = "Classifier2"
    args.tag = "w=(0,10),c2"
    Main(args)

def test_window(args, tag):
    window_option = [
        (0,1),
        (0,5),
        (0,10),
        (-5,10),
        (-2,5),
        (-2,10)
    ]
    for w in window_option:
        try:
            args.wstart = w[0]
            args.window = w[1]
            args.tag = tag + f"_w={w}"
            Main(args)
        except Exception as err:
            print(f"\n!!!!!!!!!! An error occurred: {err}\n")
            continue


def test_dt_window_depth(args, tag):
    window_option = [
        (0,1),
        (0,5),
        (0,10),
        (-5,10),
        (-2,5),
        (-2,10)
    ]
    for depth in [6,8,10,12]:
        for w in window_option:
            try:
                print(depth, w)
                args.depth = depth
                args.wstart = w[0]
                args.window = w[1]
                args.tag = tag + f"_w={w}_d={depth}"
                DT(args)
            except Exception as err:
                print(f"\n!!!!!!!!!! An error occurred: {err}\n")
                continue



def test_pipeline():
    # Stat
    args = get_args("config/system_b_stat.yaml")
    test_window(args, "s=0,stat")
    # Classifier2
    args = get_args("config/system_b_raw.yaml")
    test_window(args, "s=0,c2")
    # Classifier
    args = get_args("config/system_b.yaml")
    test_window(args, "s=0,c")

def test_dt():
    args = get_args("config/system_a.yaml")
    test_dt_window_depth(args, "dt")


if __name__ == "__main__":
    # _args = get_args("config/system_a.yaml")
    # # test1(_args)
    # test3(_args)
    test_dt()
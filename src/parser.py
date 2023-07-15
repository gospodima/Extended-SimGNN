import argparse


def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser(description="Run SimGNN.")  # 创建一个命令行参数解析器

    parser.add_argument(  # 向解析器添加了一个命令行参数的定义
        "--dataset",  # 参数名，双减号表示这是一个长参数
        nargs="?",  # 参数后面可以跟一个值，这个值可以存在也可以不存在。
        default="AIDS700nef",  # 如果不提供该参数，则默认为 "AIDS700nef"。
        help="Dataset name. Default is AIDS700nef",
    )

    parser.add_argument(
        "--gnn-operator",
        nargs="?",
        default="gcn",
        help="Type of GNN-Operator. Default is gcn",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=350,
        help="Number of training epochs. Default is 350.",
    )

    parser.add_argument(
        "--filters-1",
        type=int,
        default=64,
        help="Filters (neurons) in 1st convolution. Default is 64.",
    )

    parser.add_argument(
        "--filters-2",
        type=int,
        default=32,
        help="Filters (neurons) in 2nd convolution. Default is 32.",
    )

    parser.add_argument(
        "--filters-3",
        type=int,
        default=16,
        help="Filters (neurons) in 3rd convolution. Default is 16.",
    )

    parser.add_argument(
        "--tensor-neurons",
        type=int,
        default=16,
        help="Neurons in tensor network layer. Default is 16.",
    )

    parser.add_argument(
        "--bottle-neck-neurons",
        type=int,
        default=16,
        help="Bottle neck layer neurons. Default is 16.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of graph pairs per batch. Default is 128.",
    )

    parser.add_argument(
        "--bins", type=int, default=16, help="Similarity score bins. Default is 16."
    )

    parser.add_argument(
        "--dropout", type=float, default=0, help="Dropout probability. Default is 0."
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate. Default is 0.001.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5 * 10 ** -4,
        help="Adam weight decay. Default is 5*10^-4.",
    )

    parser.add_argument("--histogram", dest="histogram", action="store_true")

    parser.add_argument(
        "--diffpool",
        dest="diffpool",    # 参数的目标属性名称，将该参数的值存储到 args.diffpool 属性中
                            # 如果没有指定 dest 参数，argparse 默认会根据命令行参数的名称生成目标属性名称。
                            # 1. 对于长参数名（以 -- 开头），argparse 会将前缀 -- 去除，并将剩余部分的连字符（-）替换为下划线（_）作为目标属性的名称。
                            # 2. 对于短参数名（以 - 开头），argparse 会将前缀 - 去除，并将剩余部分的每个字符都作为目标属性的名称。
        action="store_true",    #参数的动作，设置为 store_true 表示如果命令行中出现了该参数，则将其设置为 True。
        help="Enable differentiable pooling.",
    )

    parser.add_argument("--plot", dest="plot", action="store_true")

    parser.add_argument("--synth", dest="synth", action="store_true")

    parser.add_argument("--save",
                        type=str,
                        default=None,
                        help="Path to save the trained model")

    parser.add_argument("--load",
                        type=str,
                        default=None,
                        help="Path to load a pretrained model")

    parser.add_argument(
        "--measure-time",
        action="store_true",
        help="Measure average calculation time for one graph pair. Learning is not performed.",
    )

    parser.add_argument(
        "--notify",
        dest="notify",
        action="store_true",
        help="Send notification message when the code is finished (only Linux & Mac OS support).",
    )

    # 设置默认参数
    parser.set_defaults(histogram=False)
    parser.set_defaults(diffpool=False)
    parser.set_defaults(plot=False)
    parser.set_defaults(measure_time=False)
    parser.set_defaults(notify=False)
    parser.set_defaults(synth=False)

    return parser.parse_args()

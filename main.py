from src.utils import tab_printer
from src.simgnn import SimGNNTrainer
from src.parser import parameter_parser


def main():
    """
    Parsing command line parameters, reading data, fitting and scoring a SimGNN model.
    """
    args = parameter_parser()  # 解析命令行输入的参数
    tab_printer(args)  # 以表格的形式打印参数
    trainer = SimGNNTrainer(args)  # 构建SimGNNTrainer类
    # 从下面的几个命令开始，根据不同的参数设置调用SimGNN类的实例的forward函数
    if args.measure_time:
        trainer.measure_time()  # Measure average calculation time for one graph pair
    else:
        if args.load:
            trainer.load()      # Load a pretrained model
        else:
            trainer.fit()   # training a model
        trainer.score()
        if args.save:   # Store the model. Default is None.
            trainer.save()

    if args.notify:     # 是否需要发送通知，根据操作系统的不同，使用不同的方法来发送通知
        import os
        import sys

        if sys.platform == "linux":     #Linux操作系统
            os.system('notify-send SimGNN "Program is finished."')
        elif sys.platform == "posix":   #macOS操作系统
            os.system(
                """
                      osascript -e 'display notification "SimGNN" with title "Program is finished."'
                      """
            )
        else:
            raise NotImplementedError("No notification support for this OS.")


if __name__ == "__main__":
    main()

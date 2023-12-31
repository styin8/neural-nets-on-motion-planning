import argparse
import os
import torch


class BaseOptions():
    def __init__(self) -> None:
        self.initialized = False

    def initialize(self, parser):

        parser.add_argument("--dataroot", default="", help="path to data")
        parser.add_argument("--num_threads", type=int,
                            default=8, help="num_threads")
        parser.add_argument("--train_test_split", default="0.9",
                            type=float, help="train test split rate")
        parser.add_argument("--train_val_split", default="0.9",
                            type=float, help="train val split rate")
        parser.add_argument("--checkpoints_dir", type=str,
                            default="./checkpoints", help="models are saved here")
        parser.add_argument(
            "--name", type=str, default="copyright@1st.", help="the name of experience")

        parser.add_argument("--gpu_ids", default="-1", help="use -1 for cpu")
        parser.add_argument("--model", type=str,
                            default="cfn", help="the name of model")
        parser.add_argument("--threshold", type=float, default=0.02, help="m")
        parser.add_argument("--save_net_freq", type=int, default=20,
                            help="save model every save_net_freq epoch")

        self.initialized = True
        return parser

    def print_options(self, opt):
        message = ""
        message += "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:<25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

        # save in the disk
        save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, f"{opt.name}opt.txt")
        with open(file_name, "wt") as f:
            f.write(message)
            f.write("\n")

    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser
        opt = parser.parse_args()
        opt.isTrain = self.isTrain

        # print options
        self.print_options(opt)

        self.opt = opt
        return self.opt

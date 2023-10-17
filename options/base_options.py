import argparse
import os
import torch


class BaseOptions():
    def __init__(self) -> None:
        self.initialized = False

    def initialize(self, parser):

        parser.add_argument("--dataroot", required=True, help="path to data")
        parser.add_argument(
            "--name", type=str, default="copyright@1st.", help="the name of experience")
        parser.add_argument("--gpu_ids", default="0", help="use -1 for cpu")
        parser.add_argument("--batch_size", type=int,
                            default=1, help="input batch size")
        parser.add_argument("--checkpoints_dir", type=str,
                            default="./checkpoints", help="models are saved here")

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
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

        # save in the disk
        save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, f"{opt.name}opt.txt")
        with open(file_name,"wt") as f:
            f.write(message)
            f.write("\n")

from .base_options import BaseOptions


class TrainOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument("--lr", type=float, default=0.002,
                            help="initial learning rate")
        parser.add_argument("--dropout", type=float, default=0.5,
                            help="drop out rate")
        parser.add_argument("--epochs", type=int,
                            default=1000, help="number of epochs")
        parser.add_argument("--batch_size", type=int,
                            default=30, help="input batch size")
        parser.add_argument("--dof", type=int, default=6,
                            help="number of degrees of freedom")
        parser.add_argument("--L", type=int, default=3,
                            help="number of sin cos kernel")
        parser.add_argument("--hidden", type=int, default=1024,
                            help="number of hidden units")
        parser.add_argument("--voxel", type=int,
                            default=3456, help="batch size")
        parser.add_argument("--init_weight", type=bool,
                            default=True, help="initial weight")

        self.isTrain = True
        return parser

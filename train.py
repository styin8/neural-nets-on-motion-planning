from options.train_options import TrainOptions
from models import create_model
from data import create_dataset
import tensorboardX
import os
import torch





if __name__ == "__main__":
    opt = TrainOptions().parse()
    tb = tensorboardX.SummaryWriter(
        os.path.join(opt.checkpoints_dir, opt.name))

    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f"The number of training images = {dataset_size}")

    model = create_model(opt)
    model.to(torch.device(f"cuda:{opt.gpu_ids}" if torch.cuda.is_available() else "cpu"))

    # best_
    for epoch in range(opt.epochs):
        print(f'Beginning Epoch {epoch:02d}')
        # TODO
        model.net_train(dataset, opt)

        tb.add_scalar('train_loss', ['loss'], epoch)

        #

from options.train_options import TrainOptions
from models import create_model
from data import create_dataset
import tensorboardX
import os
import torch
import torch.optim as optim


if __name__ == "__main__":
    opt = TrainOptions().parse()
    tb = tensorboardX.SummaryWriter(
        os.path.join(opt.checkpoints_dir, opt.name))

    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f"The number of training images = {dataset_size}")

    model = create_model(opt)
    device = torch.device(
        f"cuda:{opt.gpu_ids}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters())

    best_acc = 0
    for epoch in range(opt.epochs):
        print(f'Beginning Epoch {epoch:02d}')
        # TODO
        loss = model.net_train(dataset, device, optimizer, opt)

        tb.add_scalar('train_loss', loss, epoch)

        acc = model.net_vaild(dataset, device, opt)

        tb.add_scalar('valid_acc', acc, epoch)

        if epoch % opt.save_net_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print(f'saving the model at the end of {epoch}')
            model.save_networks(epoch)

        if acc > best_acc:
            print(f"saving the model at current best acc is {acc}")
            best_acc = acc
            model.save_networks(f"best_{acc}")

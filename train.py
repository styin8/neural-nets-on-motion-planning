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
        os.path.join(opt.checkpoints_dir, opt.name,"tensorboard"))

    train_dataset = create_dataset(opt, mode="train")
    train_dataset_size = len(train_dataset)
    print('The number of training images = %d' % train_dataset_size)

    val_dataset = create_dataset(opt, mode="val")
    val_dataset_size = len(val_dataset)
    print('The number of val images = %d' % val_dataset_size)

    model = create_model(opt)
    device = torch.device(
        f"cuda:{opt.gpu_ids}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters())

    best_acc = 0
    for epoch in range(opt.epochs):
        print(f'Beginning Epoch {epoch}')

        loss = model.net_train(train_dataset, device, optimizer, opt, epoch)

        tb.add_scalar('train_loss', loss, epoch)

        acc = model.net_vaild(val_dataset, device, opt, epoch)

        print(f'Epoch: {epoch}, Val Avg Acc:{acc}')

        tb.add_scalar('valid_acc', acc, epoch)

        if epoch % opt.save_net_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print(f'saving the model at the end of {epoch}')
            model.save_networks(epoch, acc)
            continue

        if acc > best_acc:
            print(f"saving the model at current best acc is {acc}")
            best_acc = acc
            model.save_networks(epoch, acc)

import importlib
import torch.nn as nn
import os


def find_model_using_name(model_name):
    """
    """
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    for name, cls in modellib.__dict__.items():
        if name.lower() == model_name.lower() and issubclass(cls, nn.Module):
            model = cls
    if model is None:
        print(
            f"In {model_name}.py, there are no subclasses of nn.Module that match the {model_name}!")
    return model


def print_network(net, opt):
    message = ""
    message += '--------------- Networks ----------------\n'
    for _, module in net.named_modules():
        if module.__class__.__name__ != net.__class__.__name__:
            message += '{:<25}: {:<30}\n'.format(str(module.__class__.__name__), str(
                sum(p.numel() for p in module.parameters())))
    message += '-----------------------------------------\n'
    message += f'Total number of parameters : {sum(p.numel() for p in net.parameters())/1e6:.3f} M\n'
    print(message)

    # save in the disk
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"{opt.name}net.txt")
    with open(file_name, "wt") as f:
        f.write(message)
        f.write("\n")


def create_model(opt):
    """
    根据给定配置创建一个网络模型
    """
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print(f"model {type(instance).__name__} was created!")
    print_network(instance, opt)
    return instance

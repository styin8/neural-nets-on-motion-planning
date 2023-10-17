import importlib
from models.base_model import BaseModel
import torch.nn as nn


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
        print(f"In {model_name}.py, there are no subclasses of nn.Module that match the {model_name}!")
    return model
    

def create_model(opt):
    """
    根据给定配置创建一个网络模型
    """
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print(f"model {type(instance).__name__} was created")
    return instance
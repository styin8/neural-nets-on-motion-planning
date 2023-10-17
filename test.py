import importlib

def find_model_using_name(model_name):
    """
    根据模型名返回模型对象
    """
    model_filename = "model." + model_name + "_model"
    

def create_model(opt):
    """
    根据给定配置创建一个网络模型
    """
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print(f"model {type(instance).__name__} was created")
    return instance
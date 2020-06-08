import logging
from logging import Logger
from logging.handlers import TimedRotatingFileHandler
from torch import nn
import torch
from config import Config
def init_logger(logger_name, logging_path):
    if logger_name not in Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        handler = TimedRotatingFileHandler(filename=logging_path, when='D', backupCount = 7)
        datefmt = '%Y-%m-%d %H:%M:%S'
        format_str = '[%(asctime)s]: %(name)s %(filename)s[line:%(lineno)s] %(levelname)s  %(message)s'
        formatter = logging.Formatter(format_str, datefmt)
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
    logger = logging.getLogger(logger_name)
    return logger

def hook_fn(m, i, o):
  logger = init_logger('grad', './data/grad.log')
  logger.info("HahaModel: {} hahaInput: {} hahaOutput: {}".format(m,i,o))
  logger.info("grad hahaINput min max:{}__{}".format(torch.min(i),torch.max(i)))
  logger.info("grad hahaOut min max:{}__{}".format(torch.min(o),torch.max(o)))

def get_all_layers(net):
  for name, layer in net._modules.items():
    if isinstance(layer, nn.Sequential):
      get_all_layers(layer)
    else:
      layer.register_backward_hook(hook_fn)

def check_model(state_dict):
    for k,v in state_dict.items():
        if isinstance(v, dict):
            check_model(v)
        elif isinstance(v,torch.Tensor):
            if torch.any(torch.isnan(v)):
                print("happen nan", k)
                break
def check_tensor(t,model):
    if not torch.any(torch.isnan(t)):
        get_all_layers(model)
        torch.save({
                    'epoch': 0,
                    'model': model.state_dict()
                }, f'{Config.data_dir}/{Config.fn}_error.pth')
    assert not torch.any(torch.isnan(t))
    if not torch.any(torch.isinf(t)):
        torch.save({
                    'epoch': 0,
                    'model': model.state_dict()
                }, f'{Config.data_dir}/{Config.fn}_error.pth')
    assert not torch.any(torch.isinf(t))


Mylogger = init_logger('logger_var', './data/all.log')
def logger_var(name, var):
        Mylogger.info("{}: {}".format(name,var))
        Mylogger.info("{}: {}__{}".format(name, torch.min(var),torch.max(var)))
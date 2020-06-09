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
  logger.info("HahaModel:  hahaInput: {}_{} hahaOutput: {}_{}".format(m,i,o))

def get_all_layers(net):
  for name, layer in net._modules.items():
    if isinstance(layer, nn.Sequential):
      get_all_layers(layer)
    else:
      layer.register_backward_hook(hook_fn)

def check_model(state_dict):
    for k,v in state_dict.items():
        if k=='token_encoder':
            continue
        elif isinstance(v, dict):
            check_model(v)
        elif isinstance(v,torch.Tensor):
            if torch.any(torch.isnan(v)):
                print("happen nan", k)
def check_tensor(t):
    assert not torch.any(torch.isnan(t))
    assert not torch.any(torch.isinf(t))

def is_t_nan(var):
    return  torch.any(torch.isnan(var)) or torch.any(torch.isnan(var))

Mylogger = init_logger('logger_var', './data/var.log')
def logger_var(name, step, var):
        Mylogger.info("{}_{}: {}__{}".format(step, name, torch.min(var), torch.max(var)))
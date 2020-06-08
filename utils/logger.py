import logging
from logging import Logger
from logging.handlers import TimedRotatingFileHandler
from torch import nn
import torch
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
  logger = init_logger('grad', './data-dev/grad.log')
  logger.info("{}{}{}".format(m.name,i,o))

def get_all_layers(net):
  for name, layer in net._modules.items():
    if name=='token_encoder':
        continue
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
def check_tensor(t):
    assert not torch.any(torch.isnan(t))
    assert not torch.any(torch.isinf(t))
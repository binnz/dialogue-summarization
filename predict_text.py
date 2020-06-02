import logging
from config import Config
import torch
from model import build_model
from transformers import BertTokenizer
from utils import eval_test
from utils.helper import qa,dialogue,real_report

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    logging.info('*** Initializing Predict Mode ***')
    epoch = 0
    device = torch.device(Config.device)
    tokenizer = BertTokenizer.from_pretrained(Config.pretrained_model_name_or_path)
    model = build_model(Config).to(device)
    # state_dict = torch.load(f'{Config.data_dir}/{Config.fn}_{epoch}.pth')
    # model.load_state_dict(state_dict['model'])
    model.eval()

    text = eval_test(Config, qa, dialogue, tokenizer, model, device)
    print('qa: {}'.format(qa))
    print('dialogue: {}'.format(dialogue))
    print('report: {}'.format(text))
    print('real report: {}'.format(real_report))
    print("eval test finish")

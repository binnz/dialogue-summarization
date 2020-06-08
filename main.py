import logging
import os
import pickle
import torch

from config import Config
from transformers import BertTokenizer, get_linear_schedule_with_warmup, AdamW
from utils import (DialogDataset, one_cycle, make_train_data_from_txt,
                   seed_everything, BalancedDataLoader)
from model import build_model
from utils.logger import check_model
if __name__ == '__main__':
    logging.info('*** Initializing ***')

    if not os.path.isdir(Config.data_dir):
        os.mkdir(Config.data_dir)

    seed_everything(Config.seed)
    device = torch.device(Config.device)

    start_epoch = 0
    tokenizer = BertTokenizer.from_pretrained(Config.pretrained_model_name_or_path)

    logging.info('Preparing training data')
    if Config.use_pickle:
        with open(f'{Config.pickle_path}', 'rb') as f:
            train_data = pickle.load(f)
    else:
        train_data = make_train_data_from_txt(Config, tokenizer)
        with open(f'{Config.pickle_path}', 'wb') as f:
            pickle.dump(train_data, f)

    dataset = DialogDataset(train_data, tokenizer)
    data_loader = BalancedDataLoader(dataset, tokenizer.pad_token_id)
    t_total = len(data_loader) // Config.gradient_accumulation_steps * Config.num_train_epochs

    logging.info('Define Models')
    model = build_model(Config).to(device)
    model.unfreeze()
    model.freeze_bert()
    model.zero_grad()
    logging.info('Define Loss and Optimizer')
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": Config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=Config.learning_rate, eps=Config.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=Config.warmup_steps, num_training_steps=t_total
    )

    if Config.load:
        state_dict = torch.load(f'{Config.data_dir}/{Config.fn}.pth', map_location=device)
        check_model(state_dict)
        start_epoch = 0
        print(f'Start Epoch: {start_epoch}')
        model.load_state_dict(state_dict['model'], strict=False)
        # optimizer.load_state_dict(state_dict['opt'],strict=False)
        # scheduler.load_state_dict(state_dict['scheduler'])

    logging.info('Start Training')
    print("Total steps: ", t_total)
    for epoch in range(start_epoch, Config.num_train_epochs):
        one_cycle(epoch, Config, model, optimizer, scheduler, data_loader, tokenizer, device)

import logging
import os
import pickle

import torch
import torch.nn as nn
from  tqdm import trange

from config import Config
from transformers import BertTokenizer, get_linear_schedule_with_warmup, AdamW
from utils import (DialogDataset, one_cycle, evaluate, make_train_data_from_txt,
                   seed_everything, BalancedDataLoader)
from model import build_model

logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if Config.local_rank in [-1, 0] else logging.WARN,
    )

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
    # itf = make_itf(train_data, Config.vocab_size)
    dataset = DialogDataset(train_data, tokenizer)
    data_loader = BalancedDataLoader(dataset, tokenizer.pad_token_id)
    t_total = len(data_loader) // Config.gradient_accumulation_steps * Config.num_train_epochs

    logging.info('Define Models')
    model = build_model(Config).to(device)
    model.unfreeze()
    model.freeze_bert()
    model.zero_grad()

    logging.info('Define Loss and Optimizer')
    criterion = nn.CrossEntropyLoss(reduction='none')
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
    # optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.lr, betas=Config.betas, eps=1e-9)

    if Config.load:
        state_dict = torch.load(f'{Config.data_dir}/{Config.fn}.pth')
        start_epoch = 10
        print(f'Start Epoch: {start_epoch}')
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['opt'])

    logging.info('Start Training')
    train_iterator = trange(
        start_epoch, int(Config.num_train_epochs), desc="Epoch", disable=Config.local_rank not in [-1, 0]
    )
    print("T total", t_total)
    for epoch in train_iterator:
        one_cycle(epoch, Config, model, optimizer, scheduler, criterion, data_loader,
              tokenizer, device)
        # evaluate(Config, 'おはよーーー', tokenizer, model, device)

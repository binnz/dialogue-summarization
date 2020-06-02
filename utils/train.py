import torch
import os
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from .batch import Batch
from utils.eval import eval_test
from utils.helper import qa, dialogue, real_report
from tensorboard_logger import Logger
from utils.helper import save_config

logger = Logger(logdir="./tensorboard_logs", flush_secs=10)


def one_cycle(epoch, config, model, optimizer, scheduler, criterion, data_loader,
              tokenizer, device):
    model.train()

    with tqdm(total=len(data_loader), desc=f'Epoch: {epoch + 1}') as pbar:
        for i, data in enumerate(data_loader):
            batch = Batch(data, device, pad=tokenizer.pad_token_id)
            # out,_token_features = model(batch.source, batch.source_mask,
            #                 batch.target, batch.target_mask, batch.utter_type)
            out = train_one_batch(batch, model, config)
            loss = criterion(out.transpose(1, 2), batch.target_y).mean()
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            print("Loss", loss)
            logger.log_value('loss', loss.item(), epoch*len(data_loader) + i)
            if (i + 1) % config.gradient_accumulation_steps == 0:
                clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            pbar.update(1)
            pbar.set_postfix_str(f'Loss: {loss.item():.5f}')
        # if i % 100 == 0:
        #     torch.cuda.empty_cache()
        # always overwrite f'{config.data_dir}/{config.fn}.pth'
        # if i % 10000 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'opt': optimizer.state_dict()
            }, f'{config.data_dir}/{config.fn}.pth')
        # not overwrite

            if i % 50 == 0:
                text = eval_test(config, qa, dialogue, tokenizer, model, device)
                print('report', text)
                print("real report", real_report)
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, f'{config.data_dir}/{config.fn}_{epoch}.pth')
    save_config(config, os.path.join(config.data_dir, "model_config.json"))
    print('*** Saved Model ***')
    print("epoch {} finish".format(epoch))


def train_one_batch(batch, model,config):
    source = batch.source
    source_mask = batch.source_mask
    target = batch.target
    target_mask = batch.target_mask
    utter_type = batch.utter_type
    target_len = batch.target_len
    src_features, utterance_mask, token_features, token_mask = model.encode(source, source_mask, utter_type)
    max_target_len = max(target_len)
    step_losses = []
    coverage = torch.zeros_like(utterance_mask).contiguous().float()
    for di in range(min(max_target_len, config.max_decode_output_length)):
        local_target = target[:, :di+1]
        l_target_mask = target_mask[:, di, :di+1]
        output, attn_dist, coverage = model.decode(local_target, l_target_mask, src_features,
                         utterance_mask, token_features, token_mask, coverage, di)

import torch
import os
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from .batch import Batch

def one_cycle(epoch, config, model, optimizer, scheduler, criterion, data_loader,
              tokenizer, device):
    model.train()
    with tqdm(total=len(data_loader), desc=f'Epoch: {epoch + 1}') as pbar:
        for i, data in enumerate(data_loader):
            batch = Batch(data, device, pad=tokenizer.pad_token_id)
            out = model(batch.source, batch.source_mask,
                            batch.target, batch.target_mask, batch.utter_type)

            loss = criterion(out.transpose(1, 2), batch.target_y).mean()
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            if (i + 1) % config.gradient_accumulation_steps == 0:
                clip_grad_norm_(model.parameters(), config.max_grad_norm)
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
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, f'{config.data_dir}/{config.fn}_{epoch}.pth')
    torch.save(config, os.path.join(config.data_dir, "model_config.json"))
    print('*** Saved Model ***')
    print("epoch {} finish".format(epoch))

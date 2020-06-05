import torch
import os
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from .batch import Batch
from utils.eval import eval_test
from utils.helper import qa, dialogue, real_report
from utils.helper import save_config


def one_cycle(epoch, config, model, optimizer, scheduler, data_loader,
              tokenizer, device):
    model.train()

    with tqdm(total=len(data_loader), desc=f'Epoch: {epoch + 1}') as pbar:
        for i, data in enumerate(data_loader):
            batch = Batch(data, device, pad=tokenizer.pad_token_id)
            loss = train_one_batch(batch, model, config)
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            print("Loss", loss.item())
            if (i + 1) % config.gradient_accumulation_steps == 0:
                clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            pbar.update(1)
            torch.cuda.empty_cache()
            pbar.set_postfix_str(f'Loss: {loss.item():.5f}')
            if i % 100 == 0 and i != 0:
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'opt': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, f'{config.data_dir}/{config.fn}.pth')

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


def train_one_batch(batch, model, config):
    source = batch.source
    source_mask = batch.source_mask
    target = batch.target
    target_mask = batch.target_mask
    utter_type = batch.utter_type
    target_len = batch.target_len
    target_y = batch.target_y
    src_features, utterance_mask, token_features, token_mask = model.encode(source, source_mask, utter_type)
    max_target_len = max(target_len)
    step_losses = []
    coverage = torch.zeros_like(utterance_mask).contiguous().float()
    token_coverage = torch.zeros_like(token_mask).contiguous().float()
    index_1 = torch.arange(0, config.batch_size).long()
    for di in range(min(max_target_len, config.max_decode_output_length)-1):
        local_target = target[:, :di+1]
        l_target_mask = target_mask[:, di, :di+1]
        l_target_mask = l_target_mask.unsqueeze(-1)
        vocab_dist, tgt_attn_dist, p_gen, next_cov, next_tok_cov, tok_utter_index = model.decode(local_target, l_target_mask, src_features, utterance_mask, token_features, token_mask, coverage, token_coverage)
        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * tgt_attn_dist
            token_attn_indices = source[tok_utter_index, index_1, :]
            final_dist = vocab_dist_.scatter_add(1, token_attn_indices, attn_dist_)
        else:
            final_dist = vocab_dist
        real_target = target_y[:, di]
        target_1 = real_target.unsqueeze(1)
        glob_prob = torch.gather(final_dist, 1, target_1)
        glob_prob_ = glob_prob.squeeze()
        step_loss = -torch.log(glob_prob_ + config.eps)
        if config.is_coverage:
            tgt_cov = token_coverage[tok_utter_index, index_1, :]
            res = torch.min(tgt_attn_dist, tgt_cov)
            step_coverage_loss = torch.sum(res, 1)
            step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
            coverage = next_cov
            token_coverage = next_tok_cov
        step_mask = target_mask[:, di, di]
        step_loss = step_loss * step_mask
        step_losses.append(step_loss)

    sum_loss_11 = torch.stack(step_losses, 1)
    sum_losses = torch.sum(sum_loss_11, 1)
    batch_avg_loss = sum_losses/target_len
    loss = torch.mean(batch_avg_loss)
    return loss

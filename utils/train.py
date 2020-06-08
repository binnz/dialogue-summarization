import torch
import os
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from .batch import Batch
from utils.eval import eval_test
from utils.helper import qa, dialogue, real_report
from utils.helper import save_config
from config import Config
from utils.logger import init_logger
from utils.logger import hook_fn,check_tensor,get_all_layers,logger_var
logger = init_logger(__name__, Config.logger_path)

start=1688
end=1698
def one_cycle(epoch, config, model, optimizer, scheduler, data_loader,
              tokenizer, device):
    model.train()

    with tqdm(total=len(data_loader), desc=f'Epoch: {epoch + 1}') as pbar:
        for i, data in enumerate(data_loader):
            batch = Batch(data, device, pad=tokenizer.pad_token_id)

            loss = train_one_batch(i, batch, model, config)
            loss = loss / config.gradient_accumulation_steps
            logger.info("C step:{}.step {}".format(i, loss))
            loss.backward()
            if i in range(start,end):
                get_all_layers(model)
            print("Loss", loss.item())
            if (i + 1) % config.gradient_accumulation_steps == 0:
                clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                for param_group in optimizer.param_groups:
                    logger.info('step lr {}_{}'.format(i,param_group['lr']))
            pbar.update(1)
            torch.cuda.empty_cache()
            pbar.set_postfix_str(f'Loss: {loss.item():.5f}')
            if i % 100 == 0 and i != 0:
                s = len(data_loader)//16
                st = i//s
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'opt': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, f'{config.data_dir}/{config.fn}_{st}.pth')

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


def train_one_batch(i, batch, model, config):
    source = batch.source
    source_mask = batch.source_mask
    target = batch.target
    target_mask = batch.target_mask
    utter_type = batch.utter_type
    target_len = batch.target_len
    target_y = batch.target_y
    check_tensor(source,model)
    if i in range(start,end):
        logger_var("INput source{}", source)
        logger_var("INput source_mask{}",source_mask)
        logger_var("INput target{}",target)
        logger_var("INput target_mask{}",target_mask)
        logger_var("INput utter_type{}",utter_type)
        logger_var("INput target_len{}",target_len)
        logger_var("INput target_y{}",target_y)
    src_features, utterance_mask, token_features, token_mask = model.encode(source, source_mask, utter_type)
    if i in range(start,end):
        logger_var("encoder out src_features{}",src_features)
        logger_var("encoder out utterance_mask{}",utterance_mask)
        logger_var("encoder out token_features{}",token_features)
        logger_var("encoder out token_mask{}",token_mask)

    max_target_len = max(target_len)
    assert len(max_target_len[max_target_len == 0]) == 0
    step_losses = []
    coverage = torch.zeros_like(utterance_mask).contiguous().float()
    token_coverage = torch.zeros_like(token_mask).contiguous().float()
    index_1 = torch.arange(0, config.batch_size).long()
    for di in range(min(max_target_len, config.max_decode_output_length)-1):
        local_target = target[:, :di+1]
        l_target_mask = target_mask[:, di, :di+1]
        l_target_mask = l_target_mask.unsqueeze(-1)
        check_tensor(src_features,model)
        check_tensor(token_features,model)
        check_tensor(coverage,model)
        check_tensor(token_coverage,model)
        if i in range(start,end):
            logger_var("index_1",index_1)
            logger_var("local_target",local_target)
            logger_var("l_target_mask",l_target_mask)

        vocab_dist, tgt_attn_dist, p_gen, next_cov, next_tok_cov, tok_utter_index = model.decode(local_target, l_target_mask, src_features, utterance_mask, token_features, token_mask, coverage, token_coverage)
        if i in range(start,end):
            logger_var("vocab_dist",vocab_dist)
            logger_var("tgt_attn_dist",tgt_attn_dist)
            logger_var("p_gen",p_gen)
            logger_var("next_cov",next_cov)
            logger_var("next_tok_cov",next_tok_cov)
            logger_var("tok_utter_index",tok_utter_index)

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * tgt_attn_dist
            token_attn_indices = source[tok_utter_index, index_1, :]
            final_dist = vocab_dist_.scatter_add(1, token_attn_indices, attn_dist_)
        else:
            final_dist = vocab_dist
        if i in range(start,end):
            logger_var("final_dist",final_dist)
        real_target = target_y[:, di]
        target_1 = real_target.unsqueeze(1)
        glob_prob = torch.gather(final_dist, 1, target_1)
        glob_prob_ = glob_prob.squeeze()
        if i in range(start,end):
            logger_var("glob_prob",glob_prob)
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

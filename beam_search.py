"""
https://github.com/MehwishFatimah/transformers_summarization/blob/master/model/beam.py

"""
from config import Config
import torch
from utils import subsequent_mask


def init_vars(device, src, src_mask, utter_type, model, tokenizer):

    start_index = tokenizer.cls_token_id
    mem, utter_mask, token_features, token_mask = model.encode(src, src_mask, utter_type)
    ys = torch.ones(1, 1).fill_(start_index).long().to(device)
    trg_mask = subsequent_mask(ys.size(1)).type_as(ys)

    coverage = torch.zeros_like(utter_mask).contiguous().float()
    token_coverage = torch.zeros_like(token_mask).contiguous().float()
    vocab_dist, tgt_attn_dist, p_gen, next_cov, next_tok_cov, tok_utter_index = model.decode(ys, trg_mask, mem, utter_mask, token_features, token_mask, coverage, token_coverage)
    index_1 = torch.arange(0, Config.batch_size).long()
    if Config.pointer_gen:
        vocab_dist_ = p_gen * vocab_dist
        attn_dist_ = (1 - p_gen) * tgt_attn_dist
        token_attn_indices = src[tok_utter_index, index_1, :]
        final_dist = vocab_dist_.scatter_add(1, token_attn_indices, attn_dist_)
    else:
        final_dist = vocab_dist
    final_dist = torch.log(final_dist)
    log_scores, ix = final_dist.topk(Config.beam_size)
    outputs = torch.zeros(Config.beam_size, Config.max_decode_output_length).long()

    outputs = outputs.to(device)
    outputs[:, 0] = start_index
    outputs[:, 1] = ix[0]
    e_outputs = torch.zeros(Config.beam_size, mem.size(-2), mem.size(-1))
    e_outputs = e_outputs.to(device)
    e_outputs[:, :] = mem[0]
    return outputs, e_outputs, log_scores, utter_mask, next_cov, next_tok_cov, mem, utter_mask, token_features, token_mask


def k_best_outputs(outputs, prob, log_scores, i, k):
    
    log_probs, ix = prob.topk(k)
    log_probs = log_probs + log_scores.transpose(0, 1)
    k_probs, k_ix = log_probs.view(-1).topk(k)

    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)
    return outputs, log_scores


def beam_search(device, src, src_mask, utter_type, model, tokenizer):
    outputs, e_outputs, log_scores, utter_mask, coverage, token_coverage, mem, utter_mask, token_features, token_mask = init_vars(device, src, src_mask, utter_type, model, tokenizer)
    utter_len = utter_mask.shape[1]
    token_len = token_mask.shape[2]
    coverage = coverage.expand(Config.beam_size, utter_len).contiguous()
    token_coverage = token_coverage.expand(utter_len, Config.beam_size, token_len).contiguous()
    u, b, t_l, dim = token_features.shape
    token_features = token_features.expand(u, Config.beam_size, t_l, dim).contiguous()
    end_index = tokenizer.sep_token_id
    ind = None
    index_1 = torch.arange(0, Config.batch_size).long()
    for i in range(2, Config.max_decode_output_length):
        trg_mask = torch.tensor([[True for _ in range(i)]])
        vocab_dist, tgt_attn_dist, p_gen, coverage, token_coverage, tok_utter_index = model.decode(outputs[:, :i], trg_mask, e_outputs, utter_mask, token_features, token_mask, coverage, token_coverage)
        # ys, trg_mask, mem, utter_mask, token_features, token_mask, coverage, token_coverage
        if Config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * tgt_attn_dist
            token_attn_indices = src[tok_utter_index, index_1, :]
            final_dist = vocab_dist_.scatter_add(1, token_attn_indices, attn_dist_)
        else:
            final_dist = vocab_dist
        final_dist = torch.log(final_dist)
        outputs, log_scores = k_best_outputs(outputs, final_dist, log_scores, i, Config.beam_size)

        ones = (outputs == end_index).nonzero()  # Occurrences of end symbols for all input summaries.
        summary_lengths = torch.zeros(len(outputs), dtype=torch.long).to(device)
        for vec in ones:
            i = vec[0]
            if summary_lengths[i] == 0:  # First end symbol has not been found yet
                summary_lengths[i] = vec[1]  # Position of first end symbol
        num_finished_summaries = len([s for s in summary_lengths if s > 0])

        if num_finished_summaries == Config.beam_size:
            alpha = 0.7
            div = 1/(summary_lengths.type_as(log_scores)**alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break

    if ind is None:
        return outputs[0][1:]
    else:
        return outputs[ind][1:]

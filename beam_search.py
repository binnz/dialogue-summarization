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
    coverage = torch.zeros(utter_mask.size())
    out = model.decode(ys, trg_mask, mem, utter_mask, token_features, token_mask, coverage)

    prob = model.generate(out[:, -1])
    log_scores, ix = prob.topk(Config.beam_size)
    outputs = torch.zeros(Config.beam_size, Config.max_decode_output_length).long()

    outputs = outputs.to(device)
    outputs[:, 0] = start_index
    outputs[:, 1] = ix[0]
    e_outputs = torch.zeros(Config.beam_size, mem.size(-2),mem.size(-1))
    e_outputs = e_outputs.to(device)
    e_outputs[:, :] = mem[0]
    return outputs, e_outputs, log_scores, utter_mask


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
    outputs, e_outputs, log_scores, utter_mask = init_vars(device, src, src_mask, utter_type, model, tokenizer)

    end_index = tokenizer.sep_token_id
    ind = None
    for i in range(2, Config.max_decode_output_length):
        trg_mask = subsequent_mask(i).to(device)
        out = model.decode(e_outputs, outputs[:, :i], utter_mask, trg_mask)
        prob = model.generate(out[:, -1])
        outputs, log_scores = k_best_outputs(outputs, prob, log_scores, i, Config.beam_size)

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

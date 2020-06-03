import torch
from .helper import create_single_sample
from beam_search import beam_search


def evaluate(data, tokenizer, model, device):

    src = data.source
    src_mask = data.source_mask
    utter_type = data.utter_type
    with torch.no_grad():
        output_token = beam_search(device, src, src_mask, utter_type, model, tokenizer)
    while output_token[-1] == 0:
        output_token = output_token[:-1]
    text = tokenizer.decode(output_token)
    return text


def eval_test(config, qa, dialogue, tokenizer, model, device):
    input_ids, input_mask, utter_type = create_single_sample(qa, dialogue, config.max_src_num_length, tokenizer)

    src = torch.tensor(input_ids, dtype=torch.long, device=device)
    src = src.unsqueeze(0).transpose(0, 1)
    src_mask = torch.tensor(input_mask, dtype=torch.long, device=device)
    src_mask = src_mask.unsqueeze(0).transpose(0, 1)
    utter_type = torch.tensor(utter_type, device=device)
    utter_type = utter_type.unsqueeze(0)
    with torch.no_grad():
        output_token = beam_search(device, src, src_mask, utter_type, model, tokenizer)
    text = tokenizer.decode(output_token)
    return text

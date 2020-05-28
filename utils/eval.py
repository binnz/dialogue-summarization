import torch

from .helper import subsequent_mask


def evaluate(config, data, tokenizer, model, device, verbose=True):

    src = data.source
    src_mask = data.source_mask
    utter_type = data.utter_type
    mem, utter_mask = model.encode(src, src_mask, utter_type)
    ys = torch.ones(1, 1).fill_(tokenizer.cls_token_id).long().to(device)
    with torch.no_grad():
        for i in range(config.max_decode_output_length - 1):
            out = model.decode(mem, ys, utter_mask, subsequent_mask(ys.size(1)).type_as(ys))
            prob = model.generate(out[:, -1])
            _, candidate = prob.topk(5, dim=1)
            next_word = candidate[0, 0]
            if next_word == tokenizer.sep_token_id:
                break
            ys = torch.cat([ys, torch.ones(1, 1).type_as(ys).fill_(next_word).long()], dim=1)
    ys = ys.view(-1).detach().cpu().numpy().tolist()[1:]
    text = tokenizer.decode(ys)

    return text, data.qid[0]

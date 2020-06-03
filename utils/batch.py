import torch
import torch.nn.utils.rnn as rnn

from .helper import subsequent_mask


class Batch:

    def __init__(self, data, device, mode='train', pad: int = 0):
        input_src = [torch.tensor(x["input_src_ids"]) for x in data]
        input_mask_src = [torch.tensor(x["input_src_mask"]) for x in data]
        input_src = rnn.pad_sequence(input_src, batch_first=True)
        input_mask_src = rnn.pad_sequence(input_mask_src, batch_first=True)
        input_src = input_src.transpose(0, 1).to(device)
        input_mask_src = input_mask_src.transpose(0, 1).to(device)
        self.source = input_src
        self.source_mask = input_mask_src
        utter_type_ids = [torch.tensor(x["utterances_type"]) for x in data]
        utter_type_ids = rnn.pad_sequence(utter_type_ids, batch_first=True, padding_value=0)
        utter_type_ids = utter_type_ids.to(device)
        self.utter_type = utter_type_ids
        self.qid = [x["QID"] for x in data]

        if mode == 'train':
            target_ids = [torch.tensor(x["target_ids"]) for x in data]
            target_len = [len(x) for x in target_ids]
            target = rnn.pad_sequence(target_ids, batch_first=True, padding_value=pad)
            target = target.to(device)
            self.target = target[:, :-1]
            self.target_y = target[:, 1:]
            self.target_mask = self.make_std_mask(self.target, pad)
            self.n_tokens = (self.target != pad).sum()
            self.target_len = torch.tensor(target_len).to(device)

    @staticmethod
    def make_std_mask(target: torch.Tensor, pad: int) -> torch.Tensor:
        mask = (target != pad).unsqueeze(-2)
        mask = mask & subsequent_mask(target.size(-1)).type_as(mask)
        return mask

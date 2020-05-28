from torch.utils.data import Dataset
from torch.utils.data.sampler import RandomSampler, BatchSampler, SequentialSampler

from config import Config


class BalancedDataLoader(BatchSampler):

    def __init__(self, data: Dataset, pad_id: int, mode='train'):
        if mode == 'train':
            super().__init__(RandomSampler(data), Config.batch_size, True)
        else:
            super().__init__(SequentialSampler(data), Config.batch_size, True)
        self.pad_id = pad_id
        self.count = 0

    def __iter__(self):
        data_list = list()
        # sampler is RandomSampler
        for i in self.sampler:
            self.count += 1
            sampler_data = self.sampler.data_source[i]
            data_list.append(sampler_data)
            if self.count % self.batch_size == 0:
                assert len(data_list) == self.batch_size
                # src = rnn.pad_sequence(src_list, batch_first=True, padding_value=self.pad_id)
                # tgt = rnn.pad_sequence(tgt_list, batch_first=True, padding_value=self.pad_id)
                data = data_list.copy()
                data_list.clear()
                yield data

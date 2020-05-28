import os
import tensorflow as tf
import pandas as pd
import collections
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import logging
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, qid, utterance, report=None):
        """Constructs a InputExample.

        Args:
        """
        self.qid = qid
        self.utterance = utterance
        self.report = report


class InputFeature(object):
    """A single set of bert inout feature."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids=None,
                 target_ids=None,
                 target_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.target_ids = target_ids
        self.target_mask = target_mask


class InputFeatures:
    def __init__(self, input_features, target_ids, target_mask, label_ids):
        self.input_features = input_features
        self.target_ids = target_ids
        self.target_mask = target_mask
        self.label_ids = label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        raise NotImplementedError()


class DSProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def _create_examples(self, data, set_type=None):
        examples = []

        # ['QID', 'Brand', 'Model', 'Question', 'Dialogue', 'Report']

        for qid, qa, dialogue, report in zip(
                data['QID'],
                data['Question'],
                data['Dialogue'],
                data['Report']):
            utterances = []
            qa = clear_sentence(qa)
            if qa:
                utterances.append(qa)
            dialogue = dialogue.split('|')
            for utter in dialogue:
                valid_utter = clear_sentence(utter)
                if valid_utter:
                    utterances.append(valid_utter)
            report = clear_sentence(report)
            example = (qid, utterances, report)
            examples.append(example)
        return examples

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        return read_file(input_file)


def convert_example_to_feature(example, max_seq_length_src, max_seq_length_tgt, tokenizer):

    def convert_src_feature(example, max_seq_length_src, tokenizer):
        tokens = tokenizer.tokenize(example)
        if len(tokens) > max_seq_length_src - 2:
            tokens = tokens[:(max_seq_length_src - 2)]

        tokens_src = []
        segment_ids_src = []
        tokens_src.append("[CLS]")
        segment_ids_src.append(0)
        for token in tokens:
            tokens_src.append(token)
            segment_ids_src.append(0)
        tokens_src.append("[SEP]")
        segment_ids_src.append(0)

        input_ids_src = tokenizer.convert_tokens_to_ids(tokens_src)

        input_mask_src = [1] * len(input_ids_src)
        # Zero-pad up to the sequence length.
        while len(input_ids_src) < max_seq_length_src:
            input_ids_src.append(0)
            input_mask_src.append(0)
            segment_ids_src.append(0)

        assert len(input_ids_src) == max_seq_length_src
        assert len(input_mask_src) == max_seq_length_src
        assert len(segment_ids_src) == max_seq_length_src
        return input_ids_src, input_mask_src, segment_ids_src

    def convert_tgt_feature(example, max_seq_length_tgt, tokenizer):
        tokens = tokenizer.tokenize(example.tgt_txt)
        if len(tokens) > max_seq_length_tgt - 2:
            tokens = tokens[0:(max_seq_length_tgt - 2)]

        tokens_tgt = []
        tokens_tgt.append("[CLS]")
        for token in tokens:
            tokens_tgt.append(token)
        tokens_tgt.append("[SEP]")
        input_ids_tgt = tokenizer.convert_tokens_to_ids(tokens_tgt)

        labels_tgt = input_ids_tgt[1:]

        # Adding begiining and end token
        input_ids_tgt = input_ids_tgt[:-1]
        input_mask_tgt = [1] * len(input_ids_tgt)
        while len(input_ids_tgt) < max_seq_length_tgt:
            input_ids_tgt.append(0)
            input_mask_tgt.append(0)
            labels_tgt.append(0)

        return input_ids_tgt, input_mask_tgt, labels_tgt

    features_src = []
    for utter in example.utterance:
        input_ids, input_mask, segment_ids = convert_src_feature(utter)
        features_src.append(InputFeature(input_ids, input_mask, segment_ids))
    input_ids_tgt, input_mask_tgt, labels_tgt = convert_tgt_feature(
        example.report)

    if (int(example.gid[1:]) + 1) % 1000 == 0:
        print("{} examples".format(example.gid))
        print("Input List Tokens: {}".format(
            [[token for token in tokens]for tokens in example.utterance]))
        print("Report Tokens: {}".format([token for token in example.report]))
        print("Report Input Ids: {}".format([id for id in input_ids_tgt]))
        print("Report Label: {}".format([id for id in labels_tgt]))
    return InputFeatures(features_src, input_ids_tgt, input_mask_tgt, labels_tgt)


def convert_examples_to_features(examples, max_seq_length_src, max_seq_length_tgt, tokenizer, output_file):
    writer = tf.python_io.TFRecordWriter(output_file)

    def create_int_feature(values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

    for index, example in enumerate(examples):
        feature = convert_example_to_feature(
            example, max_seq_length_src, max_seq_length_tgt, tokenizer)

        features = collections.OrderedDict()
        features["input_features"] = create_int_feature(feature.input_features)
        features["target_ids"] = create_int_feature(feature.target_ids)
        features["target_mask"] = create_int_feature(feature.target_mask)
        features['label_ids'] = create_int_feature(feature.label_ids)

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def read_file(filename):
    """

    """
    data = pd.read_csv(filename)
    return data


def clear_sentence(sentence):
    sentence = sentence.strip()
    if sentence.startswith('技师说：'):
        sentence = sentence[4:]
    elif sentence.startswith('车主说：'):
        sentence = sentence[4:]
    if sentence.startswith('[语音]') or sentence.startswith('[图片]'):
        return None
    return sentence

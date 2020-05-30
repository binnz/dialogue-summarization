import torch
import random
import os
import numpy as np
import pandas as pd
import pickle
import collections
import six

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def subsequent_mask(size: int) -> torch.Tensor:
    attn_shape = (1, size, size)
    subsequence_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequence_mask) == 0


def make_predict_data_from_txt(config, tokenizer):
    result = list()
    data = pd.read_csv(config.predict_data_path)

    # ['QID', 'Brand', 'Model', 'Question', 'Dialogue', 'Report']

    max_seq_length_src = config.max_src_num_length
    for qid, qa, dialogue in zip(
                data['QID'],
                data['Question'],
                data['Dialogue']):

        utterances = []
        utterances_type = []
        # utterance_type    0: Question  1: 车主  2: 技师
        qa, _ = clear_sentence(qa)
        if qa:
            utterances.append(qa)
            utterances_type.append(0)
        if not isinstance(dialogue, six.string_types):
            dialogue = str(dialogue)
            print("Float Dialogue QID{}, text{}".format(qid, dialogue))
        dialogue = dialogue.split('|')
        for utter in dialogue:
            valid_utter, utter_type = clear_sentence(utter)
            if valid_utter is not None and utter_type is not None:
                utterances.append(valid_utter)
                utterances_type.append(utter_type)
        input_src_ids = []
        input_src_mask = []
        for utter in utterances:
            ids_src, mask_src = convert_src_feature(utter, max_seq_length_src, tokenizer)
            input_src_ids.append(ids_src)
            input_src_mask.append(mask_src)
        features = collections.OrderedDict()
        features["input_src_ids"] = input_src_ids
        features["input_src_mask"] = input_src_mask
        features["utterances_type"] = utterances_type
        features["QID"] = qid
        result.append(features)
    with open(f'{config.predict_pickle_path}', 'wb') as f:
        pickle.dump(result, f)
    return result


def make_train_data_from_txt(config, tokenizer):
    result = list()
    data = pd.read_csv(config.train_data_path)
    # ['QID', 'Brand', 'Model', 'Question', 'Dialogue', 'Report']

    max_seq_length_src = config.max_src_num_length
    for qid, qa, dialogue, report in zip(
                data['QID'],
                data['Question'],
                data['Dialogue'],
                data['Report']):

        utterances = []
        utterances_type = []
        # utterance_type    0: Question  1: 车主  2: 技师
        qa, _ = clear_sentence(qa)
        if qa:
            utterances.append(qa)
            utterances_type.append(0)

        if not isinstance(dialogue, six.string_types):
            dialogue = str(dialogue)
            print("Float Dialogue QID{}, text{}".format(qid, dialogue))
            print("After str Float Dialogue QID{}, text{}".format(qid, dialogue))
            continue

        valid_dialogue = dialogue.split('|')
        for utter in valid_dialogue:
            valid_utter, utter_type = clear_sentence(utter)
            if valid_utter is not None and utter_type is not None:
                utterances.append(valid_utter)
                utterances_type.append(utter_type)
            elif valid_utter is not None or utter_type is not None:
                print("Error data sample {}.{}.{}".format(qid, qa, dialogue))

        if len(utterances) == 0:
            continue
        input_src_ids = []
        input_src_mask = []
        for utter in utterances:
            ids_src, mask_src = convert_src_feature(utter, max_seq_length_src, tokenizer)
            input_src_ids.append(ids_src)
            input_src_mask.append(mask_src)
        if not isinstance(report, six.string_types):
            print("Error Report sample QID is {}.report{}".format(qid, str(report)))
            continue
        target_ids = convert_tgt_feature(report, max_seq_length_src, tokenizer)
        features = collections.OrderedDict()
        features["input_src_ids"] = input_src_ids
        features["input_src_mask"] = input_src_mask
        features["utterances_type"] = utterances_type
        features["target_ids"] = target_ids
        features["QID"] = qid
        result.append(features)
    with open(f'{config.pickle_path}', 'wb') as f:
        pickle.dump(result, f)
    return result


def convert_src_feature(example, max_seq_length_src, tokenizer):
    tokens = tokenizer.tokenize(example)
    if len(tokens) > max_seq_length_src - 2:
        tokens = tokens[:(max_seq_length_src - 2)]

    tokens_src = ["[CLS]"]
    for token in tokens:
        tokens_src.append(token)
    tokens_src.append("[SEP]")

    input_ids_src = tokenizer.convert_tokens_to_ids(tokens_src)

    input_mask_src = [1] * len(input_ids_src)
    # Zero-pad up to the sequence length.
    while len(input_ids_src) < max_seq_length_src:
        input_ids_src.append(0)
        input_mask_src.append(0)

    assert len(input_ids_src) == max_seq_length_src
    assert len(input_mask_src) == max_seq_length_src
    return input_ids_src, input_mask_src


def convert_tgt_feature(example, max_seq_length_tgt, tokenizer):
    if not isinstance(example, six.string_types):
        example = str(example)
        print("Float report{}".format(example))
        tokens = tokenizer.tokenize(example)
        print("After str Float report tokens{}".format(tokens))
    else:
        tokens = tokenizer.tokenize(example)
    if len(tokens) > max_seq_length_tgt - 2:
        tokens = tokens[0:(max_seq_length_tgt - 2)]

    tokens_tgt = ["[CLS]"]
    for token in tokens:
        tokens_tgt.append(token)
    tokens_tgt.append("[SEP]")
    input_ids_tgt = tokenizer.convert_tokens_to_ids(tokens_tgt)

    return input_ids_tgt

    # features_src = []
    # for utter in example.utterance:
    #     input_ids, input_mask, segment_ids = convert_src_feature(utter)
    #     features_src.append((input_ids, input_mask, segment_ids))
    # input_ids_tgt, input_mask_tgt, labels_tgt = convert_tgt_feature(
    #     example.report)
    #
    # if (int(example.gid[1:]) + 1) % 1000 == 0:
    #     print("{} examples".format(example.gid))
    #     print("Input List Tokens: {}".format(
    #         [[token for token in tokens]for tokens in example.utterance]))
    #     print("Report Tokens: {}".format([token for token in example.report]))
    #     print("Report Input Ids: {}".format([id for id in input_ids_tgt]))
    #     print("Report Label: {}".format([id for id in labels_tgt]))
    # return (features_src, input_ids_tgt, input_mask_tgt, labels_tgt)

def clear_sentence(sentence):
    sentence = sentence.strip()
    utter_type = None
    if sentence.startswith('技师说：'):
        sentence = sentence[4:]
        utter_type = 2
    elif sentence.startswith('车主说：'):
        sentence = sentence[4:]
        utter_type = 1
    if sentence.startswith('[语音]') or sentence.startswith('[图片]'):
        return None, None
    return sentence, utter_type


def create_single_sample(qa, dialogue, max_seq_length_src, tokenizer):
    utterances = []
    utterances_type = []
    if qa:
        utterances.append(qa)
        utterances_type.append(0)

    if not isinstance(dialogue, six.string_types):
        return None, None

    valid_dialogue = dialogue.split('|')
    for utter in valid_dialogue:
        valid_utter, utter_type = clear_sentence(utter)
        if valid_utter is not None and utter_type is not None:
            utterances.append(valid_utter)
            utterances_type.append(utter_type)
        elif valid_utter is not None or utter_type is not None:
            continue
    input_src_ids = []
    input_src_mask = []
    for utter in utterances:
        ids_src, mask_src = convert_src_feature(utter, max_seq_length_src, tokenizer)
        input_src_ids.append(ids_src)
        input_src_mask.append(mask_src)
    return input_src_ids, input_src_mask, utterances_type

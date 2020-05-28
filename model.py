import torch
from torch import nn
from transformers import BertModel
from transformer_encoder import TransformerEncoder
from decoder import build_decoder
from neural import PositionalEncoding
from generator import Generator


def build_model(config):
    model = DialogueSummarization(
        config.pretrained_model_name_or_path,
        config.num_layers,
        config.dim_model,
        config.num_heads,
        config.dim_ff,
        config.dropout,
        config.max_utter_num_length,
        config.utter_type,
        config.max_decode_output_length,
        config.vocab_size,
        None)
    model.freeze_bert()
    return model


class DialogueSummarization(nn.Module):

    def __init__(self, pretrained_model_name_or_path,
                 num_layers,
                 dim_model,
                 num_heads,
                 dim_ff,
                 dropout,
                 max_utter_num_length,
                 utter_type,
                 max_decode_output_length,
                 vocab_size,
                 embeddings=None):
        super().__init__()
        self.token_encoder = BertModel.from_pretrained(
            pretrained_model_name_or_path)
        self.target_embeddings = nn.Sequential(self.token_encoder.embeddings.word_embeddings,
                                               PositionalEncoding(dropout, dim_model, max_decode_output_length))
        self.utterance_encoder = TransformerEncoder(
            num_layers, dim_model, num_heads, dim_ff, dropout, max_utter_num_length, utter_type, embeddings)
        self.decoder = build_decoder(
            num_layer=num_layers, heads=num_heads, d_model=dim_model, d_ff=dim_ff, drop_rate=dropout)
        self.generator = Generator(dim_model, vocab_size)

    def freeze_bert(self):
        for p in self.token_encoder.parameters():
            p.requires_grad = False

    def forward(self, source, source_mask, target, target_mask, utter_type):
        utter_length, batch_size, _ = source.size()
        utterance_input_mask = [
            [1 if any(i) else 0 for i in j]for j in source_mask]
        device = source_mask.device
        utterance_input_mask = torch.tensor(
            utterance_input_mask).transpose(0, 1).to(device)
        utterance_features = []
        for utter_index in range(utter_length):
            out, _ = self.token_encoder(
                source[utter_index], source_mask[utter_index])
            utterance_features.append(torch.mean(out, dim=1))
        utterance_features = torch.stack(utterance_features, dim=1)
        src_features, mask_hier = self.utterance_encoder(
            utterance_features, utterance_input_mask, utter_type)
        target_features = self.target_embeddings(target)
        src_features = src_features.transpose(0, 1)
        x = self.decoder(target_features, src_features,
                         utterance_input_mask, target_mask)
        x = self.generator(x)
        return x

    def encode(self, source, source_mask, utter_type):
        utter_length, batch_size, _ = source.size()
        utterance_input_mask = [
            [1 if any(i) else 0 for i in j]for j in source_mask]
        device = source_mask.device
        utterance_input_mask = torch.tensor(
            utterance_input_mask).transpose(0, 1).to(device)
        utterance_features = []
        for utter_index in range(utter_length):
            out, _ = self.token_encoder(
                source[utter_index], source_mask[utter_index])
            utterance_features.append(torch.mean(out, dim=1))
        utterance_features = torch.stack(utterance_features, dim=1)
        src_features, mask_hier = self.utterance_encoder(
            utterance_features, utterance_input_mask, utter_type)
        src_features = src_features.transpose(0, 1)
        return src_features, utterance_input_mask

    def decode(self, src_features, target_ids, src_mask, target_mask):
        target_features = self.target_embeddings(target_ids)
        return self.decoder(target_features, src_features, src_mask, target_mask)

    def generate(self, x):
        return self.generator(x)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

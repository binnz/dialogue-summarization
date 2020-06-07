"""
Implementation of "Attention is All You Need"
"""
import math
import torch.nn as nn
import torch
from attn import MultiHeadedAttention, MultiHeadedPooling
from neural import PositionwiseFeedForward, PositionalEncoding, sequence_mask
from utils.logger import init_logger
from config import Config

logger = init_logger(__name__, Config.logger_path)


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.
    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, inputs, mask):
        """
        Transformer Encoder Layer definition.
        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`
        Returns:
            (`FloatTensor`):
            * outputs `[batch_size x src_len x model_dim]`
        """
        input_norm = self.layer_norm(inputs)
        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerPoolingLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerPoolingLayer, self).__init__()

        self.pooling_attn = MultiHeadedPooling(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        context = self.pooling_attn(inputs, inputs,
                                    mask=mask)
        out = self.dropout(context)

        return self.feed_forward(out)

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, max_utter_num_length, utter_type, embeddings):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.max_length = max_utter_num_length
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.utterance_emb = nn.Embedding(utter_type, d_model)
        self.pos_emb = PositionalEncoding(dropout, d_model, max_utter_num_length)
        self.emb_layerNorm = nn.LayerNorm(d_model, eps=1e-6)
        self.emb_dropout = nn.Dropout(dropout)
        self.transformer_local = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, src, attention_mask, utter_type, lengths=None):
        """ See :obj:`EncoderBase.forward()`"""
        src = src[:, :self.max_length, :]
        attention_mask = attention_mask[:, :self.max_length]
        utter_type = utter_type[:, :self.max_length]
        logger.info("before emb {}__{}".format(torch.min(src), torch.max(src)))
        out = self.pos_emb(src)
        utter_emb = self.utterance_emb(utter_type)
        logger.info("after emb {}__{}".format(torch.min(out), torch.max(out)))
        logger.info("utter emb {}__{}".format(torch.min(utter_emb), torch.max(utter_emb)))
        out = out + utter_emb
        out = self.emb_layerNorm(out)
        out = self.emb_dropout(out)
        logger.info("before layer {}__{}".format(torch.min(out), torch.max(out)))
        for i in range(self.num_layers):
            out = self.transformer_local[i](out, out, 1 - attention_mask)  # all_sents * max_tokens * dim
        out = self.layer_norm(out)
        mask_hier = attention_mask[:, :, None].float()
        src_features = out * mask_hier
        return src_features, mask_hier

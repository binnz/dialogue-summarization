import torch
import torch.nn as nn
from config import Config
from attention import SourceTargetAttention, SelfAttention
from ffn import FFN
from utils.logger import init_logger,logger_var

logger = init_logger(__name__, './data/weight.log')


def build_decoder(num_layer=6, heads=8, d_model=512, d_ff=2048, drop_rate=0.1):
    decoder_layers = [DecoderLayer(heads, d_model, d_ff, drop_rate) for _ in range(num_layer)]
    decoder = Decoder(nn.ModuleList(decoder_layers), d_model)
    return decoder


class Decoder(nn.Module):

    def __init__(self, layers, d_model):
        super(Decoder, self).__init__()
        # decoder layers
        self.layers = layers
        self.norm = nn.LayerNorm(d_model)
        self.utterance_attention = LocalAttention(Config.dim_model)
        self.token_attention = LocalAttention(Config.dim_model)
        if Config.pointer_gen:
            self.p_gen_linear = nn.Linear(2 * Config.dim_model, 1, bias=True)

        self.out1 = nn.Linear(2 * Config.dim_model, Config.dim_model)
        self.out2 = nn.Linear(Config.dim_model, Config.vocab_size)

    def forward(self, target_features, target_mask, encode_features,
                         utterance_mask, token_features, token_mask, coverage, token_coverage):
        logger.info("encoder out2{}".format(self.out2.weight.grad))
        utterance_mask = utterance_mask.unsqueeze(-2)
        batch_size, _, _ = target_features.shape
        # note that memory is passed through encoder
        for layer in self.layers:
            target_features = layer(target_features, encode_features, utterance_mask, target_mask)
        target_features = self.norm(target_features)
        local_dec_feature = target_features[:, -1, :]
        #  utterance local attention
        utterance_mask = utterance_mask.squeeze(1)
        utterance_output, utterance_attn_dist, utterance_coverage = self.utterance_attention(local_dec_feature, encode_features, utterance_mask, coverage)
        utter_length = len(token_features)
        assert utter_length == len(token_mask)
        assert utter_length == encode_features.shape[1]

        # Below is target local attention to every utterances(each utterance several tokens)
        utters_attn_dist = []
        utters_outout = []
        next_tok_cov = []
        for u_index in range(utter_length):
            output, attn_dist, coverage = self.token_attention(local_dec_feature, token_features[u_index], token_mask[u_index], token_coverage[u_index])
            utters_attn_dist.append(attn_dist)
            utters_outout.append(output)
            next_tok_cov.append(coverage)
        utters_attn_dist = torch.stack(utters_attn_dist, dim=0)
        utters_outout = torch.stack(utters_outout, dim=0)
        next_tok_cov = torch.stack(next_tok_cov, dim=0)

        p_gen = torch.cat((utterance_output, local_dec_feature), dim=1)
        p_gen = self.p_gen_linear(p_gen)
        p_gen = torch.sigmoid(p_gen)
        _, utter_index = torch.topk(utterance_attn_dist, 1, dim=1)
        utter_index = utter_index.transpose(1, 0).squeeze(0)
        index_1 = torch.arange(0, batch_size).long()
        target_attn_dist = utters_attn_dist[utter_index, index_1, :]
        target_out_put = utters_outout[utter_index, index_1, :]
        output = torch.cat((target_out_put, local_dec_feature), 1)

        output = self.out1(output)
        output = self.out2(output)
        vocab_dist = torch.softmax(output, dim=1)
        return vocab_dist, target_attn_dist, p_gen, utterance_coverage, next_tok_cov, utter_index

class LocalAttention(nn.Module):
    def __init__(self, source_dim):
        super(LocalAttention, self).__init__()
        # attention
        if Config.is_coverage:
            self.W_c = nn.Linear(1, source_dim, bias=False)
        self.decode_proj = nn.Linear(source_dim, source_dim)
        self.v = nn.Linear(source_dim, 1, bias=False)
        self.source_dim = source_dim

    def forward(self, target_t, encode_features, source_mask, coverage):
        b, t_k, e_dim = encode_features.shape

        dec_fea = self.decode_proj(target_t)  # B x dim_model
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, e_dim).contiguous() # B x t_k x dim_model
        dec_fea_expanded = dec_fea_expanded.view(-1, e_dim)  # B * t_k x dim_model
        encode_features = encode_features.view(-1, e_dim)
        att_features = encode_features + dec_fea_expanded  # B * t_k x dim_model
        if Config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x dim_model
            att_features = att_features + coverage_feature

        e = torch.tanh(att_features)  # B * t_k x dim_model
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = torch.softmax(scores, dim=1)*source_mask  # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        normalization_factor[normalization_factor == 0] = 1.0
        attn_dist = attn_dist_ / normalization_factor
        encode_features = encode_features.view(b, t_k, e_dim)
        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        output = torch.bmm(attn_dist, encode_features)  # B x 1 x n
        output = output.view(-1, self.source_dim)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if Config.is_coverage:
            coverage = coverage + attn_dist

        return output, attn_dist, coverage


class DecoderLayer(nn.Module):

    def __init__(self, h=8, d_model=512, d_ff=2048, drop_rate=0.1):
        super(DecoderLayer, self).__init__()

        # Self Attention Layer
        # query key and value come from previous layer.
        self.self_attn = SelfAttention(h, d_model, drop_rate)
        # Source Target Attention Layer
        # query come from encoded space.
        # key and value come from previous self attention layer
        self.st_attn = SourceTargetAttention(h, d_model, drop_rate)
        self.ff = FFN(d_model, d_ff)

    def forward(self, x, mem, source_mask, target_mask):
        # self attention
        x = self.self_attn(x, target_mask)
        # source target attention
        x = self.st_attn(mem, x, source_mask)
        # pass through feed forward network
        return self.ff(x)


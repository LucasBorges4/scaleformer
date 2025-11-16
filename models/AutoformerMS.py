# Copyright (c) 2019-present, Royal Bank of Canada.
# Copyright (c) 2021 THUML @ Tsinghua University
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the Autoformer (https://arxiv.org/pdf/2106.13008.pdf) implementation
# from https://github.com/thuml/Autoformer by THUML @ Tsinghua University
####################################################################################

import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_mine
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp

class moving_avg(nn.Module):
    """
    Downsample series using an average pooling (robust to small sequences)
    """
    def __init__(self):
        super(moving_avg, self).__init__()

    def forward(self, x, scale=1):
        enc_seq_len = x_enc.size(1)  # comprimento da sequência encoder
        
        valid_scales = [s for s in self.scales if enc_seq_len >= s and s > 0]

        seq_len = queries.size(-1) if queries.dim() >= 3 else queries.size(1)
        if seq_len == 0:
            # Retorna tensor de forma compatível com o esperado a montante.
            # Normalmente o primeiro retorno é `out`, segundo é `attn`
            # devolvemos `queries` sem alteração e `None` para attn (ou zeros se preferir).
            out = queries.clone()
            attn = None
            return out, attn

        if len(valid_scales) == 0:
            # fallback: use scale 1 para garantir que algo rode
            valid_scales = [1]
        scales = valid_scales

        if x is None:
            return None
        # x: (batch, seq_len, channels)
        x_perm = x.permute(0, 2, 1)  # -> (batch, channels, seq_len)
        seq_len = x_perm.size(2)
        # compute output size safely (at least 1)
        out_size = max(1, seq_len // scale)
        # use adaptive avg pool to avoid zero-length outputs
        x_pooled = nn.functional.adaptive_avg_pool1d(x_perm, out_size)
        x = x_pooled.permute(0, 2, 1)
        return x


class Model(nn.Module):
    """
    Multi-scale version of Autoformer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        self.decomp = series_decomp(configs.moving_avg)

        # Embedding
        # We use our new DataEmbedding which incldues the scale information
        self.enc_embedding = DataEmbedding_mine(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding_mine(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout, is_decoder=True)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg = configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout, output_attention=False), configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout, output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg = configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        """
        following functions will be used to manage scales
        """
        self.scale_factor = configs.scale_factor
        self.scales = configs.scales
        self.mv = moving_avg()
        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='linear')
        self.input_decomposition_type = 1


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
            enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        def sample_mark(x_mark, scale, target_len):
            """
            Safely sample time-mark features with stride `scale` starting at scale//2.
            If sampled length < target_len, pad by repeating last row.
            """
            if x_mark is None:
                return None
            # attempt sampling
            start = scale // 2
            sampled = x_mark[:, start::scale, :]
            cur_len = sampled.size(1)
            if cur_len >= target_len:
                return sampled[:, :target_len, :]
            # pad by repeating last timestep
            if cur_len == 0:
                # fallback: repeat the first mark (or zeros) to match
                pad = sampled.new_zeros((x_mark.size(0), target_len, x_mark.size(2)))
                return pad
            need = target_len - cur_len
            last = sampled[:, -1:, :].repeat(1, need, 1)
            return torch.cat([sampled, last], dim=1)

        scales = self.scales
        label_len = x_dec.shape[1] - self.pred_len
        outputs = []

        for scale in scales:
            # compute safe scaled lengths
            pred_scale_len = max(1, self.pred_len // scale)
            label_scale_len = max(1, label_len // scale)

            # downsample encoder input (moving_avg already robust)
            enc_out = self.mv(x_enc, scale)

            # if enc_out sequence length is 0 (safety), skip this scale
            if enc_out is None or enc_out.size(1) == 0:
                continue

            if scale == scales[0]:  # initialize the input of decoder at first step
                if self.input_decomposition_type == 1:
                    mean = enc_out.mean(1).unsqueeze(1)
                    enc_out = enc_out - mean

                    # tmp_mean and zeros sized by pred_scale_len (>=1)
                    tmp_mean = torch.mean(enc_out, dim=1).unsqueeze(1).repeat(1, pred_scale_len, 1)
                    zeros = torch.zeros([x_dec.shape[0], pred_scale_len, x_dec.shape[2]], device=x_enc.device)

                    seasonal_init, trend_init = self.decomp(enc_out)

                    # safe slicing: if trend_init has fewer timesteps than label_scale_len, take what exists
                    trend_slice = min(trend_init.size(1), label_scale_len)
                    seasonal_slice = min(seasonal_init.size(1), label_scale_len)

                    trend_init = torch.cat([trend_init[:, -trend_slice:, :], tmp_mean], dim=1)
                    seasonal_init = torch.cat([seasonal_init[:, -seasonal_slice:, :], zeros], dim=1)

                    dec_out = self.mv(x_dec, scale) - mean
                else:
                    dec_out = self.mv(x_dec, scale)
                    mean = enc_out.mean(1).unsqueeze(1)
                    enc_out = enc_out - mean
                    # safe label slice
                    cur = min(dec_out.size(1), label_scale_len)
                    if cur > 0:
                        dec_out[:, :cur, :] = dec_out[:, :cur, :] - mean
            else:  # generation the input at each scale and cross normalization
                # upsample previous coarse output
                dec_out = self.upsample(dec_out_coarse.detach().permute(0, 2, 1)).permute(0, 2, 1)

                # safe assignment from mv of short sequences
                mv_part = self.mv(x_dec[:, :label_len, :], scale)
                cur_mv = mv_part.size(1)
                cur_dec = dec_out.size(1)
                # how many positions to fill: min(cur_mv, label_scale_len, cur_dec)
                to_fill = min(cur_mv, label_scale_len, cur_dec)
                if to_fill > 0:
                    dec_out[:, :to_fill, :] = mv_part[:, :to_fill, :]

                # compute mean across concatenated enc_out and dec_out tail robustly
                # ensure dimensions are compatible for concatenation
                # dec_out[:, label_len//scale:, :] might be out of range; do safe slicing
                tail_start = min(label_scale_len, dec_out.size(1))
                enc_cat = torch.cat((enc_out, dec_out[:, tail_start:, :]), 1)
                mean = enc_cat.mean(1).unsqueeze(1)
                enc_out = enc_out - mean
                dec_out = dec_out - mean

            # redefining the inputs to the decoder to be scale aware
            trend_init = torch.zeros_like(dec_out)
            seasonal_init = dec_out

            # sample time marks robustly to match enc_out and seasonal_init lengths
            enc_mark_sampled = sample_mark(x_mark_enc, scale, enc_out.size(1))
            dec_mark_sampled = sample_mark(x_mark_dec, scale, seasonal_init.size(1))

            enc_out = self.enc_embedding(enc_out, enc_mark_sampled, scale=scale, first_scale=scales[0], label_len=label_len)
            enc_out, attns = self.encoder(enc_out)
            dec_out = self.dec_embedding(seasonal_init, dec_mark_sampled, scale=scale, first_scale=scales[0], label_len=label_len)
            seasonal_part, trend_part = self.decoder(dec_out, enc_out, trend=trend_init)
            dec_out_coarse = seasonal_part + trend_part

            dec_out_coarse = dec_out_coarse + mean

            # append safely using pred_scale_len
            out_len = min(dec_out_coarse.size(1), pred_scale_len)
            outputs.append(dec_out_coarse[:, -out_len:, :])

        return outputs

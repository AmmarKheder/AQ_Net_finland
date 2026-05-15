#!/usr/bin/env python3
"""
Copyright (c) Ammar Kheder
Licensed under the MIT License.

v2 Finland: prediction RESIDUELLE (delta) multi-horizon.
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=2, dim_head=256, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.unify_heads = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, self.heads, -1).transpose(1, 2), qkv)
        # v2 Finland: pre-scaling de q (evite overflow bf16 avant le softmax)
        scores = torch.matmul(q * self.scale, k.transpose(-2, -1))
        time_decay = torch.arange(scores.shape[-1]).float().to(x.device)
        time_decay = torch.exp(-time_decay / 50)
        scores = scores + time_decay.unsqueeze(0).unsqueeze(0)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, n, -1)
        out = self.unify_heads(out)
        return self.norm(out), attn


# v2 Finland: variante iTransformer (ICLR 2024) - les VARIABLES deviennent
# des tokens (serie de chaque feature -> 1 token), l'attention modelise les
# correlations inter-variables. Meme signature/residuel que le LSTM (drop-in).
class _VarSelfAttn(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.3):
        super().__init__()
        self.h = heads
        self.dk = dim // heads
        self.scale = self.dk ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):                       # x: (B, n_tokens, dim)
        b, n, d = x.shape
        q, k, v = (t.view(b, n, self.h, self.dk).transpose(1, 2)
                   for t in self.qkv(x).chunk(3, dim=-1))
        att = torch.softmax((q * self.scale) @ k.transpose(-2, -1), dim=-1)
        o = (att @ v).transpose(1, 2).contiguous().view(b, n, d)
        return self.norm(x + self.drop(self.proj(o))), att


class _ITEncoderLayer(nn.Module):
    # fidele a l'officiel: AttentionLayer -> +res/LN -> FFN(d_ff) -> +res/LN
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = _VarSelfAttn(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x):
        a, att = self.attn(x)                       # _VarSelfAttn fait deja x+res,LN
        x = self.norm1(a)
        x = self.norm2(x + self.ff(x))
        return x, att


class iTransformerModel(nn.Module):
    """
    Adaptation fidele de iTransformer (thuml, MIT, ICLR 2024) :
      - RevIN / non-stationary norm : on retire mean/std PAR SERIE (par
        fenetre) en entree (le vrai driver de perf, gere la non-stationnarite)
      - embedding inverse : serie d'UNE variable -> 1 token (B,F,d_model)
      - encoder full-attention inter-variables + FFN(d_ff) + LayerNorm
      - head : pool variates -> n_horizons (la cible PM2.5 n'est pas une
        variate d'entree -> pas de "filter target channel" possible)
      - residuel : pred = pm25_current + delta
    """

    def __init__(self, input_dim, seq_length, horizons=(6, 12, 24, 48),
                 d_model=64, depth=2, heads=4, d_ff=128, dropout=0.3,
                 residual=True, use_norm=True):
        super().__init__()
        self.horizons = list(horizons)
        self.residual = residual
        self.use_norm = use_norm
        self.embed = nn.Linear(seq_length, d_model)
        self.in_drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            _ITEncoderLayer(d_model, heads, d_ff, dropout)
            for _ in range(depth)])
        self.enc_norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleList(
            [nn.Linear(d_model, 1) for _ in self.horizons])
        for hd in self.heads:
            nn.init.normal_(hd.weight, mean=0.0, std=0.01)
            nn.init.zeros_(hd.bias)

    def forward(self, x, pm25_current):
        # x: (B, L, F). RevIN: normalisation par serie sur l'axe temps.
        if self.use_norm:
            mean = x.mean(1, keepdim=True).detach()
            x = x - mean
            std = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
            x = x / std
        # (B, L, F) -> (B, F, L) -> tokens variables (B, F, d_model)
        t = self.in_drop(self.embed(x.transpose(1, 2)))
        att = None
        for layer in self.layers:
            t, att = layer(t)
        t = self.enc_norm(t)
        feats = t.mean(dim=1)                        # pool sur les variables
        deltas = torch.stack(
            [hd(feats).squeeze(-1) for hd in self.heads], dim=1)
        if self.residual:
            return pm25_current.unsqueeze(1) + deltas, att
        return deltas, att


class LSTMAttentionModel(nn.Module):
    """
    v2 Finland : backbone LSTM + Multi-Head Attention INCHANGE.
    Difference : une tete lineaire par horizon predit un DELTA (residu),
    ajoute a la valeur PM2.5 courante. Pas de Softplus sur le delta
    (la pollution peut baisser -> delta negatif autorise).
    """

    def __init__(self, input_dim, horizons=(6, 12, 24, 48), hidden_dim=128,
                 num_layers=3, attention_heads=2, attention_dim_head=128,
                 dropout_rate=0.3, residual=True):
        super().__init__()
        self.horizons = list(horizons)
        self.residual = residual
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = MultiHeadAttention(
            hidden_dim, heads=attention_heads, dim_head=attention_dim_head)
        # v2 Finland: n tetes residuelles, init petits poids -> demarre en persistance
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in self.horizons])
        for head in self.heads:
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(head.bias)

    def forward(self, x, pm25_current):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        attn_out, attn_weights = self.attention(lstm_out)
        feats = torch.mean(attn_out, dim=1)                       # (B, hidden)
        deltas = torch.stack(
            [h(feats).squeeze(-1) for h in self.heads], dim=1)    # (B, n_h)
        if self.residual:
            pred = pm25_current.unsqueeze(1) + deltas              # residuel
        else:
            pred = deltas
        return pred, attn_weights

import torch
import torch.nn as nn
import numpy as np
from math import sqrt


class TriangularCausalMask:
    def __init__(self, batch_size, length, device="cpu"):
        mask_shape = [batch_size, 1, length, length]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        batch_size, query_length, num_heads, embed_dim = queries.shape
        _, key_length, _, _ = values.shape
        scale = self.scale or 1.0 / sqrt(embed_dim)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(batch_size, query_length, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attention = self.dropout(torch.softmax(scale * scores, dim=-1))
        values = torch.einsum("bhls,bshd->blhd", attention, values)

        if self.output_attention:
            return values.contiguous(), attention
        return values.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        batch_size, query_length, _ = queries.shape
        _, key_length, _ = keys.shape
        num_heads = self.n_heads

        queries = self.query_projection(queries).view(batch_size, query_length, num_heads, -1)
        keys = self.key_projection(keys).view(batch_size, key_length, num_heads, -1)
        values = self.value_projection(values).view(batch_size, key_length, num_heads, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta,
        )
        out = out.view(batch_size, query_length, -1)
        return self.out_projection(out), attn

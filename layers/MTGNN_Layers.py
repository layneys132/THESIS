import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class NConv(nn.Module):
    def forward(self, x, adjacency):
        x = torch.einsum("ncwl,vw->ncvl", (x, adjacency))
        return x.contiguous()


class Linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class MixProp(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super().__init__()
        self.nconv = NConv()
        self.mlp = Linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adjacency):
        adjacency = adjacency + torch.eye(adjacency.size(0), device=x.device, dtype=x.dtype)
        degree = adjacency.sum(1)
        normalized = adjacency / degree.view(-1, 1)
        hidden = x
        outputs = [hidden]
        for _ in range(self.gdep):
            hidden = self.alpha * x + (1.0 - self.alpha) * self.nconv(hidden, normalized)
            outputs.append(hidden)
        mixed = torch.cat(outputs, dim=1)
        return self.mlp(mixed)


class DilatedInception(nn.Module):
    def __init__(self, c_in, c_out, dilation_factor=2):
        super().__init__()
        self.kernel_set = [2, 3, 6, 7]
        branch_channels = int(c_out / len(self.kernel_set))
        self.tconv = nn.ModuleList(
            [nn.Conv2d(c_in, branch_channels, (1, kernel), dilation=(1, dilation_factor)) for kernel in self.kernel_set]
        )

    def forward(self, x):
        outputs = [conv(x) for conv in self.tconv]
        target_length = outputs[-1].size(3)
        outputs = [output[..., -target_length:] for output in outputs]
        return torch.cat(outputs, dim=1)


class GraphConstructor(nn.Module):
    def __init__(self, nnodes, k, dim, alpha=3.0, static_feat=None):
        super().__init__()
        self.nnodes = nnodes
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

    def _node_embeddings(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1
        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))
        return nodevec1, nodevec2

    def forward(self, idx):
        nodevec1, nodevec2 = self._node_embeddings(idx)
        adjacency = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adjacency = F.relu(torch.tanh(self.alpha * adjacency))

        mask = torch.zeros(idx.size(0), idx.size(0), device=idx.device, dtype=adjacency.dtype)
        scores, topk_indices = (adjacency + torch.rand_like(adjacency) * 0.01).topk(self.k, dim=1)
        mask.scatter_(1, topk_indices, scores.fill_(1.0))
        return adjacency * mask

    def fullA(self, idx):
        nodevec1, nodevec2 = self._node_embeddings(idx)
        adjacency = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        return F.relu(torch.tanh(self.alpha * adjacency))


class MTGNNLayerNorm(nn.Module):
    __constants__ = ["normalized_shape", "weight", "bias", "eps", "elementwise_affine"]

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x, idx):
        if self.elementwise_affine:
            return F.layer_norm(
                x,
                tuple(x.shape[1:]),
                self.weight[:, idx, :],
                self.bias[:, idx, :],
                self.eps,
            )
        return F.layer_norm(x, tuple(x.shape[1:]), self.weight, self.bias, self.eps)

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.fc(x))


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.num_nodes = configs.stid_num_nodes
        self.in_features = configs.stid_in_features
        self.window_size = configs.seq_len
        hidden_dim = configs.stid_hidden_dim

        self.input_proj = nn.Linear(self.in_features, hidden_dim)
        self.node_emb = nn.Embedding(self.num_nodes, hidden_dim)
        self.time_emb = nn.Embedding(self.window_size, hidden_dim)
        self.mlp_blocks = nn.Sequential(
            *[MLPBlock(hidden_dim, configs.dropout) for _ in range(configs.stid_num_layers)]
        )
        self.node_attn = nn.Linear(hidden_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(hidden_dim // 2, self.pred_len),
        )

    def forward(self, x_enc, x_mark_enc=None, mask=None):
        batch_size, window_size, num_nodes, _ = x_enc.shape

        h = self.input_proj(x_enc)
        node_ids = torch.arange(num_nodes, device=x_enc.device)
        time_ids = torch.arange(window_size, device=x_enc.device)

        h = h + self.node_emb(node_ids)
        h = h + self.time_emb(time_ids).unsqueeze(0).unsqueeze(2)
        h = self.mlp_blocks(h.reshape(batch_size * window_size * num_nodes, -1))
        h = h.reshape(batch_size, window_size, num_nodes, -1)
        h = h.mean(dim=1)

        attn = torch.softmax(self.node_attn(h), dim=1)
        h = (h * attn).sum(dim=1)

        return self.head(h).unsqueeze(-1)

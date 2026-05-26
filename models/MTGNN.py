import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.MTGNN_Layers import DilatedInception, GraphConstructor, MTGNNLayerNorm, MixProp


class GTNet(nn.Module):
    def __init__(
        self,
        gcn_true,
        buildA_true,
        gcn_depth,
        num_nodes,
        predefined_A=None,
        static_feat=None,
        dropout=0.3,
        subgraph_size=20,
        node_dim=40,
        dilation_exponential=2,
        conv_channels=16,
        residual_channels=16,
        skip_channels=32,
        end_channels=64,
        seq_length=168,
        in_dim=1,
        out_dim=1,
        layers=5,
        propalpha=0.05,
        tanhalpha=3.0,
        layer_norm_affline=False,
    ):
        super().__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.seq_length = seq_length

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        self.gc = GraphConstructor(num_nodes, subgraph_size, node_dim, alpha=tanhalpha, static_feat=static_feat)

        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1) * (dilation_exponential**layers - 1) / (dilation_exponential - 1)
            )
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(
                    1 + i * (kernel_size - 1) * (dilation_exponential**layers - 1) / (dilation_exponential - 1)
                )
            else:
                rf_size_i = i * layers * (kernel_size - 1) + 1

            new_dilation = 1
            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(
                        rf_size_i + (kernel_size - 1) * (dilation_exponential**j - 1) / (dilation_exponential - 1)
                    )
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                self.filter_convs.append(
                    DilatedInception(residual_channels, conv_channels, dilation_factor=new_dilation)
                )
                self.gate_convs.append(
                    DilatedInception(residual_channels, conv_channels, dilation_factor=new_dilation)
                )
                self.residual_convs.append(
                    nn.Conv2d(in_channels=conv_channels, out_channels=residual_channels, kernel_size=(1, 1))
                )

                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(
                        nn.Conv2d(
                            in_channels=conv_channels,
                            out_channels=skip_channels,
                            kernel_size=(1, self.seq_length - rf_size_j + 1),
                        )
                    )
                else:
                    self.skip_convs.append(
                        nn.Conv2d(
                            in_channels=conv_channels,
                            out_channels=skip_channels,
                            kernel_size=(1, self.receptive_field - rf_size_j + 1),
                        )
                    )

                if self.gcn_true:
                    self.gconv1.append(MixProp(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(MixProp(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length > self.receptive_field:
                    normalized_shape = (residual_channels, num_nodes, self.seq_length - rf_size_j + 1)
                else:
                    normalized_shape = (residual_channels, num_nodes, self.receptive_field - rf_size_j + 1)
                self.norm.append(
                    MTGNNLayerNorm(normalized_shape, elementwise_affine=layer_norm_affline)
                )

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1, 1), bias=True)

        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, self.seq_length - self.receptive_field + 1),
                bias=True,
            )
        else:
            self.skip0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self.receptive_field),
                bias=True,
            )
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)

        self.register_buffer("idx", torch.arange(self.num_nodes, dtype=torch.long))

    def forward(self, x, idx=None):

        raw_input = x
        if self.seq_length < self.receptive_field:
            raw_input = F.pad(raw_input, (self.receptive_field - self.seq_length, 0, 0, 0))

        if self.gcn_true:
            if self.buildA_true:
                adjacency = self.gc(self.idx if idx is None else idx)
            else:
                adjacency = self.predefined_A
        else:
            adjacency = None

        x = self.start_conv(raw_input)
        skip = self.skip0(F.dropout(raw_input, self.dropout, training=self.training))

        for layer_index in range(self.layers):
            residual = x

            filter_output = torch.tanh(self.filter_convs[layer_index](x))
            gate_output = torch.sigmoid(self.gate_convs[layer_index](x))
            x = filter_output * gate_output
            x = F.dropout(x, self.dropout, training=self.training)

            skip = self.skip_convs[layer_index](x) + skip

            if self.gcn_true and adjacency is not None:
                x = self.gconv1[layer_index](x, adjacency) + self.gconv2[layer_index](x, adjacency.transpose(1, 0))
            else:
                x = self.residual_convs[layer_index](x)

            x = x + residual[:, :, :, -x.size(3):]
            norm_idx = self.idx if idx is None else idx
            x = self.norm[layer_index](x, norm_idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.model = GTNet(
            gcn_true=bool(configs.gcn_true),
            buildA_true=bool(configs.buildA_true),
            gcn_depth=configs.gcn_depth,
            num_nodes=configs.enc_in,
            dropout=configs.dropout,
            subgraph_size=min(configs.subgraph_size, configs.enc_in),
            node_dim=configs.node_dim,
            dilation_exponential=configs.dilation_exponential,
            conv_channels=configs.conv_channels,
            residual_channels=configs.residual_channels,
            skip_channels=configs.skip_channels,
            end_channels=configs.end_channels,
            seq_length=configs.seq_len,
            in_dim=1,
            out_dim=configs.pred_len,
            layers=configs.layers,
            propalpha=configs.propalpha,
            tanhalpha=configs.tanhalpha,
            layer_norm_affline=bool(configs.layer_norm_affline),
        )

    def forward(self, x_enc, mask=None):
        x = x_enc.transpose(1, 2).unsqueeze(1)
        output = self.model(x)
        return output.squeeze(-1)

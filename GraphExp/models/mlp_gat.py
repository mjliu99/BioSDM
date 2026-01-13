#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv, GraphConv
from .utils import create_activation, create_norm


def exists(x):
    return x is not None


class Denoising_Unet(nn.Module):
    def __init__(self,
                 in_dim_fmri,
                 in_dim_dti,
                 num_hidden,
                 out_dim,
                 num_layers,
                 nhead,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 norm,
                 ):
        super(Denoising_Unet, self).__init__()
        self.num_layers = num_layers

        # 保存 nhead 参数，后续分析可能用到
        self.nhead = nhead

        # MLP 输入层
        self.mlp_in_t = MlpBlock(in_dim=in_dim_fmri, hidden_dim=num_hidden * 2, out_dim=num_hidden,
                                 norm=norm, activation=activation)

        # GCN 层 (预处理)
        self.gcn_fmri = GraphConv(in_dim_fmri, in_dim_fmri)
        self.gcn_dti = GraphConv(in_dim_dti, in_dim_dti)

        # U-Net 结构组件
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.mlp_middle = MlpBlock(num_hidden, num_hidden, num_hidden, norm=norm, activation=activation)
        self.mlp_out_fmri = MlpBlock(num_hidden, out_dim, out_dim, norm=norm, activation=activation)
        self.mlp_out_dti = MlpBlock(num_hidden, out_dim, out_dim, norm=norm, activation=activation)

        # 构建 GAT 层
        for _ in range(num_layers):
            # Downsample GATs
            self.down_layers.append(GATConv(num_hidden, num_hidden // nhead, nhead, feat_drop,
                                            attn_drop, negative_slope))
            # Upsample GATs
            self.up_layers.append(GATConv(num_hidden, num_hidden // nhead, nhead, feat_drop,
                                          attn_drop, negative_slope))

        # 反转 up_layers 以便在 forward 中正确索引
        self.up_layers = self.up_layers[::-1]

    def create_graph_from_adjacency(self, adj_matrix):
        """
        辅助函数：从邻接矩阵重建图
        """
        adj_matrix_sparse = adj_matrix.to_sparse()
        edges = adj_matrix_sparse.indices()
        edge_weights = adj_matrix_sparse.values()
        num_nodes = adj_matrix.size(0)
        # 确保新图在正确的 device 上
        g = dgl.graph((edges[0], edges[1]), num_nodes=num_nodes).to(adj_matrix.device)
        g.edata['weight'] = edge_weights
        g.ndata['feat'] = adj_matrix
        return g

    def forward(self, g_fmri, g_dti, fmri_data, dti_data, time_embed):
        # === 1. GCN 预处理 ===
        fmri_data = self.gcn_fmri(g_fmri, fmri_data)
        dti_data = self.gcn_dti(g_dti, dti_data)

        # === 2. 混淆矩阵计算 (Cross-Modal Fusion) ===
        S_hat1 = fmri_data @ dti_data.transpose(-1, -2)
        S_hat2 = dti_data @ fmri_data.transpose(-1, -2)
        S_hat0 = (S_hat1 + S_hat2) / 2
        S_hat_G = (fmri_data + dti_data) / 2

        # 基于混淆矩阵建图
        g_S_hat_G = self.create_graph_from_adjacency(S_hat_G)
        batch_g_eval = g_S_hat_G

        # === 3. MLP 特征变换 ===
        fmri_data = self.mlp_in_t(fmri_data)
        dti_data = self.mlp_in_t(dti_data)
        S_hat0 = torch.clamp(S_hat0, min=-1e10, max=1e10)
        S_hat0 = self.mlp_in_t(S_hat0)

        # === 4. 第一层 GAT 下采样 ===
        # 这里 flatten(1) 是为了将多头注意力结果展平
        h_fmri = self.down_layers[0](g_fmri, fmri_data).flatten(1)
        h_dti = self.down_layers[0](g_dti, dti_data).flatten(1)

        # === 5. U-Net Encoder (Downsampling) ===
        h_t = S_hat0
        down_hidden = []
        for l in range(self.num_layers):
            if h_t.ndim > 2:
                h_t = h_t + time_embed.unsqueeze(1).repeat(1, h_t.shape[1], 1)

            # GAT Forward
            h_t = self.down_layers[l](g_S_hat_G, h_t).flatten(1)
            down_hidden.append(h_t)

        # === 6. Middle Bottleneck ===
        h_middle = self.mlp_middle(h_t)
        h_t = h_middle

        # === 7. U-Net Decoder (Upsampling) ===
        out_hidden = []

        # 初始化注意力变量，防止报错
        attn_fmri = None
        attn_dti = None

        for l in range(self.num_layers):
            # Skip Connection
            h_t = h_t + down_hidden[self.num_layers - l - 1]

            if h_t.ndim > 2:
                h_t = h_t + time_embed.unsqueeze(1).repeat(1, h_t.shape[1], 1)

            # --- 判断是否是最后一层 ---
            if l == self.num_layers - 1:
                # 到了最后一层，我们需要分流回 fMRI 和 DTI 两个分支
                h_t_fmri = h_t
                h_t_dti = h_t

                # *** 核心修改点 START ***
                # 在最后一层调用 GAT 时，开启 get_attention=True
                # 这会让 GAT 返回一个 tuple: (node_features, attention_weights)
                h_t_fmri, attn_fmri = self.up_layers[l](g_fmri, h_t_fmri, get_attention=True)
                h_t_dti, attn_dti = self.up_layers[l](g_dti, h_t_dti, get_attention=True)
                # *** 核心修改点 END ***

                # MLP 输出处理
                final_fmri_output = self.mlp_out_fmri(h_t_fmri.flatten(1))
                final_dti_output = self.mlp_out_dti(h_t_dti.flatten(1))

                out_hidden.append(h_t_fmri)
                out_hidden.append(h_t_dti)
            else:
                # 中间层继续上采样
                h_t = self.up_layers[l](g_S_hat_G, h_t).flatten(1)

        out_hidden = torch.cat(out_hidden, dim=-1)

        # 返回值中增加了 attn_fmri 和 attn_dti
        # 它们的形状通常是 [Num_Edges, Num_Heads, 1]
        return final_fmri_output, final_dti_output, out_hidden, batch_g_eval, attn_fmri, attn_dti


class Residual(nn.Module):
    def __init__(self, fnc):
        super().__init__()
        self.fnc = fnc

    def forward(self, x, *args, **kwargs):
        return self.fnc(x, *args, **kwargs) + x


class MlpBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 norm: str = 'layernorm', activation: str = 'prelu'):
        super(MlpBlock, self).__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.res_mlp = Residual(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                              create_norm(norm)(hidden_dim),
                                              create_activation(activation),
                                              nn.Linear(hidden_dim, hidden_dim)))
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.act = create_activation(activation)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.res_mlp(x)
        x = self.out_proj(x)
        x = self.act(x)
        return x
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File Name:     diffusion.py
# Author:        wangchenze
# Created Time:  202409  17:09
# Last Modified: <none>-<none>

import sys
from typing import Optional
import networkx as nx
import community as community_louvain  # 确保导入 community_louvain 库
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import math
import dgl
import dgl.function as fn
# from utils.utils import make_edge_weights
from .mlp_gat import Denoising_Unet
import numpy as np


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class BrainSTORE(nn.Module):
    def __init__(
            self,
            in_dim_fmri: int,  # 假设fMRI的输入维度是128
            in_dim_dti: int,  # 假设DTI的输入维度是64
            # in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            norm: Optional[str],
            alpha_l: float = 2,
            beta_schedule: str = 'linear',
            beta_1: float = 0.0001,
            beta_T: float = 0.02,
            T: int = 1000,
            **kwargs

    ):
        super(BrainSTORE, self).__init__()
        self.T = T
        beta = get_beta_schedule(beta_schedule, beta_1, beta_T, T)
        self.register_buffer(
            'betas', beta
        )
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar)
        )
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar)
        )

        self.alpha_l = alpha_l
        assert num_hidden % nhead == 0
        # 初始化Denoising_Unet时，提供两个输入维度：in_dim_fmri和in_dim_dti
        self.net = Denoising_Unet(
            in_dim_fmri=in_dim_fmri,
            in_dim_dti=in_dim_dti,
            num_hidden=num_hidden,
            out_dim=in_dim_fmri,  # 输出维度可以保持不变
            num_layers=num_layers,
            nhead=nhead,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=0.2,
            norm=norm
        )

        self.time_embedding = nn.Embedding(T, num_hidden)

    def forward(self, g_fmri, g_dti, x_fmri, x_dti, labels, community_fmri, community_dti):
        with torch.no_grad():
            # 对 fMRI 和 DTI 进行归一化
            x_fmri = F.layer_norm(x_fmri, (x_fmri.shape[-1],))
            x_dti = F.layer_norm(x_dti, (x_dti.shape[-1],))
        num_steps = 2
        loss_total = 0

        # for step  in range(num_steps):
        #     print(f"Step {step}/{num_steps}")
        t = torch.randint(self.T, size=(x_fmri.shape[0],), device=x_fmri.device)

        # 对 fMRI 和 DTI 进行加噪
        x_t_fmri, x_t_dti, time_embed, g_fmri, g_dti = self.sample_q(t, x_fmri, x_dti, g_fmri, g_dti)
        t_community = torch.randint(self.T, size=(1,), device=x_fmri.device).item()

        # print(f"t_community: {t_community}")
        # 计算去噪损失，结合 fMRI 和 DTI 的耦合关系
        loss = self.node_denoising(x_fmri, x_t_fmri, x_dti, x_t_dti, time_embed, g_fmri,
                                   g_dti, community_fmri, community_dti)
        loss_total += loss
        # 平均损失
        # loss_total /= num_steps

        # 返回损失值
        loss_item = {"total_loss": loss_total.item()}
        return loss_total, loss_item

    def sample_q_alone(self, t, x):
        miu, std = x.mean(dim=0), x.std(dim=0)
        noise = torch.randn_like(x, device=x.device)
        with torch.no_grad():
            noise = F.layer_norm(noise, (noise.shape[-1],))
        noise = noise * std + miu
        noise = torch.sign(x) * torch.abs(noise)
        x_t = (
                extract(self.sqrt_alphas_bar, t, x.shape) * x +
                extract(self.sqrt_one_minus_alphas_bar, t, x.shape) * noise
        )
        time_embed = self.time_embedding(t)
        return x_t, time_embed

    def sample_q(self, t, x_fmri, x_dti, g_fmri, g_dti):
        # 对 fMRI 数据进行加噪
        miu_fmri, std_fmri = x_fmri.mean(dim=0), x_fmri.std(dim=0)
        noise_fmri = torch.randn_like(x_fmri, device=x_fmri.device)
        with torch.no_grad():
            noise_fmri = F.layer_norm(noise_fmri, (noise_fmri.shape[-1],))
        noise_fmri = noise_fmri * std_fmri + miu_fmri
        noise_fmri = torch.sign(x_fmri) * torch.abs(noise_fmri)
        x_t_fmri = (
                extract(self.sqrt_alphas_bar, t, x_fmri.shape) * x_fmri +
                extract(self.sqrt_one_minus_alphas_bar, t, x_fmri.shape) * noise_fmri
        )

        # 对 DTI 数据进行加噪
        miu_dti, std_dti = x_dti.mean(dim=0), x_dti.std(dim=0)
        noise_dti = torch.randn_like(x_dti, device=x_dti.device)
        with torch.no_grad():
            noise_dti = F.layer_norm(noise_dti, (noise_dti.shape[-1],))
        noise_dti = noise_dti * std_dti + miu_dti
        noise_dti = torch.sign(x_dti) * torch.abs(noise_dti)
        x_t_dti = (
                extract(self.sqrt_alphas_bar, t, x_dti.shape) * x_dti +
                extract(self.sqrt_one_minus_alphas_bar, t, x_dti.shape) * noise_dti
        )

        # 对应时间步的嵌入
        time_embed = self.time_embedding(t)

        return x_t_fmri, x_t_dti, time_embed, g_fmri, g_dti

    def node_denoising(self, x_fmri, x_t_fmri, x_dti, x_t_dti, time_embed, g_fmri, g_dti, community_fmri,
                       community_dti):
        # print(f"x_fmri: {x_fmri.shape}")
        # print(f"x_dti: {x_dti.shape}")
        # print(f"community_fmri: {community_fmri.shape}")
        # print(f"community_dti: {community_dti.shape}")

        # 注意：这里的 16 是硬编码的 Batch 数量或子图数量，请根据你的 main_graph 实际 batch_size 调整
        # 如果报错索引越界，这里需要改成根据 x_fmri.shape[0] 动态计算
        batch_size_total = x_fmri.shape[0]
        sub_matrix_size = 90
        # 自动计算有多少个子矩阵，防止硬编码 16 导致错误
        num_sub_matrices = batch_size_total // sub_matrix_size

        total_loss = 0

        for i in range(num_sub_matrices):
            # print(f"i12356: {i}")
            start_idx = i * sub_matrix_size
            end_idx = (i + 1) * sub_matrix_size
            x_fmri_sub = x_fmri[start_idx:end_idx]
            x_t_fmri_sub = x_t_fmri[start_idx:end_idx]
            x_dti_sub = x_dti[start_idx:end_idx]
            x_t_dti_sub = x_t_dti[start_idx:end_idx]
            g_fmri_sub = g_fmri.subgraph(range(start_idx, end_idx))
            g_dti_sub = g_dti.subgraph(range(start_idx, end_idx))
            # 社区
            community_fmri_sub = community_fmri[start_idx:end_idx]
            community_dti_sub = community_dti[start_idx:end_idx]

            # 对 fMRI 进行社区检测和处理
            noisy_x_t_fmri_sub = self.process_by_community(x_t_fmri_sub, community_fmri_sub)
            # 对 DTI 进行社区检测和处理
            noisy_x_t_dti_sub = self.process_by_community(x_t_dti_sub, community_dti_sub)

            # --- 修改处：接收 6 个返回值 ---
            out_fmri, out_dti, _, _, attn_fmri, attn_dti = self.net(g_fmri_sub, g_dti_sub, noisy_x_t_fmri_sub,
                                                                    noisy_x_t_dti_sub, time_embed)

            # 计算损失 (训练时只用特征，不用注意力)
            loss_fmri = loss_fn(out_fmri, x_fmri_sub, self.alpha_l)
            loss_dti = loss_fn(out_dti, x_dti_sub, self.alpha_l)
            total_loss += loss_fmri + loss_dti
        return total_loss

    def process_by_community(self, x_t_fmri_sub, community_data):
        """
        优化版：利用矩阵广播一次性生成 Mask，消除所有 for 循环
        """
        # 1. 确保数据在同一设备
        device = x_t_fmri_sub.device

        # 2. 生成社区掩码 (Mask)
        # community_data shape: [N]
        # comm_row: [N, 1], comm_col: [1, N]
        # broadcasting -> [N, N]
        # 如果 community_data[i] == community_data[j]，则 mask[i, j] = True
        c = community_data.unsqueeze(1)
        mask = (c == c.T).float()  # 1.0 表示在同一社区，0.0 表示不同

        # 3. 处理对角线 (Self-loop)
        # 原逻辑：循环中 if i!=j 才置1，最后单独加回 diag
        # 新逻辑：先把对角线置为 0，用于生成噪声
        identity = torch.eye(mask.shape[0], device=device)
        mask_no_diag = mask * (1 - identity)  # 仅保留同一社区且非对角线的元素

        # 4. 生成子图特征 (只保留社区内部连接)
        subgraph_features = x_t_fmri_sub * mask_no_diag

        # 5. 一次性加噪 (不再需要循环)
        # 生成时间步 t (这里简单处理，给整个子图用同一个 t 分布，或者按原逻辑随机)
        # 原逻辑：每个社区随机生成 t。为了并行效率，我们对每个节点随机采样 t，或者全局采样
        # 这里的优化策略：直接对整个 subgraph_features 加噪
        t = torch.randint(self.T, size=(x_t_fmri_sub.shape[0],), device=device)
        noisy_subgraph, _ = self.sample_q_alone(t, subgraph_features)

        # 6. 再次应用掩码 (确保噪声只加在社区内部)
        noisy_subgraph = noisy_subgraph * mask_no_diag

        # 7. 加回原始对角线 (Self-connection)
        diag_part = x_t_fmri_sub * identity
        final_noisy_features = noisy_subgraph + diag_part

        return final_noisy_features

    # 获取 fMRI 或 DTI 图的边和权重
    def get_adjacency_matrix(self, graph):
        # 获取边的起点和终点，并转到 CPU
        src, dst = graph.edges()
        src = src.cpu()
        dst = dst.cpu()

        # 获取边的权重，先转到 CPU 再转为 numpy 数组
        weights = graph.edata['weight'].cpu().numpy()

        # 移除自环边
        mask = src != dst  # 只保留起点和终点不同的边
        src = src[mask]
        dst = dst[mask]
        weights = weights[mask]

        # 获取图的节点数
        num_nodes = graph.number_of_nodes()

        # 初始化一个全零的邻接矩阵
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        # 将权重填入邻接矩阵
        adj_matrix[src.numpy(), dst.numpy()] = weights

        return adj_matrix

    def embed_node_denoising(self, x_fmri, x_t_fmri, x_dti, x_t_dti, time_embed, g_fmri, g_dti, community_fmri,
                             community_dti):
        # print(f"x_fmri: {x_fmri.shape}")
        # print(f"x_dti: {x_dti.shape}")

        batch_size_total = x_fmri.shape[0]
        sub_matrix_size = 90
        # 动态计算循环次数
        num_sub_matrices = batch_size_total // sub_matrix_size

        all_hidden = []
        all_batch_g_eval = []
        for i in range(num_sub_matrices):
            start_idx = i * sub_matrix_size
            end_idx = (i + 1) * sub_matrix_size

            x_fmri_sub = x_fmri[start_idx:end_idx]
            x_t_fmri_sub = x_t_fmri[start_idx:end_idx]
            x_dti_sub = x_dti[start_idx:end_idx]
            x_t_dti_sub = x_t_dti[start_idx:end_idx]

            g_fmri_sub = g_fmri.subgraph(range(start_idx, end_idx))
            g_dti_sub = g_dti.subgraph(range(start_idx, end_idx))

            # 社区
            community_fmri_sub = community_fmri[start_idx:end_idx]
            community_dti_sub = community_dti[start_idx:end_idx]
            # 对 fMRI 进行社区检测和处理
            noisy_x_t_fmri_sub = self.process_by_community(x_t_fmri_sub, community_fmri_sub)

            # 对 DTI 进行社区检测和处理
            noisy_x_t_dti_sub = self.process_by_community(x_t_dti_sub, community_dti_sub)

            # --- 修改处：接收 6 个返回值 (attn_fmri, attn_dti) ---
            _, _, hidden, batch_g_eval, _, _ = self.net(g_fmri_sub, g_dti_sub, noisy_x_t_fmri_sub, noisy_x_t_dti_sub,
                                                        time_embed)

            all_hidden.append(hidden)
            all_batch_g_eval.append(batch_g_eval)

        # 将所有子矩阵的隐藏特征拼接在一起
        hidden = torch.cat(all_hidden, dim=0)  # 根据需要调整 dim 参数
        finally_batch_g_eval = dgl.batch(all_batch_g_eval)  # 根据需要调整 dim 参数

        # print(f"Number of nodes in finally_batch_g_eval: {finally_batch_g_eval.number_of_nodes()}")
        # print(f"Number of edges in finally_batch_g_eval: {finally_batch_g_eval.number_of_edges()}")
        return hidden, finally_batch_g_eval

    def embed(self, g_fmri, g_dti, x_fmri, x_dti, T, community_fmri, community_dti):
        # 生成时间步，批量大小为 x_fmri 的第一个维度大小
        t = torch.full((x_fmri.shape[0],), T, device=x_fmri.device)

        # 对 fMRI 和 DTI 数据进行归一化
        with torch.no_grad():
            x_fmri = F.layer_norm(x_fmri, (x_fmri.shape[-1],))
            x_dti = F.layer_norm(x_dti, (x_dti.shape[-1],))

        # 生成加噪后的数据和时间嵌入
        x_t_fmri, x_t_dti, time_embed, g_fmri, g_dti = self.sample_q(t, x_fmri, x_dti, g_fmri, g_dti)

        # 调用网络进行前向传播，获得隐藏特征
        hidden, finally_batch_g_eval = self.embed_node_denoising(x_fmri, x_t_fmri, x_dti, x_t_dti, time_embed, g_fmri,
                                                                 g_dti, community_fmri, community_dti)

        return hidden, finally_batch_g_eval

    # =============================================================
    # 新增：用于补充实验 (K步采样 + 节点重要性聚合)
    # =============================================================
    def get_node_attention(self, g_fmri, g_dti, x_fmri, x_dti, community_fmri, community_dti, K=10):
        self.eval()

        batch_size_total = x_fmri.shape[0]
        sub_matrix_size = 90
        num_sub_matrices = batch_size_total // sub_matrix_size

        final_scores_fmri = []
        final_scores_dti = []

        for i in range(num_sub_matrices):
            start_idx = i * sub_matrix_size
            end_idx = (i + 1) * sub_matrix_size

            x_f_sub = x_fmri[start_idx:end_idx]
            x_d_sub = x_dti[start_idx:end_idx]
            comm_f_sub = community_fmri[start_idx:end_idx]
            comm_d_sub = community_dti[start_idx:end_idx]

            g_f_sub = g_fmri.subgraph(range(start_idx, end_idx))
            g_d_sub = g_dti.subgraph(range(start_idx, end_idx))

            # 累加器
            node_score_f_accum = torch.zeros(g_f_sub.num_nodes()).to(x_fmri.device)
            node_score_d_accum = torch.zeros(g_d_sub.num_nodes()).to(x_dti.device)

            time_steps = torch.linspace(0, self.T - 1, K).long().to(x_fmri.device)

            with torch.no_grad():
                for t_val in time_steps:
                    t = torch.full((x_f_sub.shape[0],), t_val.item(), device=x_fmri.device)
                    time_embed = self.time_embedding(t)

                    x_t_f, x_t_d, _, _, _ = self.sample_q(t, x_f_sub, x_d_sub, g_f_sub, g_d_sub)
                    noisy_f = self.process_by_community(x_t_f, comm_f_sub)
                    noisy_d = self.process_by_community(x_t_d, comm_d_sub)

                    _, _, _, _, attn_f, attn_d = self.net(g_f_sub, g_d_sub, noisy_f, noisy_d, time_embed)

                    # === 核心修改 START: 改为计算出度重要性 ===

                    # fMRI 分支
                    if attn_f is not None:
                        # [Edges, 1]
                        edge_w_f = attn_f.mean(dim=1).squeeze()
                        # 获取边的源节点索引 (Source Nodes)
                        src_f, _ = g_f_sub.edges()
                        # 将边的权重累加到源节点上 (Source-based accumulation)
                        # 这代表：这个节点被多少其他节点关注了
                        node_score_f_accum.index_add_(0, src_f, edge_w_f)

                    # DTI 分支
                    if attn_d is not None:
                        edge_w_d = attn_d.mean(dim=1).squeeze()
                        src_d, _ = g_d_sub.edges()
                        node_score_d_accum.index_add_(0, src_d, edge_w_d)

                    # === 核心修改 END ===

            # 计算 K 步平均值
            avg_score_f = node_score_f_accum / K
            avg_score_d = node_score_d_accum / K

            final_scores_fmri.append(avg_score_f)
            final_scores_dti.append(avg_score_d)

        return final_scores_fmri, final_scores_dti


def loss_fn(x, y, alpha=2):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return torch.from_numpy(betas)
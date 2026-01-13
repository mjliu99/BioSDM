#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File Name:     main_graph.py

import numpy as np
import argparse
import shutil
import time
import os.path as osp
import dgl
from dgl.dataloading import GraphDataLoader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from utils.utils import (create_optimizer, create_pooler, set_random_seed)
from datasets.data_util import load_multimodal_graph_classification_dataset
from models import BrainSTORE
from utils import comm
from utils.collect_env import collect_env_info
from utils.logger import setup_logger
from utils.misc import mkdir
from evaluator import graph_classification_evaluation
import yaml
from easydict import EasyDict as edict

# ================= 参数定义 =================
parser = argparse.ArgumentParser(description='Graph DGL Training')
parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')
parser.add_argument("--local_rank", type=int, default=0, help="local rank")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument("--yaml_dir", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--checkpoint_dir", type=str, default=None)
args = parser.parse_args()


# ================= 训练核心函数 =================
def pretrain(model, train_loader, optimizer, device, epoch, logger, criterion):
    model.train()
    loss_list = []

    # 打印进度条前缀
    if epoch % 10 == 0:
        print(f"Epoch {epoch} Training...", end=" ")

    for batch in train_loader:
        batch_g_fmri, batch_g_dti, labels = batch
        batch_g_fmri, batch_g_dti, labels = batch_g_fmri.to(device), batch_g_dti.to(device), labels.to(device)

        feat_fmri = batch_g_fmri.ndata["attr"]
        feat_dti = batch_g_dti.ndata["attr"]
        community_fmri = batch_g_fmri.ndata["community"]
        community_dti = batch_g_dti.ndata["community"]

        # Forward
        loss, loss_dict = model(batch_g_fmri, batch_g_dti, feat_fmri, feat_dti, labels, community_fmri, community_dti)

        # 这里的 loss 是 BrainSTORE 内部计算的扩散 loss，不需要外部 criterion
        # 如果你想把分类 Loss 加进来，可以在这里改，目前 BrainSTORE 主要是预训练去噪 Loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

    lr = optimizer.param_groups[0]['lr']
    mean_loss = np.mean(loss_list)
    if epoch % 10 == 0:
        print(f"| Loss: {mean_loss:.4f}")  # 实时打印到控制台

    if logger and epoch % 10 == 0:
        logger.info(f"Epoch {epoch} | train_loss: {mean_loss:.4f} | lr: {lr:.6f}")


def collate_fn(batch):
    graphs_fmri = [x[0] for x in batch]
    graphs_dti = [x[1] for x in batch]
    labels = [torch.tensor(x[2], dtype=torch.long) for x in batch]

    batch_g_fmri = dgl.batch(graphs_fmri)
    batch_g_dti = dgl.batch(graphs_dti)
    labels = torch.stack(labels)

    return batch_g_fmri, batch_g_dti, labels


def save_checkpoint(state, is_best, filename, fold_idx):
    # 分折保存模型，防止覆盖
    fold_dir = osp.join(filename, f'fold_{fold_idx}')
    if not osp.exists(fold_dir):
        mkdir(fold_dir)
    ckp = osp.join(fold_dir, 'checkpoint.pth.tar')
    torch.save(state, ckp)
    if is_best:
        shutil.copyfile(ckp, osp.join(fold_dir, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, alpha, decay, lr):
    lr = lr * (alpha ** (epoch // decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# ================= 主函数 (修复版) =================
def main(cfg):
    print(">>> 正在初始化环境...")
    if cfg.output_dir:
        mkdir(cfg.output_dir)
        mkdir(cfg.checkpoint_dir)

    logger = setup_logger("graph", cfg.output_dir, comm.get_rank(), filename='train_log.txt')
    logger.info("Command line arguments: " + str(args))

    # 备份代码
    shutil.copyfile('./params.yaml', cfg.output_dir + '/params.yaml')
    shutil.copyfile('./main_graph.py', cfg.output_dir + '/graph.py')

    # === 修复点 1: 获取随机种子 ===
    # 你的 yaml 里写的是 seeds (列表)，我们取第一个作为当前的种子
    if hasattr(cfg, 'seeds') and isinstance(cfg.seeds, list) and len(cfg.seeds) > 0:
        current_seed = cfg.seeds[0]
    else:
        current_seed = 1234  # 默认值兜底

    print(f">>> 使用随机种子: {current_seed}")
    # 设置全局种子 (确保结果可复现)
    set_random_seed(current_seed)

    # === 1. 数据集配置 ===
    print(">>> 开始加载数据集 (这可能需要几秒钟)...")

    # 原始数据
    raw_fmri = 'datasets/ADNI/adni_fmri.mat'
    raw_dti = 'datasets/ADNI/adni_dti.mat'
    labels_path = 'datasets/ADNI/adni_labels.mat'

    # 处理后的数据 (Structure)
    struct_fmri = 'datasets/ADNI/adni_fmri_divide/fmri_processed_lambda1.0.mat'
    struct_dti = 'datasets/ADNI/adni_dti_divide/dti_processed_lambda2.0.mat'

    # 社区标签 (Partition)
    part_fmri = 'datasets/ADNI/adni_fmri_divide/fmri_partition_lambda1.0.mat'
    part_dti = 'datasets/ADNI/adni_dti_divide/dti_partition_lambda2.0.mat'

    # 加载数据
    graphs, (dim_f, dim_d, n_classes) = load_multimodal_graph_classification_dataset(
        raw_fmri_path=raw_fmri, raw_dti_path=raw_dti,
        struct_fmri_path=struct_fmri, struct_dti_path=struct_dti,
        part_fmri_path=part_fmri, part_dti_path=part_dti,
        labels_filepath=labels_path
    )

    print(f"\n>>> 数据集信息:")
    print(f"    fMRI 特征维数: {dim_f}")
    print(f"    DTI 特征维数:  {dim_d}")
    print(f"    分类任务:      {n_classes} 分类")

    # === 2. 计算类别权重 ===
    all_labels = [g[2].item() for g in graphs]
    class_counts = np.bincount(all_labels)
    total_samples = len(all_labels)
    # 防止除以0
    safe_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = torch.tensor([total_samples / (n_classes * c) for c in safe_counts], dtype=torch.float).to(
        cfg.DEVICE)
    print(f"    类别分布: {class_counts}")
    print(f"    类别权重: {class_weights.cpu().numpy()}")

    # === 3. K-Fold 交叉验证 ===
    # === 修复点 2: 这里使用 current_seed ===
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=current_seed)

    acc_list = []
    labels_array = np.array(all_labels)

    print("\n>>> 开始 5-Fold 交叉验证训练...")

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.zeros(len(labels_array)), labels_array)):
        logger.info(f"\n{'=' * 20} Fold {fold_idx + 1} / 5 {'=' * 20}")

        # 再次设置种子，确保每个 fold 的初始化一致性 (可选)
        set_random_seed(current_seed)

        # 构建 Loader
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        # --- 核心修改：加入 num_workers ---
        train_loader = GraphDataLoader(graphs, sampler=train_sampler, collate_fn=collate_fn,
                                       batch_size=cfg.DATALOADER.BATCH_SIZE,
                                       pin_memory=True,
                                       num_workers=cfg.DATALOADER.NUM_WORKERS)  # <--- 这里加上

        val_loader = GraphDataLoader(graphs, sampler=val_sampler, collate_fn=collate_fn,
                                     batch_size=len(val_idx),  # 或者 cfg.DATALOADER.BATCH_SIZE
                                     shuffle=False,
                                     num_workers=cfg.DATALOADER.NUM_WORKERS)  # <--- 这里加上
        # 初始化模型
        ml_cfg = cfg.MODEL
        ml_cfg.update({'in_dim_fmri': dim_f, 'in_dim_dti': dim_d})
        model = BrainSTORE(**ml_cfg).to(cfg.DEVICE)

        # === 新增代码 1: 计算参数量 ===
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'=' * 40}")
        print(f"Parameter No. (Total):     {total_params}")
        print(f"Parameter No. (Trainable): {trainable_params}")
        print(f"{'=' * 40}\n")
        logger.info(f"Model Parameters: {total_params}")
        # ==========================

        optimizer = create_optimizer(cfg.SOLVER.optim_type, model, cfg.SOLVER.LR, cfg.SOLVER.weight_decay)

        # 将权重传入 Loss (用于 Evaluation 阶段的计算，如果 graph_classification_evaluation 支持的话)
        # 注意：BrainSTORE 内部主要是去噪 loss，分类性能主要看 evaluation 函数
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_f1 = 0
        pooler = create_pooler(cfg.MODEL.pooler)

        for epoch in range(cfg.SOLVER.MAX_EPOCH):
            # === 新增代码 2: 计时开始 & 重置显存统计 ===
            epoch_start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            # ========================================
            adjust_learning_rate(optimizer, epoch=epoch, alpha=cfg.SOLVER.alpha, decay=cfg.SOLVER.decay,
                                 lr=cfg.SOLVER.LR)

            # 训练
            pretrain(model, train_loader, optimizer, cfg.DEVICE, epoch, logger, criterion)

            # === 新增代码 3: 计时结束 & 获取显存 & 打印 ===
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            # 获取本轮最大显存占用 (MB)
            if torch.cuda.is_available():
                max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            else:
                max_memory = 0.0

            # 打印像你给的例子那样的日志
            print(f"Running time (s/epoch): {epoch_duration:.4f}s | Memory (MB): {max_memory:.2f}")
            logger.info(f"Epoch {epoch} Stat | Time: {epoch_duration:.4f}s | Mem: {max_memory:.2f}MB")
            # ============================================

            # 评估 (修改这里：graph_classification_evaluation 可能需要适配)
            if (epoch + 1) % 5 == 0 or epoch == cfg.SOLVER.MAX_EPOCH - 1:
                model.eval()
                # 假设 graph_classification_evaluation 返回 F1
                test_f1 = graph_classification_evaluation(model, cfg.eval_T, pooler, val_loader, cfg.DEVICE, logger,
                                                          epoch)

                if test_f1 > best_f1:
                    best_f1 = test_f1
                    # 保存带 fold 编号的模型
                    save_checkpoint({'state_dict': model.state_dict(), 'best_f1': best_f1},
                                    True, cfg.checkpoint_dir, fold_idx + 1)
                    print(f"    [Fold {fold_idx + 1}] New Best F1: {best_f1:.4f}")

        acc_list.append(best_f1)
        logger.info(f"Fold {fold_idx + 1} Finished. Best F1: {best_f1:.4f}")

    final_acc, final_std = np.mean(acc_list), np.std(acc_list)
    logger.info(f"\n{'=' * 40}")
    logger.info(f"Final 5-Fold Result: F1 = {final_acc:.4f} ± {final_std:.4f}")
    logger.info(f"{'=' * 40}")


if __name__ == "__main__":
    # 配置加载逻辑
    print("Checking CUDA...")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    root_dir = osp.abspath(osp.dirname(__file__))
    yaml_dir = osp.join(root_dir, 'params.yaml')
    output_dir = osp.join(root_dir, 'log')
    checkpoint_dir = osp.join(output_dir, "checkpoint")

    if args.yaml_dir: yaml_dir = args.yaml_dir
    if args.output_dir: output_dir = args.output_dir
    if args.checkpoint_dir: checkpoint_dir = args.checkpoint_dir

    with open(yaml_dir, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    cfg = edict(config)

    cfg.output_dir = output_dir
    cfg.checkpoint_dir = checkpoint_dir

    main(cfg)
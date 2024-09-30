#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File Name:     main_graph.py
# Author:        wangchenze
# Created Time:  202409
# Last Modified: <none>-<none>
import numpy as np

import argparse

import shutil
import time
import os.path as osp

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl.dataloading import GraphDataLoader
from dgl import RandomWalkPE

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import SequentialSampler
import torch.nn as nn
from dgl.nn.functional import edge_softmax

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from utils.utils import (create_optimizer, create_pooler, set_random_seed, compute_ppr)

from datasets.data_util import load_multimodal_graph_classification_dataset

from models import SDiff

import multiprocessing
from multiprocessing import Pool


from utils import comm
from utils.collect_env import collect_env_info
from utils.logger import setup_logger
from utils.misc import mkdir

from evaluator import graph_classification_evaluation
import yaml
from easydict import EasyDict as edict


parser = argparse.ArgumentParser(description='Graph DGL Training')
parser.add_argument('--resume', '-r', action='store_true', default=False,
                    help='resume from checkpoint')
parser.add_argument("--local_rank", type=int, default=0, help="local rank")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument("--yaml_dir", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--checkpoint_dir", type=str, default=None)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
args = parser.parse_args()

def pretrain(model, train_loader, optimizer, device, epoch, logger=None):
    model.train()
    loss_list = []
    for batch in train_loader:
        # 拆解出两个模态的图
        batch_g_fmri, batch_g_dti, labels = batch
        batch_g_fmri, batch_g_dti, labels = batch_g_fmri.to(device), batch_g_dti.to(device), labels.to(device)

        # 模态1特征
        feat_fmri = batch_g_fmri.ndata["attr"]
        # 模态2特征
        feat_dti = batch_g_dti.ndata["attr"]

        community_fmri = batch_g_fmri.ndata["community"]
        # 模态2特征
        community_dti = batch_g_dti.ndata["community"]
        # 调用模型的 forward，传入多模态图
        loss, loss_dict = model(batch_g_fmri, batch_g_dti, feat_fmri, feat_dti, labels,community_fmri,community_dti)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

    lr = optimizer.param_groups[0]['lr']
    if logger:
        logger.info(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f} | lr: {lr:.6f}")
def collate_fn(batch, batch_size=16):
    # 分别获取 fMRI 图，DTI 图 和 标签
    graphs_fmri = [x[0] for x in batch]
    graphs_dti = [x[1] for x in batch]
    labels = [torch.tensor(x[2], dtype=torch.long).unsqueeze(0) for x in batch]  # 确保标签是一维张量，并添加维度

    # 如果 batch 大小小于目标大小，补充样本
    if len(batch) < batch_size:
        remaining = batch_size - len(batch)
        # 从现有 batch 开头补充所需数量的样本
        graphs_fmri += graphs_fmri[:remaining]
        graphs_dti += graphs_dti[:remaining]
        labels += labels[:remaining]

    # 批处理图
    batch_g_fmri = dgl.batch(graphs_fmri)
    batch_g_dti = dgl.batch(graphs_dti)
    # 将标签拼接为一维张量
    labels = torch.cat(labels, dim=0).squeeze()  # 移除多余的维度
    return batch_g_fmri, batch_g_dti, labels


def save_checkpoint(state, is_best, filename):
    ckp = osp.join(filename, 'checkpoint.pth.tar')
    # ckp = filename + "checkpoint.pth.tar"
    torch.save(state, ckp)
    if is_best:
        shutil.copyfile(ckp, filename+'/model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, alpha, decay, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 80 epochs"""
    lr = lr * (alpha ** (epoch // decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

import dgl

def main(cfg):
    print(dir(dgl.nn))
    best_f1 = float('-inf')
    best_f1_epoch = float('inf')

    if cfg.output_dir:
        mkdir(cfg.output_dir)
        mkdir(cfg.checkpoint_dir)

    logger = setup_logger("graph", cfg.output_dir, comm.get_rank(), filename='train_log.txt')
    logger.info("Rank of current process: {}. World size: {}".format(comm.get_rank(), comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())
    logger.info("Command line arguments: " + str(args))

    shutil.copyfile('./params.yaml', cfg.output_dir + '/params.yaml')
    shutil.copyfile('./main_graph.py', cfg.output_dir + '/graph.py')
    shutil.copyfile('models/SDiff.py', cfg.output_dir + '/SDiff.py')
    shutil.copyfile('./models/mlp_gat.py', cfg.output_dir + '/mlp_gat.py')

    #
    # fmri_filepath = 'datasets/dataOld/fmri1.mat'
    # labels_filepath = 'datasets/dataOld/labels1.mat'
    # dti_filepath= 'datasets/dataOld/dti1.mat'

    fmri_filepath = 'datasets/ppmi/fcnHCPD.mat'
    labels_filepath = 'datasets/ppmi/labHCPD.mat'
    dti_filepath= 'datasets/ppmi/scnHCPD.mat'
    # graphs, (num_features, num_classes) = load_graph_classification_dataset(fmri_filepath,
    #                                                                         labels_filepath,
    #                                                                         deg4feat=False)
    graphs, (feature_dim_fmri, feature_dim_dti, num_classes) = load_multimodal_graph_classification_dataset(fmri_filepath, dti_filepath, labels_filepath,deg4feat=False)

    print(f"Number of feature_dim_fmri→features: {feature_dim_fmri}")
    print(f"Number of feature_dim_dti→features: {feature_dim_dti}")
    print(f"Number of classes: {num_classes}")
    cfg.num_features = feature_dim_fmri
    train_idx = torch.arange(len(graphs))
    train_sampler = SubsetRandomSampler(train_idx)
    # train_sampler = SequentialSampler(train_idx)
    train_loader = GraphDataLoader(graphs, sampler=train_sampler, collate_fn=collate_fn,
                                   batch_size=cfg.DATALOADER.BATCH_SIZE, pin_memory=True)
    eval_loader = GraphDataLoader(graphs, collate_fn=collate_fn, batch_size=len(graphs), shuffle=False)
    pooler = create_pooler(cfg.MODEL.pooler)

    acc_list = []
    for i, seed in enumerate(cfg.seeds):
        logger.info(f'Run {i}th for seed {seed}')
        set_random_seed(seed)
        ml_cfg = cfg.MODEL
        ml_cfg.update({'in_dim_fmri': feature_dim_fmri})
        ml_cfg.update({'in_dim_dti': feature_dim_dti})
        model = SDiff(**ml_cfg)
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info('Total trainable params num : {}'.format(total_trainable_params))
        model.to(cfg.DEVICE)

        optimizer = create_optimizer(cfg.SOLVER.optim_type, model, cfg.SOLVER.LR, cfg.SOLVER.weight_decay)

        start_epoch = 0
        if args.resume:
            if osp.isfile(cfg.pretrain_checkpoint_dir):
                logger.info("=> loading checkpoint '{}'".format(cfg.checkpoint_dir))
                checkpoint = torch.load(cfg.checkpoint_dir, map_location=torch.device('cpu'))
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                logger.info("=> loaded checkpoint '{}' (epoch {})"
                            .format(cfg.checkpoint_dir, checkpoint['epoch']))

        logger.info("----------Start Training----------")

        for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
            adjust_learning_rate(optimizer, epoch=epoch, alpha=cfg.SOLVER.alpha, decay=cfg.SOLVER.decay, lr=cfg.SOLVER.LR)
            pretrain(model, train_loader, optimizer, cfg.DEVICE, epoch, logger)
            if ((epoch + 1) % 1 == 0) & (epoch > 1):
                model.eval()
                test_f1 = graph_classification_evaluation(model, cfg.eval_T, pooler, eval_loader,
                                                          cfg.DEVICE, logger,epoch,save_tsne=True)
                is_best = test_f1 > best_f1
                if is_best:
                    best_f1_epoch = epoch
                best_f1 = max(test_f1, best_f1)
                logger.info(f"Epoch {epoch}: get test f1 score: {test_f1: .3f}")
                logger.info(f"best_f1 {best_f1:.3f} at epoch {best_f1_epoch}")
                save_checkpoint({'epoch': epoch + 1,
                                 'state_dict': model.state_dict(),
                                 'best_f1': best_f1,
                                 'optimizer': optimizer.state_dict()},
                                is_best, filename=cfg.checkpoint_dir)
        acc_list.append(best_f1)
    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    logger.info((f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}"))
    return final_acc

import dgl
import torch


if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print("DGL version:", dgl.__version__)
    root_dir = osp.abspath(osp.dirname(__file__))
    yaml_dir = osp.join(root_dir, 'params.yaml')
    output_dir = osp.join(root_dir, 'log')
    checkpoint_dir = osp.join(output_dir, "checkpoint")
    yaml_dir = args.yaml_dir if args.yaml_dir else yaml_dir
    output_dir = args.output_dir if args.output_dir else output_dir
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else checkpoint_dir

    with open(yaml_dir, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    cfg = edict(config)

    cfg.output_dir, cfg.checkpoint_dir = output_dir, checkpoint_dir
    print(cfg)
    f1 = main(cfg)














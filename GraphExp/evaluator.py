import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from multiprocessing import Pool

import time  # <--- 记得在文件最开头导入 time


def graph_classification_evaluation(model, T, pooler, dataloader, device, logger, epoch, save_tsne=True):
    model.eval()
    embed_list = []
    y_list = []

    # === 新增变量：用于统计推理时间 ===
    total_inference_time = 0
    total_samples = 0
    # ==============================

    with torch.no_grad():
        for t_idx, t in enumerate(T):
            x_list_t = []
            y_list_t = []

            for i, (batch_g_fmri, batch_g_dti, labels) in enumerate(dataloader):
                batch_g_fmri = batch_g_fmri.to(device)
                batch_g_dti = batch_g_dti.to(device)

                feat_fmri = batch_g_fmri.ndata["attr"]
                feat_dti = batch_g_dti.ndata["attr"]
                comm_fmri = batch_g_fmri.ndata["community"]
                comm_dti = batch_g_dti.ndata["community"]

                # === 计时开始 ===
                # 只在第一个时间步 t 统计时间即可 (避免重复统计)
                if t_idx == 0:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()  # 确保 GPU 同步，计时更准
                    start_time = time.time()

                # 前向传播提取特征
                out, finally_batch_g_eval = model.embed(
                    batch_g_fmri, batch_g_dti, feat_fmri, feat_dti, t, comm_fmri, comm_dti
                )
                # 池化得到图级特征
                out = pooler(finally_batch_g_eval, out)

                # === 计时结束 ===
                if t_idx == 0:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_time = time.time()

                    batch_size = labels.shape[0]
                    total_inference_time += (end_time - start_time)
                    total_samples += batch_size
                # =================

                y_list_t.append(labels)
                x_list_t.append(out)

            # 拼接当前时间步的结果
            embed_list.append(torch.cat(x_list_t, dim=0).cpu().numpy())

        # 标签只需取一次
        y_list = torch.cat(y_list_t, dim=0).cpu().numpy()

    # === 新增代码：打印推理时间 ===
    if total_samples > 0:
        # 计算每个样本的平均时间 (秒 -> 毫秒)
        avg_time_per_sample = (total_inference_time / total_samples) * 1000
        logger.info(f"[Inference Speed] Total Samples: {total_samples} | "
                    f"Total Time: {total_inference_time:.4f}s | "
                    f"Per Sample: {avg_time_per_sample:.4f} ms")
        print(f"Inference Time (per sample): {avg_time_per_sample:.4f} ms")
    # ===========================

    # 2. 合并不同时间步的嵌入
    embed_list = np.stack(embed_list, axis=1)
    embed_list = np.reshape(embed_list, (embed_list.shape[0], -1))

    # print(f"Embeddings Shape: {embed_list.shape} | Labels Shape: {y_list.shape}")

    # 3. PCA 降维
    n_comp = min(88, embed_list.shape[0], embed_list.shape[1])
    pca = PCA(n_components=n_comp)
    reduced_embed_list = pca.fit_transform(embed_list)

    # 4. 保存 t-SNE 数据 (可选)
    if save_tsne:
        save_dir = 'tsne/adni'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        tsne_data = {
            'y': y_list,
            'embedding': reduced_embed_list
        }
        np.save(os.path.join(save_dir, f'tsne_epoch_Test{epoch}.npy'), tsne_data)

    # 5. 使用 SVM 进行评估
    test_f1, test_std = evaluate_graph_embeddings_using_svm(reduced_embed_list, y_list)

    logger.info(f"[Epoch {epoch}] Test F1: {test_f1:.4f} ± {test_std:.4f}")

    return test_f1


def inner_func(args):
    """
    Worker function for Multiprocessing SVM Evaluation (Multiclass Compatible)
    """
    train_index, test_index, embed_list, y_list = args

    x_train = embed_list[train_index]
    x_test = embed_list[test_index]
    y_train = y_list[train_index]
    y_test = y_list[test_index]

    # SVM Parameter Grid
    params = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    }

    # Initialize SVM with probability for AUC
    svc = SVC(random_state=42, probability=True, class_weight='balanced')
    clf = GridSearchCV(svc, params, cv=3, n_jobs=1)  # n_jobs=1 to avoid nested parallel issues
    clf.fit(x_train, y_train)

    # Prediction
    y_pred = clf.predict(x_test)
    y_proba = clf.predict_proba(x_test)

    # --- Metrics Calculation (Multiclass Safe) ---

    # 1. Accuracy
    acc = accuracy_score(y_test, y_pred)

    # 2. Weighted Metrics (Accounts for class imbalance: 211 vs 54 vs 195)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # 3. Multiclass AUC (One-vs-Rest)
    try:
        # y_proba shape: [n_samples, n_classes]
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
    except ValueError:
        # Fallback if a class is missing in the fold
        auc = 0.5

    return acc, recall, precision, f1, auc


def evaluate_graph_embeddings_using_svm(embed_list, y_list):
    """
    Runs K-Fold Cross Validation with SVM
    """
    # 10-Fold CV
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    process_args = [(train_idx, test_idx, embed_list, y_list)
                    for train_idx, test_idx in kf.split(embed_list, y_list)]

    # Use multiprocessing
    with Pool(processes=8) as p:
        results = p.map(inner_func, process_args)

    # Unpack results
    # Each item in results is (acc, recall, precision, f1, auc)
    acc_list, recall_list, prec_list, f1_list, auc_list = zip(*results)

    # Compute Mean and Std (More rigorous than Max)
    mean_f1 = np.mean(f1_list)
    std_f1 = np.std(f1_list)

    mean_acc = np.mean(acc_list)
    mean_auc = np.mean(auc_list)

    print("-" * 30)
    print(f"SVM Evaluation Results (10-Fold Avg):")
    print(f"  Accuracy:  {mean_acc:.4f} ± {np.std(acc_list):.4f}")
    print(f"  F1 Score:  {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  AUC (OvR): {mean_auc:.4f} ± {np.std(auc_list):.4f}")
    print("-" * 30)

    return mean_f1, std_f1
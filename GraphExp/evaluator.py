import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix
import torch
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from multiprocessing import Pool
from sklearn.decomposition import PCA
import os
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
def graph_classification_evaluation(model, T, pooler, dataloader, device, logger,epoch,save_tsne=True):
    model.eval()
    embed_list = []
    head_list = []
    optim_list = []
    with torch.no_grad():
        for t in T:
            x_list = []
            y_list = []
            for i, (batch_g_fmri, batch_g_dti, labels) in enumerate(dataloader):
                batch_g_fmri = batch_g_fmri.to(device)
                batch_g_dti = batch_g_dti.to(device)
                # feat = batch_g.ndata["attr"]
                feat_fmri = batch_g_fmri.ndata["attr"]
                # 模态2特征
                feat_dti = batch_g_dti.ndata["attr"]

                community_fmri = batch_g_fmri.ndata["community"]
                # 模态2特征
                community_dti = batch_g_dti.ndata["community"]
                out,finally_batch_g_eval = model.embed(batch_g_fmri, batch_g_dti, feat_fmri, feat_dti,t,community_fmri,community_dti)
                out = pooler(finally_batch_g_eval, out)
                y_list.append(labels)
                x_list.append(out)
            head_list.append(1)
            embed_list.append(torch.cat(x_list, dim=0).cpu().numpy())
        y_list = torch.cat(y_list, dim=0)
    # 合并不同时间的嵌入
    embed_list = np.stack(embed_list, axis=1)  # 从 (3, 250, 1024) 变为 (250, 3, 1024)
    embed_list = np.reshape(embed_list, (embed_list.shape[0], -1))  # 从 (250, 3, 1024) 变为 (250, 3072)
    y_list = y_list.cpu().numpy()

    # 打印 embed_list 和 y_list 的大小
    print(f"embed_list shape: {embed_list.shape}")
    print(f"y_list shape: {y_list.shape}")

    # 使用 PCA 降维
    pca = PCA(n_components=88)  # 可以根据需要调整 n_components
    reduced_embed_list = pca.fit_transform(embed_list)
    # 打印 reduced_embed_list 的维度
    print(f"reduced_embed_list shape: {reduced_embed_list.shape}")  #250X100
    # 使用 t-SNE 进行降维
    if save_tsne:
        # 保存 t-SNE 数据
        save_dir = 'tsne/adni'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        tsne_data = {
            'y': labels.cpu().numpy(),
            'embedding': embed_list
        }
        np.save(os.path.join(save_dir, f'tsne_epoch_Test{epoch}.npy'), tsne_data)

    test_f1, test_std = evaluate_graph_embeddings_using_svm(reduced_embed_list, y_list)
    logger.info(f"#Test_f1: {test_f1:.4f}±{test_std:.4f}")
    test_f1, test_std = evaluate_graph_embeddings_using_svm(reduced_embed_list, y_list)
    logger.info(f"#Test_f1: {test_f1:.4f}±{test_std:.4f}")
    return test_f1


def inner_func(args):
    train_index = args[0]
    test_index = args[1]
    embed_list = args[2]
    y_list = args[3]

    embeddings = embed_list
    labels = y_list
    x_train = embeddings[train_index]
    x_test = embeddings[test_index]
    y_train = labels[train_index]
    y_test = labels[test_index]

    # SVM 分类器的参数设置
    params = {
        "C": [0.01, 0.1, 1, 10],  # 惩罚系数参数
        "kernel": ["linear", "rbf"],  # 核函数类型
        "gamma": ["scale", "auto"]  # 核函数的系数
    }
    svc = SVC(random_state=42, probability=True)  # 设置probability=True以计算AUC
    clf = GridSearchCV(svc, params, cv=5)
    clf.fit(x_train, y_train)

    # 预测
    out = clf.predict(x_test)
    probas = clf.predict_proba(x_test)

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, out)
    tn, fp, fn, tp = cm.ravel()

    # 计算评价指标
    acc = accuracy_score(y_test, out)
    recall = recall_score(y_test, out, average="micro")
    precision = precision_score(y_test, out, average="micro")
    sen = recall  # 敏感度等于召回率
    spe = tn / (tn + fp)  # 特异性

    # 计算 AUC
    y_score = probas[:, 1]  # 取出每个样本属于正类的预测概率
    auc = roc_auc_score(y_test, y_score)

    f1 = f1_score(y_test, out, average="micro")

    return acc, sen, spe, auc, recall, f1



def evaluate_graph_embeddings_using_svm(embed_list, y_list):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    process_args = [(train_index, test_index, embed_list, y_list)
                    for train_index, test_index in kf.split(embed_list, y_list)]
    with Pool(10) as p:
        result = p.map(inner_func, process_args)

    acc_list, sen_list, spe_list, auc_list, recall_list, f1_list = zip(*result)

    # 计算各个指标的最大值
    test_acc = np.max(acc_list)
    test_sen = np.max(sen_list)
    test_spe = np.max(spe_list)
    test_auc = np.max(auc_list)
    test_recall = np.max(recall_list)
    test_f1 = np.max(f1_list)

    # 计算 F1 Score 的标准差
    test_f1_std = np.std(f1_list)

    print(f"Maximum Accuracy: {test_acc:.4f}")
    print(f"Maximum Sensitivity: {test_sen:.4f}")
    print(f"Maximum Specificity: {test_spe:.4f}")
    print(f"Maximum AUC: {test_auc:.4f}")
    print(f"Maximum Recall: {test_recall:.4f}")
    print(f"Maximum F1 Score: {test_f1:.4f} ± {test_f1_std:.4f}")

    return test_f1, test_f1_std
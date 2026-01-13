import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from easydict import EasyDict as edict

# 导入你的模型和数据加载器
from models import BrainSTORE
from datasets.data_util import load_multimodal_graph_classification_dataset
from utils.utils import set_random_seed

# ==========================================
# 1. 纯英文 AAL 90 脑区名称 (解决服务器乱码问题)
# ==========================================
AAL_LABELS = [
    "Precentral_L", "Precentral_R", "Frontal_Sup_L", "Frontal_Sup_R",
    "Frontal_Sup_Orb_L", "Frontal_Sup_Orb_R", "Frontal_Mid_L", "Frontal_Mid_R",
    "Frontal_Mid_Orb_L", "Frontal_Mid_Orb_R", "Frontal_Inf_Oper_L", "Frontal_Inf_Oper_R",
    "Frontal_Inf_Tri_L", "Frontal_Inf_Tri_R", "Frontal_Inf_Orb_L", "Frontal_Inf_Orb_R",
    "Rolandic_Oper_L", "Rolandic_Oper_R", "Supp_Motor_Area_L", "Supp_Motor_Area_R",
    "Olfactory_L", "Olfactory_R", "Frontal_Sup_Medial_L", "Frontal_Sup_Medial_R",
    "Frontal_Mid_Orb_L", "Frontal_Mid_Orb_R", "Rectus_L", "Rectus_R",
    "Insula_L", "Insula_R", "Cingulum_Ant_L", "Cingulum_Ant_R",
    "Cingulum_Mid_L", "Cingulum_Mid_R", "Cingulum_Post_L", "Cingulum_Post_R",
    "Hippocampus_L", "Hippocampus_R", "ParaHippocampal_L", "ParaHippocampal_R",
    "Amygdala_L", "Amygdala_R", "Calcarine_L", "Calcarine_R",
    "Cuneus_L", "Cuneus_R", "Lingual_L", "Lingual_R",
    "Occipital_Sup_L", "Occipital_Sup_R", "Occipital_Mid_L", "Occipital_Mid_R",
    "Occipital_Inf_L", "Occipital_Inf_R", "Fusiform_L", "Fusiform_R",
    "Postcentral_L", "Postcentral_R", "Parietal_Sup_L", "Parietal_Sup_R",
    "Parietal_Inf_L", "Parietal_Inf_R", "SupraMarginal_L", "SupraMarginal_R",
    "Angular_L", "Angular_R", "Precuneus_L", "Precuneus_R",
    "Paracentral_Lobule_L", "Paracentral_Lobule_R", "Caudate_L", "Caudate_R",
    "Putamen_L", "Putamen_R", "Pallidum_L", "Pallidum_R",
    "Thalamus_L", "Thalamus_R", "Heschl_L", "Heschl_R",
    "Temporal_Sup_L", "Temporal_Sup_R", "Temporal_Pole_Sup_L", "Temporal_Pole_Sup_R",
    "Temporal_Mid_L", "Temporal_Mid_R", "Temporal_Pole_Mid_L", "Temporal_Pole_Mid_R",
    "Temporal_Inf_L", "Temporal_Inf_R"
]


def get_aal_labels():
    return AAL_LABELS[:90]


def main():
    # ================= 配置 =================
    yaml_path = 'params.yaml'
    checkpoint_path = 'log/checkpoint/fold_1/model_best.pth.tar'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ================= 初始化 =================
    with open(yaml_path, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    cfg = edict(config)
    set_random_seed(1234)

    print(">>> 正在加载数据集...")
    raw_fmri = 'datasets/ADNI/adni_fmri.mat'
    raw_dti = 'datasets/ADNI/adni_dti.mat'
    labels_path = 'datasets/ADNI/adni_labels.mat'
    struct_fmri = 'datasets/ADNI/adni_fmri_divide/fmri_processed_lambda1.0.mat'
    struct_dti = 'datasets/ADNI/adni_dti_divide/dti_processed_lambda2.0.mat'
    part_fmri = 'datasets/ADNI/adni_fmri_divide/fmri_partition_lambda1.0.mat'
    part_dti = 'datasets/ADNI/adni_dti_divide/dti_partition_lambda2.0.mat'

    graphs, (dim_f, dim_d, n_classes) = load_multimodal_graph_classification_dataset(
        raw_fmri_path=raw_fmri, raw_dti_path=raw_dti,
        struct_fmri_path=struct_fmri, struct_dti_path=struct_dti,
        part_fmri_path=part_fmri, part_dti_path=part_dti,
        labels_filepath=labels_path
    )

    # ================= 加载模型 =================
    print(f">>> 正在加载模型: {checkpoint_path}")
    ml_cfg = cfg.MODEL
    ml_cfg.update({'in_dim_fmri': dim_f, 'in_dim_dti': dim_d})
    model = BrainSTORE(**ml_cfg).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # ================= 计算注意力 =================
    print(">>> 开始计算全脑节点重要性 (K=10步采样)...")

    group_scores = {0: [], 1: [], 2: []}

    with torch.no_grad():
        for i, (g_fmri, g_dti, label) in enumerate(graphs):
            if i % 100 == 0:
                print(f"    处理进度: {i}/{len(graphs)}")

            label_idx = label.item()
            g_f = g_fmri.to(device)
            g_d = g_dti.to(device)

            x_f = g_f.ndata['attr']
            x_d = g_d.ndata['attr']
            comm_f = g_f.ndata['community']
            comm_d = g_d.ndata['community']

            scores_f, scores_d = model.get_node_attention(
                g_f, g_d, x_f, x_d, comm_f, comm_d, K=10
            )

            if len(scores_f) > 0:
                s_f = scores_f[0].cpu().numpy()
                s_d = scores_d[0].cpu().numpy()
                combined_score = (s_f + s_d) / 2
                group_scores[label_idx].append(combined_score)

    # ================= 数据处理与诊断 =================
    print("\n>>> 数据诊断:")
    avg_scores = {}
    for cls_idx in group_scores:
        if len(group_scores[cls_idx]) > 0:
            data = np.array(group_scores[cls_idx])
            avg = np.mean(data, axis=0)
            avg_scores[cls_idx] = avg
            print(f"   Class {cls_idx}: Mean range [{avg.min():.6e}, {avg.max():.6e}]")
        else:
            avg_scores[cls_idx] = np.zeros(90)

    # 归一化
    all_values = np.concatenate([avg_scores[0], avg_scores[1], avg_scores[2]])
    g_min, g_max = all_values.min(), all_values.max()
    print(f"   Global Range: [{g_min:.6e}, {g_max:.6e}]")

    def normalize(x):
        if g_max - g_min < 1e-9: return x
        return (x - g_min) / (g_max - g_min)

    norm_AD = normalize(avg_scores[0])
    norm_MCI = normalize(avg_scores[1])
    norm_NC = normalize(avg_scores[2])

    region_names = get_aal_labels()

    # ================= 画图 =================
    x = np.arange(90)
    plt.figure(figsize=(20, 8))

    plt.plot(x, norm_NC, label='NC (Normal)', color='green', linewidth=1.5, alpha=0.8)
    plt.plot(x, norm_MCI, label='MCI (Mild)', color='orange', linewidth=1.5, alpha=0.8)
    plt.plot(x, norm_AD, label='AD (Alzheimer)', color='red', linewidth=1.5, alpha=0.8)

    plt.title('Brain Region Importance Analysis', fontsize=16)
    plt.ylabel('Normalized Attention Score', fontsize=12)
    plt.xlabel('AAL 90 Regions', fontsize=12)
    plt.legend(loc='upper right')

    plt.xticks(x, region_names, rotation=90, fontsize=8)
    plt.xlim(-1, 90)
    plt.grid(axis='x', alpha=0.15)

    plt.tight_layout()
    plt.savefig('biomarker_analysis_lines.png', dpi=300)
    print("\n>>> 图表已保存: biomarker_analysis_lines.png")

    # ================= 输出 Top 10 =================
    diff = norm_NC - norm_AD
    top_indices = np.argsort(np.abs(diff))[-10:][::-1]

    print("\n" + "=" * 85)
    print("🏆 Top 10 Discriminative Brain Regions")
    print("   (Difference between NC and AD)")
    print("=" * 85)
    # 使用科学计数法 (.2e) 显示微小差异
    print(f"{'Rank':<5} {'Region Name':<30} {'Diff (Sci)':<12} {'Raw NC':<10} {'Raw AD':<10} {'Trend'}")
    print("-" * 85)

    for rank, idx in enumerate(top_indices):
        d_val = diff[idx]
        region = region_names[idx]

        # 获取原始值对比
        raw_nc = avg_scores[2][idx]
        raw_ad = avg_scores[0][idx]

        trend = "NC > AD" if d_val > 0 else "AD > NC"

        print(f"{rank + 1:<5} {region:<30} {abs(d_val):.2e}     {raw_nc:.2e}   {raw_ad:.2e}   {trend}")
    print("=" * 85)


if __name__ == "__main__":
    main()
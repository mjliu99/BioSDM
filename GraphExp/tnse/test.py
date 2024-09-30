import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

# 1. 加载文件
file_path = 'tsne_epoch_Test8.npy'
data = np.load(file_path, allow_pickle=True).item()  # 使用 allow_pickle=True 以加载 Python 对象

# 2. 提取标签和嵌入特征
labels = data['y']
bottleneck = data['embedding']

# 3. 如果嵌入特征尚未降维，使用 t-SNE 进行降维
# 确保 bottleneck 和 labels 在 CPU 上
if isinstance(bottleneck, torch.Tensor):
    bottleneck = bottleneck.detach().cpu().numpy()  # 使用 detach() 来确保不需要梯度计算
if isinstance(labels, torch.Tensor):
    labels = labels.cpu().numpy()

# 检查并进行降维
if bottleneck.shape[1] > 2:
    tsne = TSNE(n_components=2, random_state=0)
    bottleneck_tsne = tsne.fit_transform(bottleneck.reshape(bottleneck.shape[0], -1))  # Flatten if needed
else:
    bottleneck_tsne = bottleneck.reshape(bottleneck.shape[0], -1)  # 如果已经是二维的

# 4. 可视化 t-SNE 结果
plt.figure(figsize=(10, 8))
scatter = plt.scatter(bottleneck_tsne[:, 0], bottleneck_tsne[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.colorbar(scatter, label='Labels')
plt.title('t-SNE Visualization of Bottleneck Features')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()

import h5py
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 打开 HDF5 文件
file_path = 'libero_goal/umap_norm_1000.h5'  # 请替换为你的文件路径

views = []
langs = []
view_labels = []
lang_labels = []

with h5py.File(file_path, 'r') as f:
    main_group = f['representations']
    for mean_group_name in main_group:
        mean_group = main_group[mean_group_name]
        view = mean_group['view'][()]
        lang = mean_group['lang'][()]
        label = mean_group['label'][()]
        
        views.append(view)
        langs.append(lang)
        view_labels.append(label.decode("utf-8"))
        lang_labels.append(label.decode("utf-8"))

# 使用UMAP降维
reducer = umap.UMAP(n_components=3)
view_embedding = reducer.fit_transform(np.array(views))
lang_embedding = reducer.fit_transform(np.array(langs))

# 设置颜色
unique_labels = list(set(view_labels + lang_labels))
colors = sns.color_palette('hsv', len(unique_labels))
color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

# 绘图
plt.figure(figsize=(12, 8))
for label in unique_labels:
    view_idx = [i for i, l in enumerate(view_labels) if l == label]
    lang_idx = [i for i, l in enumerate(lang_labels) if l == label]
    
    plt.scatter(view_embedding[view_idx, 0], view_embedding[view_idx, 1], 
                color=color_map[label], alpha=0.7, marker='o', s=25)
    plt.scatter(lang_embedding[lang_idx, 0], lang_embedding[lang_idx, 1], 
                color=color_map[label], alpha=0.7, marker='x', s=25)

plt.title('UMAP projection of HDF5 data')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.savefig(file_path.replace('.h5', '.png'))

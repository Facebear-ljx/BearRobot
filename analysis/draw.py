import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import torch

# 读取npz 文件
file = "libero130_15k_64/mean_embeddings_final.npz"
data = np.load(file)

# 获取各自的均值数据
mean_traj = data['mean_traj_emb']
mean_lang = data['mean_lang_emb']
mean_noise = data['mean_noise_emb']
cos_sim = data['cos_sim']
gau = torch.randn(1024) * torch.sqrt(torch.tensor(2.3e-10))
# man = np.random.uniform(0, 1, 1024)
# man += mean_traj
man = gau.numpy() + mean_lang

print("var noise: ", np.var(mean_noise))
print("std noise: ", np.std(mean_noise))
print("mean noise: ", np.mean(mean_noise))

# 打印总数
count = data['count']
print("total count: ", count)

# 创建 x 轴
x = np.arange(1, 1025)

# 创建光滑曲线
x_smooth = np.linspace(x.min(), x.max(), 300) 

# 使用 B 样条插值创建平滑曲线
traj_spline = make_interp_spline(x, mean_traj, k=3)
lang_spline = make_interp_spline(x, mean_lang, k=3)
noise_spline = make_interp_spline(x, mean_noise, k=3)
man_spline = make_interp_spline(x, man, k=3)

mean_traj_smooth = traj_spline(x_smooth)
mean_lang_smooth = lang_spline(x_smooth)
mean_noise_smooth = noise_spline(x_smooth)
man_smooth = man_spline(x_smooth)

# 创建图形和两个子图
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(12, 16))

# 绘制 mean_traj
ax1.plot(x_smooth, mean_traj_smooth, label='Mean Traj Embedding', color='blue')
ax1.set_title('Mean Trajectory Embedding')
ax1.set_ylabel('Value')
ax1.legend()

# 绘制 mean_lang
ax3.plot(x_smooth, mean_lang_smooth, label='Mean Lang Embedding', color='green')
ax3.set_title('Mean Language Embedding')
ax3.set_xlabel('Dimension')
ax3.set_ylabel('Value')
ax3.legend()

# 绘制 mean_noise
ax2.plot(x_smooth, mean_noise_smooth, label='Mean Noise Embedding', color='red')
ax2.set_title('Mean Noise Embedding')
ax2.set_xlabel('Dimension')
ax2.set_ylabel('Value')
ax2.legend()

ax4.plot(x_smooth, man_smooth, label='Mean Lang Embedding', color='black')
ax4.set_title('manual traj+noise Embedding')
ax4.set_xlabel('Dimension')
ax4.set_ylabel('Value')
ax4.legend()



# cos similarity

# # 显示图形
# plt.tight_layout()
# plt.savefig(file.replace('.npz', '.png'))


# # 计算均值和方差
# plt.figure()
# mean_value = np.mean(cos_sim)
# std_deviation = np.std(cos_sim)

# # 打印均值和方差
# print("Mean:", mean_value)
# print("Standard Deviation:", std_deviation)

# # 展平数组以进行直方图分析
# flattened_arr = cos_sim.flatten()

# # 生成直方图
# plt.hist(flattened_arr, bins=50, alpha=0.75, color='blue', edgecolor='black')

# # 添加标题和标签
# plt.title('Histogram of Array Elements')
# plt.xlabel('Value')
# plt.ylabel('Frequency')

# # 显示均值和方差在图上的位置
# plt.axvline(mean_value, color='k', linestyle='dashed', linewidth=1)
# plt.text(mean_value * 1.1, plt.ylim()[1] * 0.9, 'Mean: {:.2f}'.format(mean_value))

# plt.axvline(mean_value + std_deviation, color='r', linestyle='dashed', linewidth=1)
# plt.axvline(mean_value - std_deviation, color='r', linestyle='dashed', linewidth=1)
# plt.text(mean_value + std_deviation * 1.1, plt.ylim()[1] * 0.8, 'Std Dev: {:.2f}'.format(std_deviation))

# # 显示图像
# plt.savefig(file.replace('.npz', '_cos_sim.png'))

import numpy as np
import os 
import re
import matplotlib.pyplot as plt

dir_name = 'libero130_15k_64'

def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return [int(num) for num in numbers]

files = sorted(os.listdir(dir_name), key=numerical_sort)

for f in files:
    if f.endswith('final.npz'):
        print(f"-------------------------{f}-------------------------")
        data = np.load(os.path.join(dir_name, f))
        mean_traj_emb = data['mean_traj_emb']
        mean_lang_emb = data['mean_lang_emb']
        # count = data['count']
        # mean_noise_emb = data['mean_noise_emb']
        # var_noise_emb = data['var_noise_emb'] 
        # noise_cos = data['cos_sim']
        
        # 获取前 k 个最大值及其索引
        k = 10
        # indices = np.argpartition(mean_traj_emb, -k)[-k:]
        # top_k_values = mean_traj_emb[indices]

        # # 排序以便结果更直观
        # sorted_indices = indices[np.argsort(-top_k_values)]
        # sorted_top_k_values = mean_traj_emb[sorted_indices]

        # print(" traj Top k values:", sorted_top_k_values)
        # print(" traj Indices of top k values:", sorted_indices)
        

        # indices = np.argpartition(mean_lang_emb, -k)[-k:]
        # top_k_values = mean_lang_emb[indices]

        # # 排序以便结果更直观
        # sorted_indices = indices[np.argsort(-top_k_values)]
        # sorted_top_k_values = mean_lang_emb[sorted_indices]

        # print(" lang Top k values:", sorted_top_k_values)
        # print(" lang Indices of top k values:", sorted_indices)
        
        # mean_traj_emb_max = np.max(mean_traj_emb)
        # mean_traj_emb_min = np.min(mean_traj_emb)
        # mean_traj_emb_median = np.median(mean_traj_emb)
        
        # mean_lang_emb_max = np.max(mean_lang_emb)
        # mean_lang_emb_min = np.min(mean_lang_emb)
        # mean_lang_emb_median = np.median(mean_lang_emb)
        
        # 计算差值
        differences = mean_traj_emb - mean_lang_emb

        # 计算均值和标准差
        mean_diff = np.mean(differences, axis=0)
        std_diff = np.std(differences, axis=0)

        # 生成与差值同形状的高斯噪声
        gaussian_noise = np.random.normal(mean_diff, std_diff, size=(differences.shape[0], 1024))

        # 打印结果
        print("Mean of differences:", mean_diff)
        print("Standard deviation of differences:", std_diff)
        print("Gaussian noise shape:", gaussian_noise.shape)
        
        id_max = np.argmin(mean_lang_emb)
        print(f"Max value: {mean_lang_emb[id_max]}")
        print(f"Index: {id_max}")
        
        # 计算均值和方差
        mean_value = np.mean(mean_lang_emb)
        std_deviation = np.std(mean_lang_emb)

        # 创建一个新的画布
        plt.figure()

        # 绘制数组的值
        plt.plot(mean_lang_emb, marker='o', linestyle='-', color='blue', label='Values')

        # 添加标题和标签
        plt.title('Values of 1024-Dimensional Array')
        plt.xlabel('Dimension')
        plt.ylabel('Value')

        # 标注均值和方差
        plt.axhline(mean_value, color='k', linestyle='dashed', linewidth=1, label=f'Mean: {mean_value:.2f}')
        plt.axhline(mean_value + std_deviation, color='r', linestyle='dashed', linewidth=1, label=f'Std Dev: +{std_deviation:.2f}')
        plt.axhline(mean_value - std_deviation, color='r', linestyle='dashed', linewidth=1, label=f'Std Dev: -{std_deviation:.2f}')

        # 添加图例
        plt.legend()
        plt.savefig(dir_name+'/'+f.replace('.npz', '_mean_lang.png'))
        
        # mean_noise_emb_max = np.max(mean_noise_emb)
        # mean_noise_emb_min = np.min(mean_noise_emb)
        # mean_noise_emb_median = np.median(mean_noise_emb)
        
        # var_noise_emb_max = np.max(var_noise_emb)
        # var_noise_emb_min = np.min(var_noise_emb)
        # var_noise_emb_median = np.median(var_noise_emb)
        
        # cos_sim_max = np.max(noise_cos)
        # cos_sim_min = np.min(noise_cos)
        # cos_sim_median = np.median(noise_cos)
        
        # print(f"Mean Trajectory Embedding: {mean_traj_emb_max}, {mean_traj_emb_min}, {mean_traj_emb_median}")
        # print(f"Mean Language Embedding: {mean_lang_emb_max}, {mean_lang_emb_min}, {mean_lang_emb_median}")
        # print(f"Mean Noise Embedding: {mean_noise_emb_max}, {mean_noise_emb_min}, {mean_noise_emb_median}")
        # print(f"Variance Noise Embedding: {var_noise_emb_max}, {var_noise_emb_min}, {var_noise_emb_median}")
        # print(f"Cosine Similarity: {cos_sim_max}, {cos_sim_min}, {cos_sim_median}")
        # print(f"Count: {count}")

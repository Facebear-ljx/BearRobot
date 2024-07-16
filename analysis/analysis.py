import os 
import numpy as np
import argparse
import torch
import sys
import torch.nn.functional as F
import h5py
from tqdm import tqdm
from itertools import count
import umap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from BearRobot.utils.dataset.dataloader import AIRKitchenDataLoader
from BearRobot.Net.encoder.DecisionNCE import DecisionNCE_encoder, DecisionNCE_lang, DecisionNCE_visual_diff

LIBERO_DATASETS = {'libero_goal': ["libero_goal"],
                   "libero_object": ["libero_object"],
                   "libero_spatial": ["libero_spatial"],
                   "libero_10": ["libero_10"],
                   "libero_90": ["libero_90"],
                   "libero30": ["libero_goal", "libero_object", "libero_spatial"],
                   "libero130": ["libero_goal", "libero_object", "libero_spatial", "libero_10", "libero_90"]}


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_name', default='libero130', type=str, help='dataset name')
    parser.add_argument('--nce_model', default='DecisionNCE-T_libero_360ep', type=str, help='NCE model')
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    parser.add_argument('--steps', default=1000, type=int, help='steps')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--umap', default=True, type=bool, help='umap analysis')

    args = parser.parse_args()
    return args

def generator():
    while True:
        yield

@torch.no_grad()
def main(args):
    # mkdir
    directory_path = f'{args.dataset_name}'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory {directory_path} created.")
    else:
        print(f"Directory {directory_path} already exists.")
    
    # dataset and dataloader
    view_list = ['D435_image', 'wrist_image']
    img_goal = True  
    dataset_name = args.dataset_name
    json_path = f'/home/dodo/ljx/BearRobot/data/libero/{args.dataset_name}-ac.json'
    air_dataloader, _ = AIRKitchenDataLoader(
        base_dir='/data/',
        datalist=[json_path],
        view_list=view_list,
        img_goal=img_goal,
        batch_size=args.batch_size,
        one_demo=True
    )
    
    # DecisionNCE
    mm_encoder = DecisionNCE_encoder(args.nce_model, device=args.device)
    nce_encoder = DecisionNCE_visual_diff(mm_encoder)    
    
    # mean value
    mean_lang_emb = None
    mean_traj_emb = None
    mean_noise_emb = None
    var_noise_emb = None
    mean_cos_sim = None
    count = 0
    
    # umap
    lang_embs = np.empty((0, 1024))
    traj_embs = np.empty((0, 1024))
    labels = []
    label_dict = {}
    label_count = 0
    num_instructions = 10    
    
    # load mean
    data = np.load("/home/dodo/.zh1hao_space/bear_branch/BearRobot/analysis/libero130/mean_embeddings_final.npz")
    avg_taj = torch.from_numpy(data['mean_traj_emb']).to('cpu')
    avg_lang = torch.from_numpy(data['mean_lang_emb']).to('cpu')

    # process data
    epoch = 0
    air_dataloader.sampler.set_epoch(epoch)
    iterator = iter(air_dataloader)
    with tqdm(range(0,args.steps)) as pbar:
        for step in pbar:
            # iterate over the dataset
            try:
                batch = next(iterator)
            except:
                break
        
            # get data
            lang = batch['lang']
            img_begin = batch["img_begin"].to(args.device)
            img_end =  batch["img_end"].to(args.device)
            
            # get embeddings
            lang_emb = nce_encoder.embed_text(lang).to('cpu').detach()
            traj_emb = nce_encoder.embed_frame(img_begin,img_end).to('cpu').detach()
            lang_emb -= avg_lang
            traj_emb -= avg_taj
            # lang_emb = F.normalize(lang_emb, dim=-1)
            # traj_emb = F.normalize(traj_emb, dim=-1)
            noise_emb = lang_emb - traj_emb
            cos_sim = F.cosine_similarity(noise_emb[1:], noise_emb[:-1], dim=-1)

            # update mean
            if count == 0:
                count += 1
                mean_lang_emb = lang_emb.detach().clone()
                mean_traj_emb = traj_emb.detach().clone()
                mean_noise_emb = noise_emb.clone()
                var_noise_emb = torch.zeros_like(noise_emb)
                mean_cos_sim = cos_sim.clone()
            else:
                count += 1
                old_mean_noise_emb = mean_noise_emb.clone()
                
                mean_lang_emb = mean_lang_emb + (lang_emb - mean_lang_emb) / count
                mean_traj_emb = mean_traj_emb + (traj_emb - mean_traj_emb) / count
                mean_noise_emb = mean_noise_emb + (noise_emb - mean_noise_emb) / count
                mean_cos_sim = torch.cat((mean_cos_sim, cos_sim), dim=-1)
                
                var_noise_emb = var_noise_emb + (noise_emb - old_mean_noise_emb) * (noise_emb - mean_noise_emb)
            
            pbar.set_description(f"Processing step {step}")
            
            # save
            if step % 1000 == 0 and step > 0:
                ll = torch.mean(mean_lang_emb, dim=0)
                tt = torch.mean(mean_traj_emb, dim=0)
                ll = ll.cpu().numpy()
                tt = tt.cpu().numpy()
                cc = np.array(count)
                
                mn = torch.mean(mean_noise_emb, dim=0)
                vn = torch.mean(var_noise_emb, dim=0) / count
                mn = mn.cpu().numpy()
                vn = vn.cpu().numpy()
                
                # cs = torch.mean(mean_cos_sim, dim=0)
                cs = mean_cos_sim.cpu().numpy()
                
                np.savez(f'libero130/mean_embeddings_steps{step}.npz', \
                        mean_lang_emb=ll, mean_traj_emb=tt, count=cc, \
                        mean_noise_emb=mn, var_noise_emb=vn, cos_sim=cs)
                print(f"Mean embeddings saved to 'mean_embeddings.npz' at step {step}")

            # hdf5 part
            if args.umap:
                # 创建 HDF5 文件并创建主组
                with h5py.File(f"{directory_path}/umap.h5", "w") as f:
                        main_group = f.require_group("representations")
                
                        # 全局计数器
                        global_counter = main_group.attrs.get('counter', 0)

                        for i in range(args.batch_size):
                            # 获取第一个视角和第二个视角的表征
                            view1 = traj_emb[i].numpy()
                            lang_repr = lang_emb[i].numpy()
                            
                            # 使用全局计数器创建唯一的均值子组
                            mean_group_name = f"means_{global_counter}"
                            mean_group = main_group.require_group(mean_group_name)
                            mean_group.create_dataset("view", data=view1)
                            mean_group.create_dataset("lang", data=lang_repr)
                            mean_group.create_dataset('label', data=lang[i])
                            
                            # 增加计数器
                            global_counter += 1

                        # 保存全局计数器
                        main_group.attrs['counter'] = global_counter
                        
                for i in range(args.batch_size):
                    lang_embs = np.vstack((lang_embs, lang_emb[i].numpy()))
                    traj_embs = np.vstack((traj_embs, traj_emb[i].numpy()))
                    label = lang[i]
                    if label not in label_dict:
                        label_dict[label] = label_count
                        label_count += 1
                    labels.append(label_dict[label])

    # umap save
    if args.umap:
        labels = np.array(labels)
        np.savez(f'{directory_path}/umap.npz', lang_embs=lang_embs, traj_embs=traj_embs, labels=labels)
        print("UMAP embeddings saved to 'umap.npz'")
        
        # UMAP降维
        umap_model = umap.UMAP(n_components=2, random_state=42)
        lang_umap = umap_model.fit_transform(lang_embs)
        img_umap = umap_model.fit_transform(traj_embs)

        # 定义颜色和形状
        colors = plt.cm.rainbow(np.linspace(0, 1, num_instructions))
        markers = ['o', '^']

        # 绘制UMAP分布图
        plt.figure(figsize=(16, 10))
        for i in range(num_instructions):
            idx = labels == i
            plt.scatter(lang_umap[idx, 0], lang_umap[idx, 1], color=colors[i], marker=markers[0], s=100, label=f'Lang {i}')  # 增大数据点
            plt.scatter(img_umap[idx, 0], img_umap[idx, 1], color=colors[i], marker=markers[1], s=100, label=f'Img {i}')  # 增大数据点

        # 创建渐变颜色的图例
        legend_elements = [Line2D([0], [0], color=plt.cm.rainbow(i/num_instructions), lw=4, label=str(i)) for i in range(num_instructions)]
        plt.legend(handles=legend_elements, title='Instructions', bbox_to_anchor=(1.05, 1), loc='upper left')

        # 设置标题和标签
        plt.title('UMAP of Lang and Img Embeddings')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')

        # 显示图表
        plt.savefig(f'{directory_path}/umap.png')
    
    # resize
    mean_lang_emb = torch.mean(mean_lang_emb, dim=0)
    mean_traj_emb = torch.mean(mean_traj_emb, dim=0)
    count = np.array(count)
    mean_noise_emb = torch.mean(mean_noise_emb, dim=0)
    var_noise_emb = torch.mean(var_noise_emb, dim=0) / count
    # mean_cos_sim = torch.mean(mean_cos_sim, dim=0)


    # 将均值转换为 numpy 数组并保存为 .npz 文件
    mean_lang_emb_np = mean_lang_emb.cpu().numpy()
    mean_traj_emb_np = mean_traj_emb.cpu().numpy()
    mean_noise_emb_np = mean_noise_emb.cpu().numpy()
    var_noise_emb_np = var_noise_emb.cpu().numpy()
    cos_sim_np = mean_cos_sim.cpu().numpy()

    np.savez(f'{directory_path}/mean_embeddings_final.npz', \
            mean_lang_emb=mean_lang_emb_np, mean_traj_emb=mean_traj_emb_np, count=count,\
            mean_noise_emb=mean_noise_emb_np, var_noise_emb=var_noise_emb_np, cos_sim=cos_sim_np)

    print("Mean embeddings saved to 'mean_embeddings_final.npz'")


if __name__ == '__main__':
    args = get_args()
    main(args)

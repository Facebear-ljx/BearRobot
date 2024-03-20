import math
import os
import json
import random

import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms

import d4rl
import gym

EPS = 1e-5

OPENXDATASETS = [
    'kaist_nonprehensile_converted_externally_to_rlds',#1
    'fractal20220817_data', #1
    'bridge',#1
    'kuka',#1
    'taco_play',#1
    'jaco_play',#1
    'berkeley_cable_routing',#1
    'roboturk',#1
    'nyu_door_opening_surprising_effectiveness',#1
    'viola',#1
    'berkeley_autolab_ur5',#1
    'toto',#1
    'language_table', # 1
    'columbia_cairlab_pusht_real', # 1
    'stanford_kuka_multimodal_dataset_converted_externally_to_rlds', # 1
    'nyu_rot_dataset_converted_externally_to_rlds', #1
    'stanford_hydra_dataset_converted_externally_to_rlds', #1
    'austin_buds_dataset_converted_externally_to_rlds',#1
    'nyu_franka_play_dataset_converted_externally_to_rlds',#1
    'maniskill_dataset_converted_externally_to_rlds',#1
    'cmu_franka_exploration_dataset_converted_externally_to_rlds',#1
    'ucsd_kitchen_dataset_converted_externally_to_rlds',#1
    'ucsd_pick_and_place_dataset_converted_externally_to_rlds',#1
    'austin_sailor_dataset_converted_externally_to_rlds',#1
    'austin_sirius_dataset_converted_externally_to_rlds',#1
    'bc_z', #1
#     'usc_cloth_sim_converted_externally_to_rlds', #1 low resolution
    'utokyo_pr2_opening_fridge_converted_externally_to_rlds', #1
    'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds', #1
#     'utokyo_saytap_converted_externally_to_rlds', #1 all black images
    'utokyo_xarm_pick_and_place_converted_externally_to_rlds', #1
    'utokyo_xarm_bimanual_converted_externally_to_rlds', #1
    'robo_net',#1
#     'berkeley_mvp_converted_externally_to_rlds',#1 only wrist
#     'berkeley_rpt_converted_externally_to_rlds',#1 only wrist
    # 'kaist_nonprehensile_converted_externally_to_rlds',#1
    'stanford_mask_vit_converted_externally_to_rlds',#1
    'tokyo_u_lsmo_converted_externally_to_rlds',#1
    'dlr_sara_pour_converted_externally_to_rlds',#1
    'dlr_sara_grid_clamp_converted_externally_to_rlds',#1
    'dlr_edan_shared_control_converted_externally_to_rlds',#1
    'asu_table_top_converted_externally_to_rlds',#1
#     'stanford_robocook_converted_externally_to_rlds',#1  # wrong images
#     'eth_agent_affordances',#1   # black images
#     'imperialcollege_sawyer_wrist_cam',#1  low resolution
    'iamlab_cmu_pickup_insert_converted_externally_to_rlds',#1
    'uiuc_d3field',#1
    'utaustin_mutex',#1
    'berkeley_fanuc_manipulation',#1
    'cmu_play_fusion',#1
    'cmu_stretch',#1
    # 'berkeley_gnm_recon',#1 navigation tasks
    # 'berkeley_gnm_cory_hall', #1 navigation tasks
    # 'berkeley_gnm_sac_son', #1 navigation tasks
]


def return_range(dataset, max_episode_steps):
       returns, lengths = [], []
       ep_ret, ep_len = 0., 0
       for r, d in zip(dataset['rewards'], dataset['terminals']):
              ep_ret += float(r)
              ep_len += 1
              if d or ep_len == max_episode_steps:
                     returns.append(ep_ret)
                     lengths.append(ep_len)
                     ep_ret, ep_len = 0., 0
       # returns.append(ep_ret)    # incomplete trajectory
       lengths.append(ep_len)      # but still keep track of number of steps
       assert sum(lengths) == len(dataset['rewards'])
       return min(returns), max(returns)


class D4RLDataset(Dataset):
       def __init__(
              self,
              env_name: str,
              norm_dict: dict={"norm_s": False,
                               "norm_a": False,
                               "norm_r": True},
              clip_to_eps: bool=True, 
       ):
              super().__init__()
              
              self.env_name = env_name
              env = gym.make(env_name)
              self.dataset = d4rl.qlearning_dataset(env)  # dict with np.array
              self.data_statistics = {"dataset_num" : self.dataset['observations'].shape[0]}
              
              if clip_to_eps:
                     lim = 1 - EPS
                     self.dataset["actions"] = np.clip(self.dataset["actions"], -lim, lim)
              
              dones = np.full_like(self.dataset["rewards"], False, dtype=bool)
              for i in range(len(dones) - 1):
                     if (
                            np.linalg.norm(
                            self.dataset["observations"][i + 1]
                            - self.dataset["next_observations"][i]
                            )
                            > 1e-6
                            or self.dataset["terminals"][i] == 1.0
                     ):
                            dones[i] = True

              dones[-1] = True
              self.dataset["dones"] = dones
              
              self._normalize_dataset(**norm_dict)
              self._tran2tensor()
              
              self.s_dim = self.data_statistics['s_mean'].shape[1]
              self.a_dim = self.data_statistics['a_mean'].shape[1]
       
             
       def _normalize_dataset(self, norm_s=False, norm_a=False, norm_r=False):
              if norm_s:
                     s_mean = self.dataset['observations'].mean(axis=0, keepdims=True)
                     s_std = self.dataset['observations'].std(axis=0, keepdims=True)
              else:
                     s_mean = np.zeros_like(self.dataset['observations'].mean(axis=0, keepdims=True))
                     s_std = np.ones_like(self.dataset['observations'].std(axis=0, keepdims=True))
                     
              if norm_a:
                     a_mean = self.dataset['actions'].mean(axis=0, keepdims=True)
                     a_std = self.dataset['actions'].std(axis=0, keepdims=True)
              else:
                     a_mean = np.zeros_like(self.dataset['actions'].mean(axis=0, keepdims=True))
                     a_std = np.ones_like(self.dataset['actions'].std(axis=0, keepdims=True))
              
              if any(s in self.env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
                     if norm_r:
                            (_, _, episode_returns) = self._trajectory_boundaries_and_returns()
                            self.dataset["rewards"] /= (np.max(episode_returns) - np.min(episode_returns))
                            self.dataset['rewards'] *= 1000
              elif 'antmaze' in self.env_name:
                     self.dataset['rewards'] -= 1.
              else:
                     raise NotImplementedError
                     
              self.dataset['observations'] = (self.dataset['observations'] - s_mean) / (s_std + EPS)
              self.dataset['next_observations'] = (self.dataset['next_observations'] - s_mean) / (s_std + EPS)
              self.dataset['actions'] = (self.dataset['actions'] - a_mean) / (a_std + EPS)
              
              self.data_statistics = {
                     "dataset_num" : self.dataset['observations'].shape[0],
                     "s_mean" : s_mean,
                     "s_std" : s_std,
                     "a_mean" : a_mean,
                     "a_std" : a_std,            
              }

       def _trajectory_boundaries_and_returns(self):
              episode_starts = [0]
              episode_ends = []

              episode_return = 0
              episode_returns = []

              for i in range(len(self)):
                     episode_return += self.dataset["rewards"][i]

                     if self.dataset["dones"][i]:
                            episode_returns.append(episode_return)
                            episode_ends.append(i + 1)
                            if i + 1 < len(self):
                                   episode_starts.append(i + 1)
                                   episode_return = 0.0

              return episode_starts, episode_ends, episode_returns
                     
       def _tran2tensor(self):
              for key in ['observations', 'actions', 'rewards', 'next_observations', 'terminals']:
                     self.dataset[key] = torch.from_numpy(self.dataset[key]).to(torch.float32).view(self.data_statistics["dataset_num"], -1)
       
       def __len__(self):
              return self.data_statistics["dataset_num"]

       def __getitem__(self, idx):
              s = self.dataset['observations'][idx]
              a = self.dataset['actions'][idx]
              r = self.dataset['rewards'][idx]
              next_s = self.dataset['next_observations'][idx]
              d = 1. - self.dataset['terminals'][idx]
              return {"s": s, "a": a, "r": r, "next_s": next_s, "d": d}
       

class AIROpenXDataset(Dataset):
       def __init__(
              self,
              base_dir='/data/openxdata_npy',
              datalist='/data/openxdata_npy/datalist.json',
              dataset_name='bridge',
       ):
              self.base_dir = base_dir
              self.dataset_name = dataset_name
              
              self.datalist = json.load(open(datalist, "r"))
              self.dataset_weights = self._prepare_sampling_weights()
              
              self.transform = transforms.ToTensor()
       
       def _prepare_indexes(self):
              """
              prepare indexes with (N1, N2, N3)
              N1 is the dataset num
              N2 is the episodes num for each dataset
              N3 is the steps num for each episode
              
              Only callable when the dataset is small
              """
              idxes = []
              for dataset_idx, dataset in enumerate(self.datalist):
                     for episode_idx, episode in enumerate(dataset['data']):
                            for step_idx in range(episode['image_length']):
                                   idxes.append((dataset_idx, episode_idx, step_idx))
              return idxes
       
       def _prepare_sampling_weights(self):
              """
              random sample datasets according to the dataset num N1
              """
              dataset_nums = []
              for _, dataset in enumerate(self.datalist):
                     dataset_nums.append(dataset['all_num'])
              sum_dataset_nums = sum(dataset_nums)
              dataset_weights = [num / sum_dataset_nums for num in dataset_nums]
              return dataset_weights
       
       def _openxdataversion(self, dataset_name):
              """
              transform openxdata dataset_name to its version id
              """
              if dataset_name == 'robo_net':
                     version = '1.0.0'
              elif dataset_name == 'language_table':
                     version = '0.0.1'
              else:
                     version = '0.1.0'
              return version
       
       def __len__(self):
              try:
                     return len(self.indexes)
              except:
                     return 10000

       def __getitem__(self, idx):
              dataset_idx = random.choices(range(len(self.dataset_weights)), weights=self.dataset_weights, k=1)[0]
              dataset_idx = 1 # TODO, comment this line when full data is available
              episode_idx = random.randint(0, self.datalist[dataset_idx]['all_num'] - 1)
              step_idx = random.randint(0, self.datalist[dataset_idx]['data'][episode_idx]['image_length'] - 1)
              
              dataset_name = self.datalist[dataset_idx]['dataset_name']
              version = self._openxdataversion(dataset_name)
              
              episode_id = self.datalist[dataset_idx]['data'][episode_idx]['id']
              image_type = 'image0'  # assume all dataset contains image0
              
              # e.g. /data/openxdata_npy/bridge/0.1.0/train-0/image0/0.jpg
              image_path = f'{self.base_dir}/{dataset_name}/{version}/{episode_id}/{image_type}/{step_idx}.jpg'
              image = Image.open(image_path)
              
              if self.transform:
                     image = self.transform(image)
              return image
       


class RT1Dataset(AIROpenXDataset):
       def __init__(
              self,
              base_dir: str='/data/openxdata_npy',
              datalist: str='/data/openxdata_npy/datalist.json',
              dataset_name: str='bridge',
              img_size: int=128,
              frames: int=1,
              view_list: list[str]=['image0'],
       ):
              super().__init__(base_dir,
                               datalist,
                               dataset_name)
              self.img_size = img_size
              self.frames = frames
              self.view_list = view_list
              
              transform = [
                     transforms.Resize(256, interpolation=Image.BICUBIC),
                     transforms.CenterCrop(img_size),
                     transforms.ToTensor()
              ]
              self.transform = transforms.Compose(transform)
              
              self.action_keys = ["world_vector", "rotation_delta", "open_gripper"]
       
       
       def discretize(self, tensor, num_bins, min_val, max_val):
              normalized_tensor = (tensor - min_val) / (max_val - min_val)
              discretized_tensor = torch.floor(normalized_tensor * num_bins).clamp(0, num_bins - 1)
              discretized_tensor = discretized_tensor
              return discretized_tensor
       
       def __getitem__(self, idx):
              dataset_idx = random.choices(range(len(self.dataset_weights)), weights=self.dataset_weights, k=1)[0]
              dataset_idx = 1 # TODO, comment this line when full data is available
              episode_idx = random.randint(0, self.datalist[dataset_idx]['all_num'] - 1)
              step_idx = random.randint(0, self.datalist[dataset_idx]['data'][episode_idx]['image_length'] - 1 - self.frames)
              
              dataset_name = self.datalist[dataset_idx]['dataset_name']
              version = self._openxdataversion(dataset_name)
              
              episode_id = self.datalist[dataset_idx]['data'][episode_idx]['id']
              
              # images
              frame_list = []
              for _ in range(self.frames):
                     view_list = []
                     for view in self.view_list:
                     # e.g. /data/openxdata_npy/bridge/0.1.0/train-0/image0/0.jpg
                            image_path = f'{self.base_dir}/{dataset_name}/{version}/{episode_id}/{view}/{step_idx}.jpg'
                            image = Image.open(image_path)
                            if self.transform:
                                   image = self.transform(image)
                            view_list.append(image) # TODO warning, this operation may bring different sequences for different views
                     frame_list.append(view_list)
                     step_idx += 1
              images = torch.stack([torch.stack([view.reshape(3, self.img_size, self.img_size) for view in view_list]) for view_list in frame_list])
              
              # actions
              action_lang_path = f'{self.base_dir}/{dataset_name}/{version}/{episode_id}/action.json'
              with open(f'{action_lang_path}', 'r') as f:
                     step_info = json.load(f)
              
              ## discretize the action label
              action = step_info[step_idx]['action']
              action = torch.cat([torch.tensor(action[key], dtype=torch.float32).view(-1) for key in self.action_keys])
              action[:6] = self.discretize(action[:6], 256, -1., 1.)
              action[-1] = torch.tensor(255) if action[-1] == 1. else torch.tensor(0)  # TODO check here
              
              # language instruction
              lang = step_info[0]['lang']
              
              return {"imgs": images,
                      "a": action,
                      "lang": lang}
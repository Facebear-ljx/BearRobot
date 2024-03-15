import math
import os
import json
import random

import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

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


class D4RLDataset(Dataset):
       def __init__(
              self,
              env_name,
              norm_dict = {"norm_s": True,
                           "norm_a": False,
                           "norm_r": True},
       ):
              super().__init__()
              
              env = gym.make(env_name)
              self.dataset = d4rl.qlearning_dataset(env, terminate_on_end=True)  # dict with np.array
              self._get_statistics()
              self._normalize_dataset(**norm_dict)
              self._tran2tensor()
       
       def _get_statistics(self):
              self.dataset_num = self.dataset['observations'].shape[0]
              self.s_mean = self.dataset['observations'].mean(axis=0, keepdims=True)
              self.s_std = self.dataset['observations'].std(axis=0, keepdims=True)
              self.a_mean = self.dataset['actions'].mean(axis=0, keepdims=True)
              self.a_std = self.dataset['actions'].mean(axis=0, keepdims=True)
              self.r_mean = self.dataset['rewards'].mean()
              self.r_std = self.dataset['rewards'].mean()
       
       def _normalize_dataset(self, norm_s=False, norm_a=False, norm_r=False):
              if norm_s:
                     self.dataset['observations'] = (self.dataset['observations'] - self.s_mean) / (self.s_std + EPS)
                     self.dataset['next_observations'] = (self.dataset['next_observations'] - self.s_mean) / (self.s_std + EPS)
              if norm_a:
                     self.dataset['actions'] = (self.dataset['actions'] - self.a_mean) / (self.a_std + EPS)
              if norm_r:
                     self.dataset['rewards'] = (self.dataset['rewards'] - self.r_mean) / (self.r_std + EPS)
                     
       def _tran2tensor(self):
              for key in ['observations', 'actions', 'rewards', 'next_observations', 'terminals']:
                     self.dataset[key] = torch.from_numpy(self.dataset[key]).to(torch.float32).view(self.dataset_num, -1)
       
       def __len__(self):
              return self.dataset_num

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
              transform=T.ToTensor(),
       ):
              self.base_dir = base_dir
              self.dataset_name = dataset_name
              
              self.datalist = json.load(open(datalist, "r"))
              self.dataset_weights = self._prepare_sampling_weights()
              
              self.transform = transform
       
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
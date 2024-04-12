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

from utils import ddp
from torch.utils.data import DistributedSampler

from mmengine import fileio
import io

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

def openimage(path):
       value  = fileio.get(path)
       img_bytes = np.frombuffer(value, np.uint8)
       buff = io.BytesIO(img_bytes)
       img = Image.open(buff).convert('RGB')
       return img


def openjson(path):
       value  = fileio.get_text(path)
       dict = json.loads(value)
       return dict


class AIROpenXDataset(Dataset):
       def __init__(
              self,
              base_dir='/data/openxdata_npy',
              datalist='/data/openxdata_npy/datalist.json',
              dataset_name='bridge',
       ):
              self.base_dir = base_dir
              self.dataset_name = dataset_name
              
              self.datalist = openjson(datalist)
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
              image = openimage(image_path)
              
              if self.transform:
                     image = self.transform(image)
              return image
       

class RT1Dataset_new():
       def __init__(
              self,
              datalist: str='/home/lijx/ljx/robotics/bearobot/data/bridge/bridge_datalist.json',
              img_size: int=128,
       ):       
              self.img_size = img_size
              self.datalist = openjson(datalist)
              
              transform = [
                     # transforms.Resize(256, interpolation=Image.BICUBIC),
                     transforms.RandomResizedCrop(img_size, scale=(0.75, 1), interpolation=Image.BICUBIC),
                     transforms.ColorJitter(0.2, [0.8, 1.2], [0.8, 1.2], 0.1),
                     transforms.ToTensor()
              ]
              
              self.transform = transforms.Compose(transform)        
              
              self.action_keys = ["world_vector", "rotation_delta", "open_gripper"]   
       
       def discretize(self, tensor, num_bins, min_val, max_val):
              """
              discretize the continuous actions from [min_val, max_val] to num_bins bins
              """
              normalized_tensor = (tensor - min_val) / (max_val - min_val)
              discretized_tensor = torch.floor(normalized_tensor * num_bins).clamp(0, num_bins - 1)
              discretized_tensor = discretized_tensor
              return discretized_tensor
       
       def __len__(self):
              return len(self.datalist)
       
       def __getitem__(self, idx):
              step = self.datalist[idx]
              imgs_path = step['imgs']
              action = step['label']
              lang = step['lang']
              
              # discretize the action
              action = torch.cat([torch.tensor(action[key], dtype=torch.float32).view(-1) for key in self.action_keys])
              action[:6] = self.discretize(action[:6], 256, -1., 1.)
              action[-1] = torch.tensor(255) if action[-1] == 1. else torch.tensor(0)
              
              # images
              # [image t+2, image t+1, image t] -> [image t, image t+1, image t+2]
              image = openimage(imgs_path[0])
              if self.transform:
                     images = self.transform(image).reshape(1, 1, 3, self.img_size, self.img_size)
              
              return {"imgs": images,
                      "label": action,
                      "lang": lang}
                                   

class RT1Dataset(AIROpenXDataset):
       def __init__(
              self,
              base_dir: str='/data/openxdata_npy',
              datalist: str='/data/openxdata_npy/datalist.json',
              dataset_name: str='bridge',
              img_size: int=128,
              frames: int=1,
              view_list: list=['image0'],
       ):
              """
              rt1 dataset
              base_dir: the dataset base dir 
              datalistt: a json file that summarize the dataset
              dataset_name: not been used
              img_size: resize the img to [img_size, img_size]
              frames: history frames per sample
              vide_list: default to image0, as almost all openx data contains image0
              """
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
       
       
       def __len__(self):
              return 100000 * 128
       
       def discretize(self, tensor, num_bins, min_val, max_val):
              """
              discretize the continuous actions from [min_val, max_val] to num_bins bins
              """
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
                            image = openimage(image_path)
                            if self.transform:
                                   image = self.transform(image)
                            view_list.append(image) # TODO warning, this operation may bring different sequences for different views
                     frame_list.append(view_list)
                     step_idx += 1
              images = torch.stack([torch.stack([view.reshape(3, self.img_size, self.img_size) for view in view_list]) for view_list in frame_list])
              
              # actions
              action_lang_path = f'{self.base_dir}/{dataset_name}/{version}/{episode_id}/action.json'
              step_info = openjson(action_lang_path)
              
              ## discretize the action label
              action = step_info[step_idx]['action']
              action = torch.cat([torch.tensor(action[key], dtype=torch.float32).view(-1) for key in self.action_keys])
              action[:6] = self.discretize(action[:6], 256, -1., 1.)
              action[-1] = torch.tensor(255) if action[-1] == 1. else torch.tensor(0)  # TODO check here
              
              # language instruction
              lang = step_info[0]['lang']
              
              return {"imgs": images,
                      "label": action,
                      "lang": lang}


class VPDataset(AIROpenXDataset):
       def __init__(
              self,
              base_dir: str='/data/openxdata_npy',
              datalist: str='/data/openxdata_npy/datalist.json',
              dataset_name: str='bridge',
              img_size: int=128,
              frames: int=1,
              skip_frame: int=5,
              view_list: list=['image0'],
       ):
              """
              video prediction dataset
              base_dir: the dataset base dir 
              datalistt: a json file that summarize the dataset
              dataset_name: not been used
              img_size: resize the img to [img_size, img_size]
              frames: history frames per sample
              skip_frame: given (s_{t-n}, ..., s_t), predict (s_{t+skip_frame})
              vide_list: default to image0, as almost all openx data contains image0
              """
              super().__init__(base_dir,
                               datalist,
                               dataset_name)
              self.img_size = img_size
              self.frames = frames
              self.skip_frame = skip_frame
              self.view_list = view_list
              
              transform = [
                     transforms.Resize(256, interpolation=Image.BICUBIC),
                     transforms.CenterCrop(img_size),
                     transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                     transforms.ToTensor()
              ]
              self.transform = transforms.Compose(transform)
       
       
       def __len__(self):
              return int(1e+6) * 128
       
       
       def __getitem__(self, idx):
              # select one dataset, one episode, one step
              dataset_idx = random.choices(range(len(self.dataset_weights)), weights=self.dataset_weights, k=1)[0]
              episode_idx = random.randint(0, self.datalist[dataset_idx]['all_num'] - 1)
              episode_length = self.datalist[dataset_idx]['data'][episode_idx]['image_length']
              step_idx = random.randint(0, max(0, episode_length - 1 - self.frames - self.skip_frame))
              
              # some dataset path components
              dataset_name = self.datalist[dataset_idx]['dataset_name']
              version = self._openxdataversion(dataset_name)
              episode_id = self.datalist[dataset_idx]['data'][episode_idx]['id']
              view = self.view_list[0] # TODO: random select a view
              
              # history images
              frame_list = []
              for _ in range(self.frames):
                     # e.g. /data/openxdata_npy/bridge/0.1.0/train-0/image0/0.jpg
                     image_path = f'{self.base_dir}/{dataset_name}/{version}/{episode_id}/{view}/{step_idx}.jpg'
                     image = openimage(image_path)
                     if self.transform:
                            image = self.transform(image)
                     
                     frame_list.append(image)
                     step_idx += 1
                     step_idx = min(step_idx, episode_length - 1)
              images = torch.stack([image.reshape(3, self.img_size, self.img_size) for image in frame_list])
              
              
              # future images
              future_idx = min(step_idx + self.skip_frame, episode_length - 1)
              image_path = f'{self.base_dir}/{dataset_name}/{version}/{episode_id}/{view}/{future_idx}.jpg'
              image = openimage(image_path)
              if self.transform:
                     image = self.transform(image)
              future_images = image.reshape(3, self.img_size, self.img_size)
              
              
              # action and language instruction
              action_lang_path = f'{self.base_dir}/{dataset_name}/{version}/{episode_id}/action.json'
              step_info = openjson(action_lang_path)
              lang = step_info[0]['lang']
              
              return {"imgs": images,
                      "label": future_images,
                      "lang": lang}


class AIRKitchenDataset():
       def __init__(
              self,
              datalist='/home/dodo/ljx/BearRobot/data/bridge/AIR-toykitchen.json',
              img_size: int=128,
              frames: int=3,
              view_list: list=['D435_image', 'wrist_image'],
       ):
              self.img_size = img_size
              self.frames = frames
              self.view_list = view_list
              
              self.datalist = openjson(datalist)
              
              transform = [
                     transforms.RandomResizedCrop(img_size, scale=(0.75, 1) ,interpolation=Image.BICUBIC),
                     transforms.ColorJitter(0.2, [0.8, 1.2], [0.8, 1.2], 0.1),
                     transforms.ToTensor()
              ]
              self.transform = transforms.Compose(transform)


       def discretize(self, tensor, num_bins, min_val, max_val):
              """
              discretize the continuous actions from [min_val, max_val] to num_bins bins
              """
              normalized_tensor = (tensor - min_val) / (max_val - min_val)
              discretized_tensor = torch.floor(normalized_tensor * num_bins).clamp(0, num_bins - 1)
              discretized_tensor = discretized_tensor
              return discretized_tensor

       def __len__(self):
              return len(self.datalist)

       def __getitem__(self, idx):
              step = self.datalist[idx]
              
              imgs_path = [step[f'{view}'] for view in self.view_list]
              action = step['action']
              lang = step['instruction']
              
              # discretize the action
              action = torch.tensor(action)
              action[:6] = self.discretize(action[:6], 256, -1., 1.)
              action[-1] = torch.tensor(255) if action[-1] == 1. else torch.tensor(0)
              
              # images
              images = [openimage(img_path) for img_path in imgs_path]
              images = torch.cat([self.transform(image).reshape(1, 1, 3, self.img_size, self.img_size) for image in images], dim=1)
              
              return {"imgs": images,
                      "label": action,
                      "lang": lang}
              
             
def VideoPredictDataLoader(
       base_dir: str='/data/openxdata_npy',
       datalist: str='/data/openxdata_npy/datalist.json',
       dataset_name: str='bridge',
       img_size: int=128,
       frames: int=1,
       skip_frame: int=5,
       view_list: list=['image0'],
       batch_size: int=64,
       num_workers: int=8,
       pin_mem: bool=True,
):
       """
       this is the video prediction dataloader
       """
       rt1dataset = VPDataset(base_dir=base_dir,
                            datalist=datalist,
                            dataset_name=dataset_name,
                            img_size=img_size,
                            frames=frames,
                            skip_frame=skip_frame,
                            view_list=view_list)

       num_tasks = ddp.get_world_size()
       global_rank = ddp.get_rank()
       sampler = DistributedSampler(
            rt1dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
       )
       
       rt1dataloader = DataLoader(
              rt1dataset, 
              # sampler=sampler,
              batch_size=batch_size // num_tasks, 
              num_workers=num_workers,
              pin_memory=pin_mem,
              drop_last=True
       )
       
       return rt1dataloader


def RT1DataLoader(
       base_dir: str='/data/openxdata_npy',
       datalist: str='/data/openxdata_npy/datalist.json',
       dataset_name: str='bridge',
       img_size: int=128,
       frames: int=1,
       view_list: list=['image0'],
       batch_size: int=64,
       num_workers: int=8,
       pin_mem: bool=True,
):
       rt1dataset = RT1Dataset_new(datalist=datalist,
                                   img_size=img_size)

       num_tasks = ddp.get_world_size()
       global_rank = ddp.get_rank()
       sampler = DistributedSampler(
            rt1dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
       )
       
       rt1dataloader = DataLoader(
              rt1dataset, 
              sampler=sampler,
              batch_size=batch_size // num_tasks, 
              num_workers=num_workers,
              pin_memory=pin_mem,
              drop_last=True
       )
       
       return rt1dataloader


def RT1ValDataLoader(
       datalist: str='/data/openxdata_npy/datalist.json',
       img_size: int=128,
       batch_size: int=16,
       num_workers: int=8,
       pin_mem: bool=True,
):
       rt1dataset = RT1Dataset_new(datalist=datalist,
                                   img_size=img_size)

       num_tasks = ddp.get_world_size()
       
       rt1dataloader = DataLoader(
              rt1dataset, 
              batch_size=batch_size // num_tasks, 
              num_workers=num_workers,
              pin_memory=pin_mem,
              drop_last=True
       )
       
       return rt1dataloader
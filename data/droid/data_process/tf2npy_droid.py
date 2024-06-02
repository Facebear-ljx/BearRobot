import os
import tqdm
import json

import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from PIL import Image

# make save dir

DATASETS = [
    'bridge',#1
    'kuka',#1
    'taco_play',#1
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
#     'usc_cloth_sim_converted_externally_to_rlds', #1 像素过低，弃用
    'utokyo_pr2_opening_fridge_converted_externally_to_rlds', #1
    'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds', #1
#     'utokyo_saytap_converted_externally_to_rlds', #1 全黑，弃用
    'utokyo_xarm_pick_and_place_converted_externally_to_rlds', #1
    'utokyo_xarm_bimanual_converted_externally_to_rlds', #1
    'robo_net',#1
#     'berkeley_mvp_converted_externally_to_rlds',#1 只有wrist，弃用
#     'berkeley_rpt_converted_externally_to_rlds',#1 只有wrist，弃用
    'kaist_nonprehensile_converted_externally_to_rlds',#1
    'stanford_mask_vit_converted_externally_to_rlds',#1
    'tokyo_u_lsmo_converted_externally_to_rlds',#1
    'dlr_sara_pour_converted_externally_to_rlds',#1
    'dlr_sara_grid_clamp_converted_externally_to_rlds',#1
    'dlr_edan_shared_control_converted_externally_to_rlds',#1
    'asu_table_top_converted_externally_to_rlds',#1
#     'stanford_robocook_converted_externally_to_rlds',#1  #图片存在缺陷，弃用
#     'eth_agent_affordances',#1   # 全黑图片，弃用
#     'imperialcollege_sawyer_wrist_cam',#1  像素过低，弃用
    'iamlab_cmu_pickup_insert_converted_externally_to_rlds',#1
    'uiuc_d3field',#1
    'utaustin_mutex',#1
    'berkeley_fanuc_manipulation',#1
    'cmu_play_fusion',#1
    'cmu_stretch',#1
    'berkeley_gnm_recon',#1
    'berkeley_gnm_cory_hall', #1 
    'berkeley_gnm_sac_son', #1
]


def dataset2readpath(dataset_name):
  if dataset_name == 'robo_net' or dataset_name == 'droid':
    version = '1.0.0'
  elif dataset_name == 'language_table':
    version = '0.0.1'
  else:
    version = '0.1.0'
  return f'/data/{dataset_name}/{version}'


def dataset2savepath(dataset_name):
  if dataset_name == 'robo_net' or dataset_name == 'droid':
    version = '1.0.0'
  elif dataset_name == 'language_table':
    version = '0.0.1'
  else:
    version = '0.1.0'
  return f'/data/openxdata_npy/{dataset_name}/{version}'


def convert_to_list(o):
    try:
        return o.tolist()  # 将numpy数组转换为列表
    except:
        # 遍历字典，递归转换每个值
        return {k: convert_to_list(v) for k, v in o.items()}
 

if __name__=="__main__":
       # process the dataset one by one
       split = 'train'
       for dataset_name in ['droid']:
              try:
                     print('-'*80)
                     print('Processing', dataset_name)
                     print('-'*80)
                     
                     save_dir = dataset2savepath(dataset_name)
                     read_dir = dataset2readpath(dataset_name)

                     if not os.path.exists(save_dir):
                            os.makedirs(save_dir, exist_ok=True)

                     # dataloader from save path
                     b = tfds.builder_from_directory(builder_dir=dataset2readpath(dataset_name))
                     ds = b.as_dataset(split=split)  # TODO, replace with [train, test, val]

                     # see whether the dataset has wrist image
                     with open(f'{dataset2readpath(dataset_name)}/features.json', 'r') as f:
                            feature_content = f.read()

                     # start process
                     for i, d in enumerate(tqdm.tqdm(ds)):
                            # define some paths to save and some temp list
                            dataset_only_have_wrist_image = ['berkeley_rpt_converted_externally_to_rlds', 'berkeley_mvp_converted_externally_to_rlds']
                            
                            save_list = []
                            base_path = f'{save_dir}/{split}-{i}'
                            image0_path = f'{base_path}/image0'
                            image1_path = f'{base_path}/image1'
                            wrist_path = f'{base_path}/wrist'
                            
                            os.makedirs(image0_path, exist_ok=True)
                            os.makedirs(image1_path, exist_ok=True)
                            os.makedirs(wrist_path, exist_ok=True)
                            action_path = f'{base_path}/action.json'
                            lang_path = f'{base_path}/lang.txt'
                            
                            # save image & action & language
                            for j, step in enumerate(d['steps'].as_numpy_iterator()):
                                   save_dict = {}
                                   
                                   # save image
                                   image0 = step['observation']['exterior_image_1_left']
                                   image1 = step['observation']['exterior_image_2_left']
                                                 
                                   image0 = image0.numpy() if isinstance(image0, tf.Tensor) else image0
                                   image0 = Image.fromarray(image0.astype(np.uint8))
                                   image_save_path = f'{image0_path}/{j}.jpg'
                                   image0.save(image_save_path)
                                   save_dict.update({"image0_path": image_save_path})

                                   image1 = image1.numpy() if isinstance(image1, tf.Tensor) else image1
                                   image1 = Image.fromarray(image1.astype(np.uint8))
                                   image_save_path = f'{image1_path}/{j}.jpg'
                                   image1.save(image_save_path)
                                   save_dict.update({"image1_path": image_save_path})
                                   
                                   # save wrist image                             
                                   wrist_image = step['observation']['wrist_image_left']
                                   wrist_image = wrist_image.numpy() if isinstance(wrist_image, tf.Tensor) else wrist_image
                                   wrist_image = Image.fromarray(wrist_image.astype(np.uint8))
                                   
                                   wrist_image_save_path = f'{wrist_path}/{j}.jpg'
                                   wrist_image.save(wrist_image_save_path)
                                   save_dict.update({"wrist_image_path": wrist_image_save_path})
                                   
                                          
                                   # save action
                                   save_dict.update({"action": convert_to_list(step['action'])})
                                   
                                   # save lang
                                   try:
                                          save_dict.update({"lang": step['language_instruction'].decode('utf-8')})
                                   except:
                                          try:
                                                 save_dict.update({"lang": step['observation']['natural_language_instruction'].decode('utf-8')})
                                          except:
                                                 if dataset_name == 'language_table':
                                                        utf8_code = step['observation']['instruction'].tolist()
                                                        original_string = ''.join([chr(code) for code in utf8_code])
                                                        cleaned_string = original_string.replace('\u0000', '')
                                                        save_dict.update({"lang": cleaned_string})
                                                        
                                   
                                   save_list.append(save_dict)
                            
                            # dump the save_list to json
                            with open(action_path, 'w') as f:
                                   json.dump(save_list, f, indent=4)
                                   
                     print(f'Processing {dataset_name} DONE \n' + f'episode_num={len(ds)}')
              except:
                     print(f'{dataset_name} wrong')
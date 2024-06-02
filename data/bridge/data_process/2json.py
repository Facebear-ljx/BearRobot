import os
import tqdm
import json

import numpy as np
from PIL import Image

split = 'train'

def convert_to_list(o):
    try:
        return o.tolist()  # 将numpy数组转换为列表
    except:
        # 遍历字典，递归转换每个值
        return {k: convert_to_list(v) for k, v in o.items()}


def openjson(path):
       from mmengine import fileio
       import io
       value  = fileio.get_text(path)
       dict = json.loads(value)
       return dict


if __name__=="__main__":
       asset = 'train'
       base_dir = '/data/openxdata_npy/bridge/0.1.0'
       datalist_path = '/data/openxdata_npy/datalist.json'
       frame_num = 3
       
       # process the dataset one by one
       datalist = openjson(datalist_path)
       
       bridge_datalist = datalist[1]
       
       count = 0
       
       list_all = []
       
       for episode in bridge_datalist['data']:
              episode_id = episode['id']
              if asset not in episode_id:
                     continue
              for step in range(episode['image_length']):
                     imgs = []
                     for i in range(frame_num):
                            step_idx = max(0, step - i)
                            img = f'{base_dir}/{episode_id}/image0/{step_idx}.jpg'
                            imgs.append(img)
                            
                     action_lang = openjson(f'{base_dir}/{episode_id}/action.json')
                     
                     lang = action_lang[0]['lang']
                     action = action_lang[step]['action']

                     one_sample = {"imgs": imgs,
                                   "label": action,
                                   "lang": lang}

                     list_all.append(one_sample)
                     
                     count += 1
                     print(count)
                     
       save_path = f'bridge_datalist_{asset}.json'
       with open(save_path, 'w') as f:
              json.dump(list_all, f, indent=4)
                     
                     
       

       
       
from tqdm import tqdm
from mmengine import fileio
import io
import json
import os
import numpy as np

import fam
from PIL import Image

def openjson(path):
       value  = fileio.get_text(path)
       dict = json.loads(value)
       return dict

def openimage(path):
       from mmengine import fileio
       import io
       value  = fileio.get(path)
       img_bytes = np.frombuffer(value, np.uint8)
       buff = io.BytesIO(img_bytes)
       img = Image.open(buff).convert('RGB')
       return img

import torch
ckpt_path = "/home/dodo/ljx/SeeWhatYouNeedDeploy/assets/ckpt/model_3.bin"
ckpt = torch.load(ckpt_path, map_location="cpu")
print(ckpt.keys())

from lisa_deploy import load
model = load(ckpt_path, type="lisa", low_gpu_memory = False)

datalist = openjson("/home/dodo/ljx/BearRobot/data/airkitchen/AIR-toykitchen-ac.json")


keys = ['blur_image', 'highlight_image', 'alpha_image']
threshold = 0.5
new_data_list = []
for data in tqdm(datalist):
       # open img & read instruction
       d435_path = data['D435_image'].replace("/home/dodo/bridge_data_v2/bridge_data_robot/widowx_envs/widowx_data/robonetv2/bridge_data_v2/", "/data/AIRKITCHEN/human/")
       d435 = openimage(d435_path)
       instruction = data['instruction']
       
       # get blur image
       d435_output_imgs = model(d435, instruction, threshold = threshold, blur_kernel_size = 201, dilate_kernel_size =7)
       data['D435_image'] = d435_path
       data['wrist_image'] = d435_path.replace("images0", "images1")
       for key in keys:
              img = Image.fromarray(d435_output_imgs[key].astype(np.uint8))
              save_path = d435_path.replace("images0", key)
              img_name = save_path.split("/")[-1]
              save_dir_path = save_path.replace(img_name, "")
              
              if not os.path.exists(save_dir_path):
                     os.makedirs(save_dir_path)
              
              if key == "alpha_image":
                     save_path = save_path.replace("jpg", "png")
                     img.save(save_path, "PNG")
              else:
                     img.save(save_path, "JPEG")   
              
              data.update({key: save_path})
              print(save_path)
       
       # masks
       masks = d435_output_imgs['soft']
       save_path = d435_path.replace("images0", "mask").replace(".jpg", ".npy")
       mask_name = save_path.split("/")[-1]
       save_dir_path = save_path.replace(mask_name, "")
       if not os.path.exists(save_dir_path):
              os.makedirs(save_dir_path)   
       np.save(save_path, masks)
       print(save_path)
       data.update({"mask": save_path})
       
       new_data_list.append(data)

with open("/home/dodo/ljx/BearRobot/data/airkitchen/AIR-toykitchen-ac-blur-lisa.json", "w") as f:
       json.dump(new_data_list, f, indent=4)
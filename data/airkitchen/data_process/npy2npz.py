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

datalist = openjson("/home/dodo/ljx/BearRobot/data/airkitchen/AIR-toykitchen-ac-blur-lisa.json")


keys = ['mask']
threshold = 0.5
new_data_list = []
for data in tqdm(datalist):
    npy_path = data['mask']
    # save_path = npy_path.replace(".npy", ".npz")
    # print(save_path)
    # try:
    #     npy_file = np.load(npy_path)
    #     np.savez_compressed(save_path, npy_file)
    # except:
    #     assert os.path.exists(save_path)
            
    # data.update({"mask": save_path})
    if os.path.exists(npy_path):
        os.remove(npy_path)

    # new_data_list.append(data)
    

# with open("/home/dodo/ljx/BearRobot/data/airkitchen/AIR-toykitchen-ac-blur-lisa-npyz.json", "w") as f:
#        json.dump(new_data_list, f, indent=4)
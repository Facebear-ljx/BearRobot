import re
import os
import torch
from mmengine import fileio
import json
from data.libero.data_process import demo2frames

def get_demofixed_begin_end_frame(step,base_dir,frame_length_dic,img_base_dir = "libero/data_jpg/libero_goal/"):
    pattern = re.compile(r'([^/]+_demo)/demo_(\d+)/')

    match = pattern.search(step['D435_image'])
    
    if match:
            instruction = match.group(1)
            demo_id = match.group(2)
            key = f"{instruction}/demo_{demo_id}"
            frame_length = frame_length_dic.get(key, 100) - 1 # Default to 1000 if not found

    pattern_step_idx = re.compile(r'/(\d+)\.jpg$')
    match = pattern_step_idx.search(step['D435_image'])

    img_begin_path = os.path.join(base_dir,img_base_dir,key,f"image0/{0}.jpg")
    img_end_path = os.path.join(base_dir,img_base_dir,key,f"image0/{frame_length}.jpg")

    return img_begin_path,img_end_path

def get_demofixed_random_frame(step,base_dir,frame_length_dic,img_base_dir = "libero/data_jpg/libero_goal/"):
    pattern = re.compile(r'([^/]+_demo)/demo_(\d+)/')

    match = pattern.search(step['D435_image'])
    
    if match:
            instruction = match.group(1)
            demo_id = match.group(2)
            key = f"{instruction}/demo_{demo_id}"
            frame_length = frame_length_dic.get(key, 100) - 1 # Default to 1000 if not found

    pattern_step_idx = re.compile(r'/(\d+)\.jpg$')
    match = pattern_step_idx.search(step['D435_image'])
    
    step_idx = int(match.group(1))
    if step_idx == 0:
        idx0 = step_idx
    else:
        idx0 = torch.randint(0, step_idx, (1,)).item()
    idx1 = torch.randint(step_idx, frame_length, (1,)).item()

    img_begin_path = os.path.join(base_dir,img_base_dir,key,f"image0/{idx0}.jpg")
    img_end_path = os.path.join(base_dir,img_base_dir,key,f"image0/{idx1}.jpg")

    return img_begin_path,img_end_path

def test_func():
    global base_dir,datalist
    base_dir='/home/dodo/ljx/BearRobot/data/libero/dataset/'
    datalist=['/home/dodo/ljx/BearRobot/data/libero/libero_goal-ac.json']
    datalist_img = []  

    def openjson(path):
           value  = fileio.get_text(path)
           dict = json.loads(value)
           return dict
    
    for one_list in datalist: 
        datalist = openjson(one_list)
        datalist_img += datalist
        
    random_index = torch.randint(0, 10000, (1,)).item()
    step = datalist_img[random_index]

    return step

if __name__ == "__main__":
    get_demofixed_begin_end_frame(test_func(),base_dir,demo2frames.frame_counts_dict())
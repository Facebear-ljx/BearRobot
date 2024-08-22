import json
import os
import numpy as np
from tqdm import tqdm
import random
import re

# random lang mask
lang_prop = 0.75
index_list = list(range(130))
select_num = round(len(index_list) * lang_prop)
random.seed(0)
lang_mask = random.sample(index_list, select_num)
lang_mask = set(lang_mask)

print("len(lang_mask):", len(lang_mask))
print("lang_mask:", lang_mask)

prefix = "/home/dodo/ljx/BearRobot/data/libero/dataset/"
base_dir = "libero/data_jpg/"
base_dir = os.path.join(prefix, base_dir)
dataset_lists = os.listdir(base_dir)

data = []
frame_count = 0
task_count = 0
demo_count = 0
for dataset in tqdm(dataset_lists):
    
    dataset_dir = os.path.join(base_dir, dataset)
    if not os.path.isdir(dataset_dir):
        continue
    print(dataset)
    # if dataset != 'libero_goal':
    #     continue
    task_lists = os.listdir(dataset_dir)
    
    for task in task_lists:
        
        use_lang = False
        task_name = re.sub(r'^[A-Z_]+\d+_', '', task)
        if task_count in lang_mask:
            print(f"task name: {task_name}, task count: {task_count}")
            use_lang = True
            
        if not use_lang:
            task_count += 1
            continue
            
        task_dir = os.path.join(dataset_dir, task)
        demo_lists = os.listdir(task_dir)
        
        for demo in demo_lists:
            demo_count += 1
            demo_dir = os.path.join(task_dir, demo)
            
            action = np.load(demo_dir + "/action.npy")
            state = np.load(demo_dir + "/state.npy")
            with open(demo_dir + "/lang.txt", "r") as f:
                instruction = f.read()
            
            # print(f"use_lang: {use_lang}, instruction: {instruction}")
            
            length = action.shape[0]
            for i in range(length):
                save_dict = {}
                save_dict["instruction"] = instruction
                save_dict['D435_image'] = (demo_dir + f"/image0/{i}.jpg").replace(prefix, "")
                save_dict['wrist_image'] = (demo_dir + f"/image1/{i}.jpg").replace(prefix, "")
                save_dict["state"] = state[i].tolist()
                
                # action chunking a
                action_list = []
                for idx in range(6):
                    a = action[min(i+idx, length-1)].tolist()
                    action_list.append(a)

                save_dict['action'] = action_list
            
                data.append(save_dict)
                frame_count += 1
    
        task_count += 1
    
    
print('done, data number:', frame_count)
print("done, task number:", task_count)
print("done, demo number:", demo_count)
with open(f"/home/dodo/.zh1hao_space/bear_branch/BearRobot/data/libero/json_prop/libero_goal-{lang_prop}-ac.json", 'w') as f:
    json.dump(data, f, indent=4)
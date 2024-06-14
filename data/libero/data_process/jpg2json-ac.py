import json
import os
import numpy as np
from tqdm import tqdm

prefix = "/data/"
base_dir = "libero/data_jpg/"
base_dir = os.path.join(prefix, base_dir)

dataset_lists = os.listdir(base_dir)

data = []
count = 0
for dataset in tqdm(dataset_lists):
    print(dataset)
    dataset_dir = os.path.join(base_dir, dataset)
    task_lists = os.listdir(dataset_dir)
    
    for task in task_lists:
        task_dir = os.path.join(dataset_dir, task)
        demo_lists = os.listdir(task_dir)
        
        for demo in demo_lists:
            demo_dir = os.path.join(task_dir, demo)
            
            action = np.load(demo_dir + "/action.npy")
            state = np.load(demo_dir + "/state.npy")
            with open(demo_dir + "/lang.txt", "r") as f:
                instruction = f.read()
            
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
                count += 1

print('done, data number:', count)
with open("/home/dodo/ljx/BearRobot/data/libero/libero130-ac.json", 'w') as f:
    json.dump(data, f, indent=4)
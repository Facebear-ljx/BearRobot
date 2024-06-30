import json
import numpy as np
import os 
from tqdm import tqdm

base_dir = '/data/libero'
path = 'data_jpg'

dirs_path = f"{base_dir}/{path}"
save_path = "/home/dodo/ljx/BearRobot/data/libero/libero_decisionnce.json"

for i, dir_name in enumerate(tqdm(os.listdir(dirs_path))):
    save_dict = {}
    
    dir_path = os.path.join(dirs_path, dir_name)
    
    for j, task_name in enumerate(os.listdir(dir_path)):
        task_path = os.path.join(dir_path, task_name)
        
        for k, demo_name in enumerate(os.listdir(task_path)):
            
            demo_path = os.path.join(task_path, demo_name)
            
            with open(f"{demo_path}/lang.txt", 'r') as f:
                lang = f.readlines()[0]
            
                actions = np.load(f"{demo_path}/action.npy")
                length = actions.shape[0]
                save_dict.update({"length": length})
                save_dict.update({"lang": lang})
                save_dict.update({"image": ["image0", "image1"]})
                save_dict.update({"path": f"libero/{path}/{dir_name}/{task_name}/{demo_name}"})
        
            with open(save_path, 'a+') as f:
                line = json.dumps(save_dict)
                f.write(line+'\n')
                # f.write(line)        
        
print("---------------------done--------------------------")
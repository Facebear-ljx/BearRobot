import json
import numpy as np
import os 
from tqdm import tqdm

base_dir = '/data/openxdata_npy'
path = 'droid/1.0.0'

dirs_path = f"{base_dir}/{path}"
save_path = "/home/dodo/ljx/BearRobot/data/droid/droid_decicionnce.json"

for i, dir_name in enumerate(tqdm(os.listdir(dirs_path))):
    save_dict = {}
    
    dir_path = os.path.join(dirs_path, dir_name)
    with open(f"{dir_path}/action.json", 'r') as f:
        actions = json.load(f)
        
    lang = actions[0]['lang']
    if lang == '':
        print(f"{dir_name} no lang")
    else:
        length = len(actions)
        save_dict.update({"length": length})
        save_dict.update({"lang": lang})
        save_dict.update({"image": ["image0", "image1", "wrist"]})
        save_dict.update({"path": f"{path}/{dir_name}"})
        
        with open(save_path, 'a+') as f:
            line = json.dumps(save_dict)
            f.write(line+'\n')
            # f.write(line)        
        
print("---------------------done--------------------------")
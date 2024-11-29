import json
import os
import numpy as np
from tqdm import tqdm

path = "/home/dodo/ljx/BearRobot/data/libero/libero130-ac.json"
save_path = "/home/dodo/ljx/BearRobot/data/libero/libero130_no-op-ac.json"
with open(path, "r") as f:
    data_list = json.load(f)
    
new_data_list = []
prev_action = [0., 0., 0., 0., 0., 0., 1000000]
thred = 1e-4
for data in tqdm(data_list):
    action = data['action'][0]
    norm = np.linalg.norm(np.array(action[:-1]))
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    if norm <= thred and gripper_action == prev_gripper_action:
        continue
    else:
        new_data_list.append(data)
    
    prev_action = action

with open(save_path, "w") as f:
    json.dump(new_data_list, f, indent=4)
print("done")
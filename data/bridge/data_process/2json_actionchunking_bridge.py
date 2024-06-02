from tqdm import tqdm
from glob import glob
import pickle
import os
import json
import numpy as np


if __name__ == '__main__':
    relabel_action = False
    # generate json files for BridgeData
    print("========================begin_read======================")
    path = ""
    files = glob(r"/data/demos_8_17/raw/*/*/*/*/*/*/*/*/agent_data.pkl")

    print(f"===================Get Path Number {len(files)}====================")
    print(len(glob(r"/data/demos_8_17/raw/bridge_data_v1/*/*/*/*/*/*/*/agent_data.pkl")))
    print(len(glob(r"/data/demos_8_17/raw/bridge_data_v2/*/*/*/*/*/*/*/agent_data.pkl")))
    print(len(glob(r"/data/demos_8_17/raw/flap/*/*/*/*/*/*/*/agent_data.pkl")))
    print(len(glob(r"/data/demos_8_17/raw/icra/*/*/*/*/*/*/*/agent_data.pkl")))
    print(len(glob(r"/data/demos_8_17/raw/rss/*/*/*/*/*/*/*/agent_data.pkl")))
    print(len(glob(r"/data/demos_8_17/raw/rss/*/*/*/*/*/*/*/agent_data.pkl")))

    final_list = []
    legal_path = 0
    for traj in tqdm(files):
        try:
            lang = open(traj.replace("agent_data.pkl", "lang.txt"),"r").readlines()
            obs = pickle.load(open(traj.replace("agent_data.pkl", "obs_dict.pkl"), "rb"))
            policy = pickle.load(open(traj.replace("agent_data.pkl", "policy_out.pkl"), "rb"))
        except:
            print(f"traj <{traj}>:  missing package")
            continue
        legal_path+=1

        states = obs['full_state']
        full_item_len = len(policy)
        for idx in range(full_item_len):
            action_list = []
            for i in range(6):
                action = policy[min(idx+i, full_item_len-1)]['actions'].tolist()
                
                if relabel_action:
                    next_obs = states[min(idx+i+1, full_item_len)][:6]
                    curr_obs = states[min(idx+i, full_item_len-1)][:6]
                    obs_diff = (next_obs - curr_obs).reshape(1, -1)
                    gripper = policy[min(idx+i, full_item_len-1)]['actions'][-1].reshape(1, -1)
                    
                    action = np.concatenate([obs_diff, gripper], axis=-1).reshape(-1).tolist()
                else:
                    action = policy[min(idx+i, full_item_len-1)]['actions'].tolist()
                
                action_list.append(action)
                
            state = states[idx].tolist()
            
            D435_image = traj.replace("agent_data.pkl", f"images0/im_{idx}.jpg")
            wrist_image = traj.replace("agent_data.pkl", f"wrist/im_{idx}.jpg")
            
            if not os.path.isfile(D435_image): 
                continue 
            else:
                D435_image = D435_image.replace("/data/demos_8_17/raw/", "") 
                wrist_image = wrist_image.replace("/data/demos_8_17/raw/", "") 

            item = {
                "action": action_list,
                "state": state,
                "D435_image": D435_image,
                "wrist_iamge": wrist_image,
                "instruction": lang[0].strip()
            }

            final_list.append(item)

    
    print(f"================legal traj number: {legal_path} ================")
    print(f"================sample number: {len(final_list)}================")

    with open("/home/dodo/ljx/BearRobot/data/bridge/Bridgedatav2-ac.json", "w") as f:
        json.dump(final_list, f, indent=4)
from tqdm import tqdm
from glob import glob
import pickle
import os
import json
import numpy as np
import random


if __name__ == '__main__':
    # generate json files for AIR-toykitchen
    
    random.seed(42)

    print("========================begin_read======================")
    path = ""
    files = glob(r"/home/dodo/bridge_data_v2/bridge_data_robot/widowx_envs/widowx_data/robonetv2/bridge_data_v2/AIR-toykitchen/*/*/*/*/*/*/agent_data.pkl")

    print(f"===================Get Path Number {len(files)}====================")
    print(len(glob(r"/home/dodo/bridge_data_v2/bridge_data_robot/widowx_envs/widowx_data/robonetv2/bridge_data_v2/AIR-toykitchen/pick_place/00/*/*/*/*/agent_data.pkl")))
    print(len(glob(r"/home/dodo/bridge_data_v2/bridge_data_robot/widowx_envs/widowx_data/robonetv2/bridge_data_v2/AIR-toykitchen/pick_place/01/*/*/*/*/agent_data.pkl")))
    print(len(glob(r"/home/dodo/bridge_data_v2/bridge_data_robot/widowx_envs/widowx_data/robonetv2/bridge_data_v2/AIR-toykitchen/pick_place/02/*/*/*/*/agent_data.pkl")))
    print(len(glob(r"/home/dodo/bridge_data_v2/bridge_data_robot/widowx_envs/widowx_data/robonetv2/bridge_data_v2/AIR-toykitchen/pick_place/03/*/*/*/*/agent_data.pkl")))
    print(len(glob(r"/home/dodo/bridge_data_v2/bridge_data_robot/widowx_envs/widowx_data/robonetv2/bridge_data_v2/AIR-toykitchen/pick_place/04/*/*/*/*/agent_data.pkl")))
    print(len(glob(r"/home/dodo/bridge_data_v2/bridge_data_robot/widowx_envs/widowx_data/robonetv2/bridge_data_v2/AIR-toykitchen/open_close/*/*/*/*/*/agent_data.pkl")))
    print(len(glob(r"/home/dodo/bridge_data_v2/bridge_data_robot/widowx_envs/widowx_data/robonetv2/bridge_data_v2/AIR-toykitchen/flip/*/*/*/*/*/agent_data.pkl")))
    print(len(glob(r"/home/dodo/bridge_data_v2/bridge_data_robot/widowx_envs/widowx_data/robonetv2/bridge_data_v2/AIR-toykitchen/fold/*/*/*/*/*/agent_data.pkl")))
    print(len(glob(r"/home/dodo/bridge_data_v2/bridge_data_robot/widowx_envs/widowx_data/robonetv2/bridge_data_v2/AIR-toykitchen/move/*/*/*/*/*/agent_data.pkl")))
    final_list = []
    legal_path = 0
    for traj in tqdm(files):
        try:
            obs = pickle.load(open(traj.replace("agent_data.pkl", "obs_dict.pkl"), "rb"))
            lang = open(traj.replace("agent_data.pkl", "lang.txt"),"r").readlines()
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
                action_list.append(action)
            # action0 = policy[idx]['actions'].tolist()
            # action1 = policy[min(idx+1, full_item_len)]['actions'].tolist()
            # action2 = policy[min(idx+2, full_item_len-1)]['actions'].tolist()
            # action3 = policy[min(idx+3, full_item_len-1)]['actions'].tolist()
            # action4 = policy[min(idx+4, full_item_len-1)]['actions'].tolist()
            # action5 = policy[min(idx+5, full_item_len-1)]['actions'].tolist()
            # action6 = policy[min(idx+6, full_item_len-1)]['actions'].tolist()

            state = states[idx].tolist()

            D435_image = traj.replace("agent_data.pkl", f"images0/im_{idx}.jpg").replace("/home/dodo/bridge_data_v2/bridge_data_robot/widowx_envs/widowx_data/robonetv2/bridge_data_v2/", "")
            wrist_image = traj.replace("agent_data.pkl", f"images1/im_{idx}.jpg").replace("/home/dodo/bridge_data_v2/bridge_data_robot/widowx_envs/widowx_data/robonetv2/bridge_data_v2/", "")

            item = {
                "action": action_list,
                "state": state,
                "D435_image": D435_image,
                "wrist_image": wrist_image,
                "instruction": lang[0]
            }
            final_list.append(item)

    
    print(f"================legal traj number: {legal_path} ================")
    print(f"================sample number: {len(final_list)}================")

    with open("/home/dodo/ljx/BearRobot/data/airkitchen/AIR-toykitchen-ac-relative.json", "w") as f:
        json.dump(final_list, f, indent=4)




    


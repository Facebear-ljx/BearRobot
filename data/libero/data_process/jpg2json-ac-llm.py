import json
import os
import numpy as np
from tqdm import tqdm

def process_string(input_string):
    # 将第一个字符转换为小写
    input_string = input_string[0].lower() + input_string[1:]
    
    # 移除尾部的句号
    if input_string.endswith('.'):
        input_string = input_string[:-1]
    
    return input_string

prefix = "/home/dodo/zxa/BearRobot/data/libero/dataset/"
base_dir = "libero/data_jpg/"

task_suite_name = 'libero130'
base_dir = os.path.join(prefix, base_dir)

dataset_lists = os.listdir(base_dir)

data = []
count = 0
for dataset in tqdm(dataset_lists):
    print(dataset)
    # if dataset != task_suite_name:
    #     continue
    dataset_dir = os.path.join(base_dir, dataset)
    task_lists = os.listdir(dataset_dir)
    
    for task in task_lists:
        task_dir = os.path.join(dataset_dir, task)
        if not os.path.isdir(task_dir):
            continue
        demo_lists = os.listdir(task_dir)
        
        for demo in demo_lists:
            demo_dir = os.path.join(task_dir, demo)
            if not os.path.isdir(demo_dir):
                continue
            
            action = np.load(demo_dir + "/action.npy")
            state = np.load(demo_dir + "/state.npy")
            with open(demo_dir + "/lang_llm.txt", "r", encoding='gbk') as f:
                instruction = f.read()
            # 将内容编码为UTF-8，并保持字节类型
            instruction = instruction.encode('utf-8')

            # 使用字节类型的后缀进行检查
            instruction = instruction.strip()

            # 如果需要将其转换回字符串进行进一步处理
            instruction = instruction.decode('utf-8')
            instruction = instruction.lower()
            if instruction.endswith('.'):
                instruction = instruction[:-1]
            # print("instruction:", instruction)
            
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
with open(f"/home/dodo/.zh1hao_space/bear_branch/BearRobot/data/libero/{task_suite_name}-ac-llm.json", 'w') as f:
    json.dump(data, f, indent=4)
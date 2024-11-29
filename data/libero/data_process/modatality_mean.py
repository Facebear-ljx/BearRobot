from data.libero.data_process import demo2frames, get_libero_frame
from BearRobot.Net.encoder.DecisionNCE import DecisionNCE_visual_diff, DecisionNCE_encoder

from torchvision import transforms as T
from PIL import Image
import json
import numpy as np
import os
import tqdm

dataset_name = 'libero30'
view_list = ['D435_image', 'wrist_image']
json_path = f'/home/lijx/ljx/robotics/BearRobot/data/libero/{dataset_name}-ac.json'
base_dir = '/data/libero/data_jpg'

encoder = DecisionNCE_encoder(name="DecisionNCE-T_all_680ep")
model = DecisionNCE_visual_diff(encoder)
demo_dict = demo2frames.frame_counts_dict(json_path)
transform = T.ToTensor()

# visual embedding
visual_diff_list = []
for demo in tqdm.tqdm(demo_dict):
    demo_path = os.path.join(base_dir, demo)
    img_begin_path = os.path.join(demo_path, "image0/0.jpg")
    img_end_path = os.path.join(demo_path, f"image0/{demo_dict[demo]-1}.jpg")
    img_begin = transform(Image.open(img_begin_path)).unsqueeze(0)
    img_end = transform(Image.open(img_end_path)).unsqueeze(0)
    
    visual_diff = model.embed_frame(img_begin, img_end).detach().cpu().numpy()
    visual_diff_list.append(visual_diff)
    
visual_diff_all = np.concatenate(visual_diff_list)
visual_diff_all_mean = visual_diff_all.mean(axis=0, keepdims=False)


# text embedding
task_suite_list = ['libero_goal', 'libero_object', 'libero_spatial']
text_embed_list = []
postfix = 'demo_0/lang.txt'
for task_suite in task_suite_list:
    task_suite_path = os.path.join(base_dir, task_suite)
    task_dirs = os.listdir(task_suite_path)
    for task in task_dirs:
        text_path = os.path.join(task_suite_path, task, postfix)
        
        with open(text_path, "r") as f:
            text = f.readline()
            
        text_embed = model.embed_text(text).detach().cpu().numpy()
        text_embed_list.append(text_embed)
        
text_embed_all = np.concatenate(text_embed_list)
text_embed_mean = text_embed_all.mean(0, keepdims=False)

embeding_info = {"visual_mean": visual_diff_all_mean.tolist(),
                 "text_mean": text_embed_mean.tolist()}

with open("/home/lijx/ljx/robotics/BearRobot/data/libero/embedding_all680ep.json", 'w') as f:
    json.dump(embeding_info, f)
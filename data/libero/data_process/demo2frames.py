import os
import json
import re,random
from collections import defaultdict

def frame_counts_dict(json_name = 'libero_goal-ac.json'):
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir,json_name)
    with open(json_path) as json_file:
        data = json.load(json_file)
    demo_count = defaultdict(int)
    pattern = re.compile(r'([^/]+_demo)/demo_(\d+)/')

    for item in data:
        match = pattern.search(item['D435_image'])
        if match:
            instruction = match.group(1)
            demo_id = match.group(2)
            key = f"{instruction}/demo_{demo_id}"
            demo_count[key] += 1
    return demo_count

def get_demos_for_task(task_name, frame_length_dict):
    pattern = re.compile(rf'{task_name}/demo_\d+')
    return [match.group(0) for key in frame_length_dict for match in [pattern.search(key)] if match]

def test_func():
    frame_length_dict = frame_counts_dict()
    task_description = "open the middle drawer of the cabinet" 
    task_name = task_description.replace(" ", "_")+ "_demo"
    demo_paths = get_demos_for_task(task_name, frame_length_dict)
    demo_path = random.choice(demo_paths)

if __name__ == "__main__":
    test_func()
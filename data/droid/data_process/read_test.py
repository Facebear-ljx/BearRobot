import json
path = "/home/dodo/ljx/BearRobot/data/droid/droid_decicionnce.json"
data = []
with open(path, 'r') as f:
    for i in f.readlines():
        one_data = json.loads(i)
        data.append(one_data)
    
print(len(data))
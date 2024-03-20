import torch

from Net.my_model.RT_model import RT1Model
from Agent.RT_agent import RT1Agent

from utils.dataset.dataloader import RT1Dataset
from torch.utils.data import DataLoader

rt1dataset = RT1Dataset(frames=3)
rt1dataloader = DataLoader(rt1dataset, batch_size=64, shuffle=True, num_workers=1)

iterator = iter(rt1dataloader)
data = next(iterator)

rt1model = RT1Model(img_size=128, device='cuda', vision_pretrain=False).to('cuda')
rt1agent = RT1Agent(rt1model)

text = ['test1', 'test2', 'test3']

def image(img_size=(3, 128, 128)):
      return torch.randn(img_size, device='cuda')

images = [
[[image(), image()], [image(), image()], [image(), image()]],
[[image(), image()], [image(), image()], [image(), image()]],
[[image(), image()], [image(), image()], [image(), image()]],       
]

# output = rt1agent(data['images'].to('cuda'), data['lang'])

loss = rt1agent.policy_loss(data['imgs'].to('cuda'), data['lang'], data['a'].to('cuda'))

image_action = [
       [[image(), image()], [image(), image()], [image(), image()]]
]
text_action = ['action']
action = rt1agent.get_action(data['images'].to('cuda'), data['lang'])
print(output)
import torch

from Net.my_model.RT_model import RT1Model
from Agent.RT_agent import RT1Agent
from torchinfo import summary

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

output = rt1agent(images, text)

image_action = [
       [[image(), image()], [image(), image()], [image(), image()]]
]
text_action = ['action']
action = rt1agent.get_action(image_action, text_action)
print(output)
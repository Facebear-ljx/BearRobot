import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import copy

from Agent.base_agent import BaseAgent
from Net.my_model.RT_model import RT1Model


class RT1Agent(BaseAgent):
       def __init__(
              self, 
              model: RT1Model, 
       ):
              super(BaseAgent, self).__init__()
              self.model = model
              self.img_size = self.model.img_size
              
              transform = [
                     transforms.Resize(256, interpolation=Image.BICUBIC),
                     transforms.CenterCrop(224),
                     transforms.ToTensor()
              ]
              self.transform = transforms.Compose(transform)
              
              self.device = self.model.device

       def forward(self, images_list: list[list[list[torch.Tensor]]], texts: list[str], state=None):
              """
              predict the loogits
              # images_list: list of frames of different views, B x Frame x View x torch.tensor[C, H, W]
              # texts: list of instructions
              # state shape [B, D] robot arm x,y,z, gripper state, et al
              """
              # List of torch.tensor [1, Frame, View, C, H, W]
              images = [torch.stack([torch.stack([view.reshape(3, self.img_size, self.img_size) for view in frame_list]) for frame_list in one_images]).unsqueeze(0) for one_images in images_list]
              
              images = torch.cat(images, dim=0)  # tensor [B, Frame, View, C, H, W]
                            
              logits = self.model(images, texts, state)
              return logits
       
       def policy_loss(self, images_list: list[list[list[torch.Tensor]]], texts: list[str], action_gt: torch.Tensor, state=None):
              '''
              calculate RT1 loss / cross entropy
              # images: list of frames of different views, B x Frame x View x torch.tensor[C, H, W]
              # texts: list of instructions
              # state shape [B, D_s], batch of robot arm x,y,z, gripper state, et al
              # action_gt shape [B, D_a], batch of robot control value, e.g., delta_x, delta_y, delta_z,..., et al.
              '''
              logits = self(images_list, texts, state)
              _, _, N = logits.shape  # [bs, a_dim, bins]
              logits = logits.view(-1, N)
              loss = F.cross_entropy(logits, action_gt)
              return loss
       
       def q_loss(self):
              raise ValueError("RT1 agent is a BC agent, has no q value")
       
       def v_loss(self):
              raise ValueError("RT1 agent is a BC agent, has no v value")
              
       @torch.no_grad()
       def get_action(self, images: list[list[list[torch.Tensor]]], text: list[str], state=None):
              """
              get one action
              # images: list of frames of different views, 1 x Frame x View x torch.tensor[C, H, W]
              # texts: list of instruction
              # state shape [1, D] robot arm x,y,z, gripper state, et al
              """
              logits = self(images, text, state).view(-1, 256)  # [7, 256]
              joint = logits[:6].softmax(-1).max(-1).indices  # [6]
              joint = -1 + joint / 128  # decode to -1~1
              
              gripper = logits[-1].softmax(-1).max(-1).indices.unsqueeze(0) / 256 # [1]
              
              action = torch.cat([joint, gripper]).cpu().numpy()
              return action
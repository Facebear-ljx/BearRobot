import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import copy

from Agent.base_agent import BaseAgent
from Net.my_model.ACT_model import ACTModel
from Net.my_model.t5 import T5Encoder

class ACTAgent(BaseAgent):
       def __init__(
              self, 
              policy: ACTModel, 
              text_encoder = 't5',
              device = 'cuda'
       ):
              super().__init__(
                     policy,
                     None,
                     None
              )
              self.img_size = self.policy.img_size
              
              transform = [
                     transforms.Resize(256, interpolation=Image.BICUBIC),
                     transforms.CenterCrop(self.img_size),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
              ]
              self.transform = transforms.Compose(transform)
              assert text_encoder == 't5'
              self.lang_encoder = T5Encoder(device=device)
              print("lang encoder load success")
              self.device = device

       def forward(self, images: torch.Tensor, texts: list, action_gt: torch.Tensor, state=None):
              '''
              calculate ACT loss (L1 or L2) and CVAE loss
              # images: batch of frames of different views, [B, Frame, View, C, H, W]
              # texts: list of instructions
              # state shape [B, D_s], batch of robot arm x,y,z, gripper state, et al
              # action_gt shape [B, D_a], batch of robot control value, e.g., delta_x, delta_y, delta_z,..., et al.
              '''
              text_emb = self.lang_encoder.embed_text(texts).to(images.device).detach()
              loss = self.policy_loss(images, text_emb, action_gt, state)
              return loss
       
       def policy_loss(self, images: torch.Tensor, text_emb: torch.Tensor, action_gt: torch.Tensor, state=None):
              '''
              calculate ACT loss (L1 or L2) and CVAE loss
              # images: batch of frames of different views, [B, Frame, View, C, H, W]
              # text_emb: text embedding, [B, N]
              # state shape [B, D_s], batch of robot arm x,y,z, gripper state, et al
              # action_gt shape [B, D_a], batch of robot control value, e.g., delta_x, delta_y, delta_z,..., et al.
              '''
              a_hat, is_pad_hat, [mu, logvar] = self.policy(state, images, action_gt, text_emb)
              raise NotImplementedError("to be implemented soon")              
       
       def q_loss(self):
              raise ValueError("RT1 agent is a BC agent, has no q value")
       
       def v_loss(self):
              raise ValueError("RT1 agent is a BC agent, has no v value")
              
       @torch.no_grad()
       def get_action(self, images, text: list, state=None):
              """
              get one action
              # images: one frames of different views, [1, Frame, View, C, H, W]
              # texts: list of instruction
              # state shape [1, D] robot arm x,y,z, gripper state, et al
              """
              # TODO here has a bug
              if not isinstance(images, torch.Tensor):
                     # transform lists to torch.Tensor
                     images = [torch.stack([torch.stack([self.transform(view).reshape(3, self.img_size, self.img_size) for view in frame_list]) for frame_list in one_images]).unsqueeze(0) for one_images in images]
                     images = torch.cat(images, dim=0)  # tensor [1, Frame, View, C, H, W]
              
              logits = self.logits(images, text, state).view(-1, 256)  # [7, 256]
              joint = logits[:6].softmax(-1).max(-1).indices  # [6]
              joint = -1 + joint / 128  # decode to -1~1
              
              gripper = logits[-1].softmax(-1).max(-1).indices.unsqueeze(0) / 256 # [1]
              
              action = torch.cat([joint, gripper]).cpu().numpy()
              return action
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

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class ACTAgent(BaseAgent):
       def __init__(
              self, 
              policy: ACTModel, 
              text_encoder = 't5',
              kl_weight: float=10,
              loss_type: str="l1",
              device = 'cuda'
       ):
              super().__init__(
                     policy,
                     None,
                     None
              )
              
              transform = [
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
              ]
              self.transform = transforms.Compose(transform)
              assert text_encoder == 't5'
              self.lang_encoder = T5Encoder(device=device)
              print("lang encoder load success")
              
              self.kl_weight = kl_weight
              if loss_type == "l1":
                     self.loss_fn = nn.L1Loss()
              elif loss_type == 'l2':
                     self.loss_fn = nn.MSELoss()
              else:
                     raise ValueError(f"{loss_type} is not supported, only support l1 and l2 loss")
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
              a_hat, is_pad_hat, (mu, logvar) = self.policy(state, images, action_gt, text_emb)
              
              total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
              recons_loss = self.loss_fn(action_gt, a_hat)
              loss = recons_loss + self.kl_weight * total_kld[0]
              return loss
       
       def q_loss(self):
              raise ValueError("ACT agent is a BC agent, has no q value")
       
       def v_loss(self):
              raise ValueError("ACT agent is a BC agent, has no v value")
              
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
              
              text_emb = self.lang_encoder.embed_text(text).to(images.device)
              action, _, (_, _) = self.policy(state, images, text_emb)
              return action
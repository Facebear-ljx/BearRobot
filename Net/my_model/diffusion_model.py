import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Net.basic_net.mlp import MLP, MLPResNet
from Net.basic_net.resnet import ResNet
from Net.my_model.FiLM import FiLM_layer


# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
       def __init__(self, input_size, output_size):
              super().__init__()
              self.output_size = output_size
              
       def forward(self, x):
              device = x.device
              half_dim = self.output_size // 2
              f = math.log(10000) / (half_dim - 1)
              f = torch.exp(torch.arange(half_dim, device=device) * -f)
              f = x * f[None, :]
              f = torch.cat([f.cos(), f.sin()], axis=-1)
              return f

# learned positional embeds
class LearnedPosEmb(nn.Module):
       def __init__(self, input_size, output_size):
              super().__init__()
              self.output_size = output_size
              self.kernel = nn.Parameter(torch.randn(output_size // 2, input_size) * 0.2)
              
       def forward(self, x):
              f = 2 * torch.pi * x @ self.kernel.T
              f = torch.cat([f.cos(), f.sin()], axis=-1)
              return f       

TIMEEMBED = {"fixed": SinusoidalPosEmb,
             "learned": LearnedPosEmb}


# the simplest mlp model that takes
class MLPDiffusion(nn.Module):
       def __init__(
              self, 
              input_dim,
              output_dim,
              cond_dim=0,
              time_embeding='fixed',
              device='cpu',
       ):
              super(MLPDiffusion, self).__init__()
              self.input_dim = input_dim
              self.output_dim = output_dim
                
              self.cond_dim = cond_dim
              
              # base encoder
              hidden_size = 256
              self.base_model = MLP(input_dim+cond_dim, [hidden_size, hidden_size], hidden_size)
              
              # time embedding
              if time_embeding not in TIMEEMBED.keys():
                     raise ValueError(f"Invalid time_embedding '{time_embeding}'. Expected one of: {list(TIMEEMBED.keys())}")
              self.time_process = TIMEEMBED[time_embeding](1, hidden_size)
              
              # decoder
              self.decoder = MLP(hidden_size+hidden_size, [hidden_size, hidden_size], output_dim)
              
              self.device = device
              
       def forward(self, xt, t, cond=None):
              # encode
              time_embedding = self.time_process(t.view(-1, 1))
              if cond is not None:
                     xt = torch.concat([xt, cond], dim=-1)
              base_embedding = self.base_model(xt)
              embedding = torch.cat([time_embedding, base_embedding], dim=-1)
              
              # decode
              noise_pred = self.decoder(embedding)
              return noise_pred
              

# the diffusion model used in IDQL
class IDQLDiffusion(nn.Module):
       """
       the diffusion model used in IDQL: arXiv:2304.10573
       """
       def __init__(
              self, 
              input_dim,  # a dim
              output_dim,  # a dim
              cond_dim=0,  # s dim, if condition on s
              hidden_dim=256,
              num_blocks=3,
              time_dim=64,
              ac_fn='mish',
              time_embeding='fixed',
              device='cpu',
       ):
              super(IDQLDiffusion, self).__init__()
              self.input_dim = input_dim
              self.output_dim = output_dim
              self.cond_dim = cond_dim
              
              # time embedding
              if time_embeding not in TIMEEMBED.keys():
                     raise ValueError(f"Invalid time_embedding '{time_embeding}'. Expected one of: {list(TIMEEMBED.keys())}")
              self.time_process = TIMEEMBED[time_embeding](1, time_dim)
              self.time_encoder = MLP(time_dim, [128], 128, ac_fn='mish')
              
              # decoder
              self.decoder = MLPResNet(num_blocks, input_dim + 128 + cond_dim, hidden_dim, output_dim, ac_fn, True, 0.1)
              
              self.device = device
              
       def forward(self, xt, t, cond=None):
              # encode
              if not isinstance(t, torch.Tensor):
                     t = torch.tensor(t, device=self.device)
              time_embedding = self.time_process(t.view(-1, 1))
              time_embedding = self.time_encoder(time_embedding)
              if cond is not None:
                     xt = torch.concat([xt, cond], dim=-1)
              embedding = torch.cat([time_embedding, xt], dim=-1)
              
              # decode
              noise_pred = self.decoder(embedding)
              return noise_pred       

visual_dim = {"resnet18": 512,
              "resnet34": 512,
              "resnet50": 2048}

# Modified diffusion model used in bridge_data_v2: arXiv:2308.12952
class VisualDiffusion(nn.Module):
       """
       the diffusion model used in bridge_data_v2: arXiv:2308.12952
       """
       def __init__(
              self, 
              img_size: int=224,  # a dim
              view_num: int=2,
              output_dim: int=7,  # a dim
              cond_dim: int=768,  # s dim, if condition on s
              hidden_dim: int=256,
              num_blocks: int=3,
              time_dim: int=32,
              time_hidden_dim: int=256,
              vision_encoder: str='resnet18',
              vision_pretrained: bool=False,
              ft_vision: bool=True,
              norm_type: str="bn",
              pooling_type: str='avg',
              add_spatial_coordinates: bool=False,
              ac_fn: str='mish',
              time_embeding: str='learned',
              film_fusion: bool=False,
              encode_s: bool=False,
              encode_a: bool=False,
              device: str='cpu',
       ):
              """this is a vision-language diffusion model for robotics control

              Args:
                  img_size (int, optional): input image size. Defaults to 224.
                  output_dim (int, optional): the action dim. Defaults to 7.
                  cond_dim (int, optional): the conditional (language embed) information dim. Defaults to 256.
                  hidden_dim (int, optional): decoder hidden dim. Defaults to 256.
                  num_blocks (int, optional): decoder block num. Defaults to 3.
                  time_dim (int, optional): time embedding dim. Defaults to 32.
                  time_hidden_dim (int, optional): time encoder dim. Defaults to 256.
                  vision_encoder (str, optional): vision backbone name. Defaults to 'resnet18'.
                  vision_pretrained (bool, optional): load pretrained vision backbone. Defaults to False.
                  ft_vision (bool, optional): train the vision backbone. Defaults to True.
                  norm_type (str, optional): vision backbone norm type. Defaults to "bn".
                  pooling_type (str, optional): vision backbone pooling type. Defaults to 'avg'.
                  add_spatial_coordinates (bool, optional): add spatial coordinates to the image. Defaults to False.
                  ac_fn (str, optional): decoder activation. Defaults to 'mish'.
                  time_embeding (str, optional): learned or fixed time embedding. Defaults to 'learned'.
                  film_fusion (bool, optional): use film fusion for decoder. Defaults to False.
                  device (str, optional): cpu or cuda. Defaults to 'cpu'.
              """              
              super().__init__()
              self.img_size = img_size
              self.output_dim = output_dim
              self.ft_vision = ft_vision
              self.cond_dim = cond_dim
              
              # visual encoder
              assert vision_encoder in visual_dim
              self.visual_encoder = ResNet(vision_encoder, pretrained=vision_pretrained, norm_type=norm_type, pooling_type=pooling_type, add_spatial_coordinates=add_spatial_coordinates)
              self.visual_dim = visual_dim[vision_encoder]
              
              # time embedding
              if time_embeding not in TIMEEMBED.keys():
                     raise ValueError(f"Invalid time_embedding '{time_embeding}'. Expected one of: {list(TIMEEMBED.keys())}")
              self.time_process = TIMEEMBED[time_embeding](1, time_dim)
              self.time_encoder = MLP(time_dim, [time_hidden_dim], time_hidden_dim, ac_fn='mish')
              
              # film condition layer
              feature_info = self.visual_encoder.model.feature_info[1:]
              self.film_layer = nn.ModuleList()
              idx = 0
              for name, child in self.visual_encoder.model.named_children():
                     if isinstance(child, nn.Sequential):
                            feature_dim = feature_info[idx]["num_chs"]
                            feature_depth = len(child)
                            self.film_layer.extend([FiLM_layer(cond_dim, feature_dim) for _ in range(feature_depth)])
                            idx += 1
              
              # state encoder
              self.encode_s, self.encode_a = encode_s, encode_a
              self.film_fusion = film_fusion
              self.state_encoder = nn.Linear(7, hidden_dim) if self.encode_s else None # HARD CODE
              s_dim = hidden_dim if self.encode_s else 7 # HARD CODE
              if self.film_fusion:
                     # add image, state, time embedding as film condition to action decoder
                     input_dim = output_dim
                     self.decoder = MLPResNet(num_blocks, input_dim, hidden_dim, output_dim, ac_fn, True, 0.1)
                     
                     cond_dim = self.visual_dim * view_num + time_hidden_dim + s_dim
                     self.film_fusion_layer = nn.ModuleList()
                     for name, child in self.decoder.named_children():
                            if isinstance(child, nn.ModuleList):
                                   for block in child:
                                          feature_dim = block.hidden_dim
                                          self.film_fusion_layer.append(FiLM_layer(cond_dim, feature_dim))
              else:
                     # simply concat image, state, time embedding and action as input to action decoder
                     self.action_encoder = nn.Linear(output_dim, hidden_dim) if self.encode_a else None
                     a_dim = hidden_dim if self.encode_a else output_dim

                     # decoder
                     input_dim = self.visual_dim * view_num + time_hidden_dim + s_dim + a_dim
                     self.decoder = MLPResNet(num_blocks, input_dim, hidden_dim, output_dim, ac_fn, True, 0.1)
              
              self.device = device      

       def forward_visual_feature(self, images: torch.Tensor, condition: torch.Tensor):
              # images shape [B, F, View, C, H, W]
              # condition shape [B, D]
              assert images.shape[0] == condition.shape[0]
              B, F, V, C, H, W = images.shape
              B, D = condition.shape
              
              condition = condition.unsqueeze(1).repeat(1, F*V, 1).view(B*F*V, self.cond_dim)       
              
              # image feature
              image_feature = images.view(B * F * V, C, H, W)
              image_feature = self.visual_encoder.stem(image_feature)
              
              # film
              visual_blocks = []
              for name, child in self.visual_encoder.model.named_children():
                     if isinstance(child, nn.Sequential):
                            for visual_block in child:
                                    visual_blocks.append(visual_block)
              
              for visual_block, film_layer in zip(visual_blocks, self.film_layer):
                     if self.ft_vision:
                            image_feature = visual_block(image_feature)
                     else:
                            with torch.no_grad():
                                   image_feature = visual_block(image_feature)
                     image_feature = film_layer(condition, image_feature)
              
              # output feature
              image_feature = self.visual_encoder.model.global_pool(image_feature)
              image_feature = self.visual_encoder.model.fc(image_feature)
              return image_feature.view(B, F * V * self.visual_dim)
 
       
       def forward(self, xt: torch.Tensor, t: torch.Tensor, imgs: torch.Tensor, cond: list=None, state=None):
              """_summary_

              Args:
                  xt (torch.Tensor): noisy action
                  t (torch.Tensor): time step
                  imgs (torch.Tensor): [Batch, Frames, Views, C, H, W], batch of frames of different views
                  cond (list): language condition. Defaults to None.
                  state (_type_, optional): robot arm state. Defaults to None.

              Raises:
                  NotImplementedError: _description_

              Returns:
                  torch.Tensor: predicted noise
              """
              # flatted xt
              xt = xt.reshape([xt.shape[0], -1])
              state = state.reshape([state.shape[0], -1])
              s_feature = self.state_encoder(state) if self.encode_s else state
              
              # encode
              if not isinstance(t, torch.Tensor):
                     t = torch.tensor(t, device=self.device)
              time_embedding = self.time_process(t.view(-1, 1))
              time_embedding = self.time_encoder(time_embedding)
              if cond is not None:
                     image_feature = self.forward_visual_feature(imgs, cond)
              else:
                     raise NotImplementedError(f"cond must be given, not None")
              
              if self.film_fusion:
                     output = self.decoder.dense1(xt)
                     condition = torch.concat([image_feature, time_embedding, s_feature], dim=-1)
                     for block, film_layer in zip(self.decoder.mlp_res_blocks, self.film_fusion_layer):
                            output = block(output)
                            output = film_layer(condition, output)
                     noise_pred = self.decoder.dense2(self.decoder.ac_fn(output))
                     return noise_pred
              else:
                     xt_feature = self.action_encoder(xt) if self.encode_a else xt
                     input_feature = torch.concat([image_feature, time_embedding, xt_feature, s_feature], dim=-1)
                     
                     # decode
                     noise_pred = self.decoder(input_feature)
              return noise_pred  


# TODO, implement more customized diffusion model
    
       
if __name__ =='__main__':
       embed = SinusoidalPosEmb(128)
       embeded_t = embed(torch.tensor((1.,)))
       print(embeded_t)
       
       # model = MLPDiffusion(100, 100).to(device)
       # x = torch.randn(256, 100).to(device)
       t = torch.randint(0, 1000, (256)).to('device')
       # print(model(x, t))
       
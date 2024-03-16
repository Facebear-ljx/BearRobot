import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, reduce, repeat

from Net.basic_net.mlp import MLP

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
       def __init__(self, output_size):
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

TIMEEMBED = {"fixed": SinusoidalPosEmb}

# the simplest mlp model that takes
class MLPDiffusion(nn.Module):
       def __init__(
              self, 
              input_dim,
              output_dim,
              cond_dim=0,
              time_embeding='fixed',
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
              self.time_process = TIMEEMBED[time_embeding](hidden_size)
              
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
              
              


# TODO, implement more customized diffusion model


if __name__ =='__main__':
       embed = SinusoidalPosEmb(128)
       embeded_t = embed(torch.tensor((1.,)))
       print(embeded_t)
       
       # model = MLPDiffusion(100, 100).to(device)
       # x = torch.randn(256, 100).to(device)
       t = torch.randint(0, 1000, (256)).to(device)
       # print(model(x, t))
       
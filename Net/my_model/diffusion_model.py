import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, reduce, repeat

from Net.basic_net.mlp import MLP, MLPResNet

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
              f = x * f
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
              time_embedding = self.time_process(t.view(-1, 1))
              time_embedding = self.time_encoder(time_embedding)
              if cond is not None:
                     xt = torch.concat([xt, cond], dim=-1)
              embedding = torch.cat([time_embedding, xt], dim=-1)
              
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
       t = torch.randint(0, 1000, (256)).to('device')
       # print(model(x, t))
       
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def linear_beta_schedule(timesteps):
       """
       linear schedule, proposed in original ddpm paper
       """
       scale = 1000 / timesteps
       beta_start = scale * 0.0001
       beta_end = scale * 0.02
       return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
       """
       cosine schedule
       as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
       """
       steps = timesteps + 1
       t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
       alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
       alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
       betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
       return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
       """
       sigmoid schedule
       proposed in https://arxiv.org/abs/2212.11972 - Figure 8
       better for images > 64x64, when used during training
       """
       steps = timesteps + 1
       t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
       v_start = torch.tensor(start / tau).sigmoid()
       v_end = torch.tensor(end / tau).sigmoid()
       alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
       alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
       betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
       return torch.clip(betas, 0, 0.999)


def vp_beta_schedule(timesteps):
       """Discret VP noise schedule
       """
       t = torch.arange(1, timesteps + 1)
       T = timesteps
       b_max = 10.
       b_min = 0.1
       alpha = torch.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
       betas = 1 - alpha
       return betas       


SCHEDULE = {'linear': linear_beta_schedule,
            'cosine': cosine_beta_schedule,
            'sigmoid': sigmoid_beta_schedule,
            'vp': vp_beta_schedule}


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
              f = x[:, None] * f[None, :]
              f = torch.cat([f.cos(), f.sin()], axis=-1)
              return f
                     


class DDPM():
       def __init__(
              self, 
              model, 
              schedule='cosine',
       ):
              super().__init__()
              self.model = model
              self.time_model = SinusoidalPosEmb(128)
              self.schedule = SCHEDULE[schedule]

       def forward(self, x, t, cond=None):
              x_flat = x.view(x.size(0), -1)
              noise_pred = self.predictor(x_flat)
              
              return noise_pred.view_as(x)


if __name__ == '__main__':
       timesteps = 100
       schedule = SCHEDULE['linear']
       print(schedule(timesteps)[0])
       
       embed = SinusoidalPosEmb(128)
       embeded_t = embed(torch.tensor(1.))
       print(embeded_t)
       
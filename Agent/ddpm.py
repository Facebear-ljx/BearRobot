import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Agent.base_agent import BaseAgent


def extract(a, x_shape):
       '''
       align the dimention of alphas_cumprod_t to x_shape
       
       a: alphas_cumprod_t, B
       x_shape: B x F x F x F
       output: alphas_cumprod_t B x 1 x 1 x 1]
       '''
       b, *_ = a.shape
       return a.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
       """
       linear schedule, proposed in original ddpm paper
       """
       scale = 1000 / timesteps
       beta_start = scale * 0.0001
       beta_end = scale * 0.02
       return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
       """
       cosine schedule
       as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
       """
       steps = timesteps + 1
       t = torch.linspace(0, timesteps, steps) / timesteps
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
       t = torch.linspace(0, timesteps, steps) / timesteps
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
                     
class DDPM_Agent(BaseAgent):
       def __init__(
              self, 
              model, 
              schedule='cosine',
              num_timesteps=1000,
       ):
              super(BaseAgent, self).__init__()
              self.model = model
              
              self.device = model.device
              if schedule not in SCHEDULE.keys():
                     raise ValueError(f"Invalid schedule '{schedule}'. Expected one of: {list(SCHEDULE.keys())}")
              self.schedule = SCHEDULE[schedule]
              
              self.num_timesteps = num_timesteps
              self.betas = self.schedule(self.num_timesteps).to(self.device)
              self.alphas = (1 - self.betas).to(self.device)
              self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)

       def forward(self, xt, t, cond=None):
              """
              predict the noise
              """
              noise_pred = self.model(xt, t, cond)
              return noise_pred
       
       def loss(self, x0, cond=None):
              '''
              calculate ddpm loss
              '''
              batch_size = x0.shape[0]
              
              noise = torch.randn_like(x0, device=self.device)
              t = torch.randint(0, self.num_timesteps, (batch_size, ), device=self.device)
              
              xt = self.q_sample(x0, t, noise)
              
              noise_pred = self.forward(xt, t, cond)
              loss = F.mse_loss(noise_pred, noise)
              
              return loss
              
       def q_sample(self, x0, t, noise):
              """
              sample noisy xt from x0, q(xt|x0), forward process
              """
              alphas_cumprod_t = self.alphas_cumprod[t]
              xt = x0 * extract(torch.sqrt(alphas_cumprod_t), x0.shape) + noise * extract(torch.sqrt(1 - alphas_cumprod_t), x0.shape)
              return xt
       
       @torch.no_grad()
       def p_sample(self, xt, t, cond=None, guidance_strength=0):
              """
              sample xt-1 from xt, p(xt-1|xt)
              """
              noise_pred = self.forward(xt, t, cond)
              
              alpha1 = 1 / torch.sqrt(self.alphas[t])
              alpha2 = (1 - self.alphas[t]) / (torch.sqrt(1 - self.alphas_cumprod[t]))
              
              xtm1 = alpha1 * (xt - alpha2 * noise_pred)
              
              noise = torch.randn_like(xtm1, device=self.device)
              xtm1 = xtm1 + (t > 0) * (torch.sqrt(self.betas[t]) * noise)
              return xtm1
       
       @torch.no_grad()
       def ddpm_sampler(self, shape, cond=None, guidance_strength=0):
              """
              sample x0 from xT, reverse process
              """
              x = torch.randn(shape, device=self.device)
              
              for t in reversed(range(self.num_timesteps)):
                     x = self.p_sample(x, torch.full((shape[0], 1), t, device=self.device), cond)
              return x

       @torch.no_grad()
       def get_action(self, state):
              return self.ddpm_sampler((1, self.model.output_dim), cond=state)
       

if __name__ == '__main__':
       timesteps = 100
       schedule = SCHEDULE['linear']
       print(schedule(timesteps)[0])

       
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

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
                     
class DDPM_BC(BaseAgent):
       def __init__(
              self, 
              model: torch.nn.Module, 
              schedule: str='cosine',
              num_timesteps: int=5,
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
       
       def policy_loss(self, x0, cond=None):
              '''
              calculate ddpm loss
              '''
              batch_size = x0.shape[0]
              
              noise = torch.randn_like(x0, device=self.device)
              t = torch.randint(0, self.num_timesteps, (batch_size, ), device=self.device)
              
              xt = self.q_sample(x0, t, noise)
              
              noise_pred = self.forward(xt, t, cond)
              loss = (((noise_pred - noise) ** 2).sum(axis = -1)).mean()
              
              return loss
              
       def q_sample(self, x0, t, noise):
              """
              sample noisy xt from x0, q(xt|x0), forward process
              """
              alphas_cumprod_t = self.alphas_cumprod[t]
              xt = x0 * extract(torch.sqrt(alphas_cumprod_t), x0.shape) + noise * extract(torch.sqrt(1 - alphas_cumprod_t), x0.shape)
              return xt
       
       @torch.no_grad()
       def p_sample(self, xt, t, cond=None, guidance_strength=0, clip_sample=True, ddpm_temperature=1.):
              """
              sample xt-1 from xt, p(xt-1|xt)
              """
              noise_pred = self.forward(xt, t, cond)
              
              alpha1 = 1 / torch.sqrt(self.alphas[t])
              alpha2 = (1 - self.alphas[t]) / (torch.sqrt(1 - self.alphas_cumprod[t]))
              
              xtm1 = alpha1 * (xt - alpha2 * noise_pred)
              
              noise = torch.randn_like(xtm1, device=self.device) * ddpm_temperature
              xtm1 = xtm1 + (t > 0) * (torch.sqrt(self.betas[t]) * noise)
              
              if clip_sample:
                     xtm1 = torch.clip(xtm1, -1., 1.)
              return xtm1
       
       @torch.no_grad()
       def ddpm_sampler(self, shape, cond=None, guidance_strength=0):
              """
              sample x0 from xT, reverse process
              """
              x = torch.randn(shape, device=self.device)
              cond = cond.repeat(x.shape[0], 1)
              
              for t in reversed(range(self.num_timesteps)):
                     x = self.p_sample(x, torch.full((shape[0], 1), t, device=self.device), cond)
              return x

       @torch.no_grad()
       def get_action(self, state, num=1):
              return self.ddpm_sampler((num, self.model.output_dim), cond=state)
       

class IDQL_Agent(BaseAgent):
       def __init__(
              self, 
              policy_model: torch.nn.Module,
              v_model: torch.nn.Module, 
              q_models: torch.nn.Module,
              schedule: str='vp', 
              num_timesteps: int=5,
              num_sample: int=64,
              gamma: float=0.99,
              expectile: float=0.7,
       ):       
              super(BaseAgent, self).__init__()
              ddpm_policy = DDPM_BC(policy_model, schedule=schedule, num_timesteps=num_timesteps)
              self.policy = ddpm_policy
              self.policy_target = copy.deepcopy(ddpm_policy)
              self.v_model = v_model
              self.q_models = q_models
              self.q_models_target = copy.deepcopy(q_models)
              
              self.gamma = gamma
              self.expectile = expectile
              self.num_sample = num_sample
              self.device = self.policy.device
              

       def q_loss(self, s, a, r, next_s, d):
              with torch.no_grad():
                     target_q = r + self.gamma * d * self.v_model(next_s)
              
              qs = self.q_models(s, a)
              
              loss = [F.mse_loss(q, target_q) for q in qs]
              loss = sum(loss) / len(loss)
              return loss
       
       
       def v_loss(self, s, a):
              with torch.no_grad():
                     target_v = torch.cat(self.q_models_target(s, a), axis=1).min(axis=1, keepdim=True)[0]
              
              v = self.v_model(s)
              loss = self.expectile_loss(target_v - v)
              return loss, v.mean()
       
       
       def policy_loss(self, x0, cond=None):
              loss = self.policy.policy_loss(x0, cond)
              return loss
       
       
       def expectile_loss(self, diff):
              weight = torch.where(diff > 0, self.expectile, (1 - self.expectile))
              return torch.mean(weight * (diff**2))
       
       
       @torch.no_grad
       def get_action(self, state, from_target=True):
              if from_target:
                     actions = self.policy_target.get_action(state, self.num_sample)
              else:
                     actions = self.policy.get_action(state, self.num_sample)
              
              state = state.repeat(actions.shape[0], 1)
              qs = torch.cat(self.q_models_target(state, actions), axis=1).min(axis=1)[0]
              idx = torch.argmax(qs)
              
              action = actions[idx]
              return action
              
              
              

if __name__ == '__main__':
       timesteps = 100
       schedule = SCHEDULE['linear']
       print(schedule(timesteps)[0])

       
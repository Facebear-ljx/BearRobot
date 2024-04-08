import torch
import torch.nn as nn
import copy

class BaseAgent(nn.Module):
       def __init__(
              self, 
              policy: torch.nn.Module,
              v_model: torch.nn.Module, 
              q_models: torch.nn.Module,
              gamma: float=0.99,
              utd: int=2,
              start_steps: int=int(25e3),
       ):       
              super().__init__()
              self.policy = policy
              # self.policy_target = copy.deepcopy(policy)
              self.v_model = v_model
              self.q_models = q_models
              self.q_models_target = copy.deepcopy(q_models)
              
              self.start_steps = start_steps
              self.utd = utd
              self.gamma = gamma
              self.device = policy.device
              
       def get_action(self, state):
              """
              get the action during evaluation
              """
              pass
       
       def explore_action(self, state):
              """
              get the action during online interaction, used only for online RL methods
              """
              pass
       
       def load(self, ckpt_path):
              pass
       
       def policy_loss(self):
              pass
       
       def q_loss(self):
              pass
       
       def v_loss(self):
              pass
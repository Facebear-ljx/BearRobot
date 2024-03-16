import torch
import torch.nn as nn

class BaseAgent(nn.Module):
       def __init__(
              self,
              config=None,
       ):
              self.config = config
              
       def get_action(self, state):
              pass
       
       def load(self, ckpt_path):
              pass
       
       def loss(self):
              pass
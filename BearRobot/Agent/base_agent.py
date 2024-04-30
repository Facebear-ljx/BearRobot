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

       def get_statistics(self, path):
              from mmengine import fileio
              import json
              def openjson(path):
                     value  = fileio.get_text(path)
                     dict = json.loads(value)
                     return dict

              self.statistics = openjson(path)
              self.a_max = torch.tensor(self.statistics['a_max'])
              self.a_min = torch.tensor(self.statistics['a_min'])
              self.a_mean = torch.tensor(self.statistics['a_mean'])
              self.a_std = torch.tensor(self.statistics['a_std'])

              self.s_mean = torch.tensor(self.statistics['s_mean'])
              self.s_std = torch.tensor(self.statistics['s_std'])
              print('dataset statistics load succ:')
              print_dict = json.dumps(self.statistics, indent=4, sort_keys=True)
              print(print_dict)


       def get_transform(self, img_size):
              from PIL import Image
              from torchvision import transforms
              transform_list  = [
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]
              if img_size != 0:
                     transform_list.insert(0, transforms.CenterCrop(img_size))
                     transform_list.insert(0, transforms.Resize(256, interpolation=Image.BICUBIC))       
              self.transform = transforms.Compose(transform_list)
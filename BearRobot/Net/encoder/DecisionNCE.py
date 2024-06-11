import DecisionNCE
import torch
import torch.nn as nn

   
class DecisionNCE_encoder(nn.Module):
       def __init__(
              self,
              name: str="DecisionNCE-T",
              device: int=0,
              *args,
              **kwargs
       ):
             super().__init__()
             self.name = name
             self.device = device
             
             self.model = DecisionNCE.load(name, device=device)
             self.v_dim = self.model.model.visual.output_dim 
             self.l_dim = self.v_dim
             
       def encode_image(self, imgs: torch.Tensor):
              image_features = self.model.encode_image(imgs)
              return image_features
       
       def encode_lang(self, langs: list):
              lang_features = self.model.encode_text(langs)
              return lang_features


class DecisionNCE_visual(nn.Module):
       def __init__(
              self,
              mm_encoder: DecisionNCE_encoder,
       ):
              super().__init__()
              self.mm_encoder = mm_encoder
       
       def forward(self, imgs: torch.Tensor):
              image_features = self.mm_encoder.encode_image(imgs)
              return image_features
              
              

class DecisionNCE_lang(nn.Module):
       def __init__(
              self,
              mm_encoder: DecisionNCE_encoder,
       ):
              super().__init__()
              self.mm_encoder = mm_encoder
              
       def forward(self, langs: list):
              lang_features = self.mm_encoder.encode_lang(langs)
              return lang_features
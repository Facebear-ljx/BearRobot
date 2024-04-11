import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm

norm_layer = {"bn": nn.BatchNorm2d, 
              "gn": nn.GroupNorm}

pooling_layer = {"avg": nn.AdaptiveAvgPool2d, 
                 "max": nn.AdaptiveMaxPool2d}

class ResNet(nn.Module):
       def __init__(self, 
              model_name: str = "resnet18",
              pretrained: bool = False,
              norm_type: str = "bn",
              pooling_type: str = "avg",
              *args, **kwargs) -> None:
              super().__init__(*args, **kwargs)
              self.model = timm.create_model(model_name, pretrained=pretrained)

              self.norm_type = norm_type
              assert norm_type in norm_layer
              self.norm_layer = norm_layer[norm_type]
              
              assert pooling_type in pooling_layer
              self.pooling_layer = pooling_layer[pooling_type]
              
              if norm_type is not "bn":
                     self._replace_bn()

              del self.model.fc
              self.model.fc = nn.Identity()
              
       def forward(self, img):
              output = self.model(img)
              return output
       
       
       def _replace_bn(self):
              root_module = self.model
              bn_list = [k.split('.') for k, m in self.model.named_modules() if isinstance(m, nn.BatchNorm2d)]
              
              for *parent, k in bn_list:
                     parent_module = root_module
                     if len(parent) > 0:
                            parent_module = root_module.get_submodule('.'.join(parent))
                     if isinstance(parent_module, nn.Sequential):
                            src_module = parent_module[int(k)]
                     else:
                            src_module = getattr(parent_module, k)
                     
                     tgt_module = self.norm_layer(src_module.num_features//16, src_module.num_features)
                     
                     if isinstance(parent_module, nn.Sequential):
                            parent_module[int(k)] = tgt_module
                     else:
                            setattr(parent_module, k, tgt_module)
                     
              # verify that all BN are replaced
              bn_list = [k.split('.') for k, m in self.model.named_modules() if isinstance(m, nn.BatchNorm2d)]
              assert len(bn_list) == 0
              
              
              
if __name__ == '__main__':
       timmresnet = ResNet(model_name="resnet50", norm_type="gn")
       x = torch.randn(1, 3, 224, 224)
       print(timmresnet(x).shape)
      
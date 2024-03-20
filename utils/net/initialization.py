import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weight(m):
       if isinstance(m, nn.Linear):
              nn.init.kaiming_normal_(m.weight)
              if m.bias is not None:
                     nn.init.constant_(m.bias, 0)
       elif isinstance(m, nn.LayerNorm):
              nn.init.constant_(m.bias, 0)
              nn.init.constant_(m.weight, 1.0)
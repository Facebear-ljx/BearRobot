import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm


OPTIMIZER = {"adam": torch.optim.Adam}

class DiffusionTrainer:
       def __init__(
              self,
              diffusion,
              train_dataloader,
              val_dataloader,
              lr=1e-4,
              optimizer='adam',
       ):
              self.diffusion = diffusion
              self.train_dataloader = train_dataloader # TODO to be implemented
              self.val_dataloader = val_dataloader # TODO to be implemented
              self.optimizer = OPTIMIZER[optimizer](self.diffusion.parameters(), lr=lr)
              self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
              
       def train_epoch(self, epochs):
              """
              train some epochs
              """
              self.diffusion.train()
              for epoch in tqdm.tqdm(range(epochs)):
                     for batch in self.train_dataloader:
                            # TODO add train code here
                            
                            # self.optimizer.zero_grad()
                            
                            # loss = self.diffusion.loss(x0)
                            
                            # loss.backward()
                            
                            # self.optimizer.step()
                            pass
                            
       
       def train_steps(self, steps):
              """
              train some gradient steps
              """
              pass
              
       def save_model(self, path):
              """
              save the model to path
              """
              torch.save(self.diffusion.state_dict(), path)
              
       def load_model(self, path):
              """
              load ckpt from path
              """
              self.diffusion.load_state_dict(torch.load(path, map_location=self.device))
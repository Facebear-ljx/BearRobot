import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


OPTIMIZER = {"adam": torch.optim.Adam}

class DiffusionTrainer:
       def __init__(
              self,
              diffusion,
              train_dataloader,
              val_dataloader,
              logger,
              lr=1e-4,
              optimizer='adam',
              device='cpu',
       ):
              # model
              self.diffusion = diffusion
              
              # dataloader
              self.train_dataloader = train_dataloader
              self.val_dataloader = val_dataloader
              
              # optimizer
              self.optimizer = OPTIMIZER[optimizer](self.diffusion.parameters(), lr=lr)
              
              # logger
              self.logger = logger
              self.device = device
              
       def train_epoch(self, epochs):
              """
              train some epochs
              """
              self.diffusion.train()
              for epoch in range(0, epochs):
                     epoch_loss = 0.
                     with tqdm(self.train_dataloader, unit="batch") as pbar:
                            for batch in pbar:
                                   s = batch['s'].to(self.device)
                                   
                                   self.optimizer.zero_grad()
                                   loss = self.diffusion.loss(s)
                                   loss.backward()
                                   self.optimizer.step()
                                   
                                   pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")
                                   epoch_loss += loss.item()
                     
                     avg_loss = epoch_loss / len(self.train_dataloader)
                     self.logger.log_metrics({"train/loss": avg_loss}, step=epoch)
                     print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
              
              self.logger.finish()
                     
                            
       
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
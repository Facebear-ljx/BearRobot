import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


OPTIMIZER = {"adam": torch.optim.Adam}

class DiffusionBCTrainer:
       def __init__(
              self,
              diffusion,
              train_dataloader,
              val_dataloader,
              logger,
              evaluator,
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
              
              # evaluator
              self.evaluator = evaluator
              
       def train_epoch(self, epochs):
              """
              train some epochs
              """
              self.diffusion.train()
              self.evaluator.eval_episodes(self.diffusion, 0)
              for epoch in range(0, epochs):
                     epoch_loss = 0.
                     with tqdm(self.train_dataloader, unit="batch") as pbar:
                            for batch in pbar:
                                   cond = batch['s'].to(self.device)
                                   x = batch['a'].to(self.device)
                                   
                                   self.optimizer.zero_grad()
                                   loss = self.diffusion.loss(x, cond)
                                   loss.backward()
                                   self.optimizer.step()
                                   
                                   pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")
                                   epoch_loss += loss.item()
                     
                     avg_loss = epoch_loss / len(self.train_dataloader)
                     self.logger.log_metrics({"train/loss": avg_loss}, step=epoch)
                     self.evaluator.eval_episodes(self.diffusion, epoch+1)
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
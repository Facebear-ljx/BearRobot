import time

import torch
from torch.utils.data import DataLoader

from Net.my_model.diffusion_model import MLPDiffusion
from Agent.ddpm import DDPM
from utils.dataset.dataloader import AIROpenXDataset, D4RLDataset
from Trainer.trainer import DiffusionBCTrainer
from utils.evaluation.d4rl_eval import D4RLEval
from utils.logger.wandb_log import WandbLogger

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # test MLPDiffusion
# model = MLPDiffusion(100, 100).to(device)

# x = torch.randn(256, 100).to(device)
# t = torch.randint(0, 1000, (256, )).to(device)
# a = model(x, t)

# # test DDPM
# test_ddpm = DDPM(model)

# x = torch.randn(256, 100).to(device)

# loss = test_ddpm.loss(x)

# sample = test_ddpm.ddpm_sampler((256, 100))

# # test AIROpenXDataset
# openx_dataset = AIROpenXDataset()
# openx_dataset.__getitem__(1)
# openx_dataloader = DataLoader(openx_dataset, batch_size=64, shuffle=True, num_workers=8)
# openx_iterator = iter(openx_dataloader)

# # test D4RLDataset
# d4rl_dataset = D4RLDataset('halfcheetah-medium-v2')
# d4rl_dataset.__getitem__(1)
# d4rl_dataloader = DataLoader(d4rl_dataset, batch_size=64, shuffle=True, num_workers=8)
# d4rl_iterator = iter(d4rl_dataloader)

# test Trainer
model = MLPDiffusion(6, 6, 17).to(device)
test_ddpm = DDPM(model, num_timesteps=15)

env_name = 'halfcheetah-medium-v2'
d4rl_dataset = D4RLDataset(env_name)
d4rl_dataloader = DataLoader(d4rl_dataset, batch_size=256, shuffle=True, num_workers=8)
wandb_logger = WandbLogger(project_name='name', config={"nothing": None})
evaluator = D4RLEval(env_name, d4rl_dataset.data_statistics, wandb_logger, 10)
test_trainer = DiffusionBCTrainer(test_ddpm, d4rl_dataloader, d4rl_dataloader, wandb_logger, evaluator, device='cuda')
test_trainer.train_epoch(250)    
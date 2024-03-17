import time

import torch
from torch.utils.data import DataLoader

from Net.my_model.diffusion_model import MLPDiffusion, IDQLDiffusion
from Agent.ddpm_bc import DDPM_BC
from utils.dataset.dataloader import AIROpenXDataset, D4RLDataset
from Trainer.trainer import DiffusionBCTrainer
from utils.evaluation.d4rl_eval import D4RLEval
from utils.logger.wandb_log import WandbLogger

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# dataset and dataloader
env_name = 'hopper-medium-v2'
d4rl_dataset = D4RLDataset(env_name)
d4rl_dataloader = DataLoader(d4rl_dataset, batch_size=1024, shuffle=True, num_workers=8)

# agent and the model for agent
model = IDQLDiffusion(d4rl_dataset.a_dim, d4rl_dataset.a_dim, d4rl_dataset.s_dim, device=device).to(device)
test_ddpm = DDPM_BC(model, num_timesteps=5)

# logger
wandb_logger = WandbLogger(project_name='name', run_name=env_name, config={"env_name": env_name})

# evaluator
evaluator = D4RLEval(env_name, d4rl_dataset.data_statistics, wandb_logger, 10, 25000)

# trainer
test_trainer = DiffusionBCTrainer(test_ddpm, d4rl_dataloader, d4rl_dataloader, wandb_logger, evaluator, int(1e+6), lr=3e-4, device=device)
test_trainer.train_steps()    
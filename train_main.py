import time

import torch
from torch.utils.data import DataLoader

from Net.my_model.diffusion_model import MLPDiffusion, IDQLDiffusion
from Net.my_model.critic_model import MLPV, MLPQs
from Agent.ddpm_bc import DDPM_BC, IDQL_Agent

from utils.dataset.dataloader import AIROpenXDataset, D4RLDataset
from Trainer.trainer import DiffusionBCTrainer, RLTrainer

from utils.evaluation.d4rl_eval import D4RLEval
from utils.logger.wandb_log import WandbLogger

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# dataset and dataloader
env_name = 'hopper-medium-v2'
d4rl_dataset = D4RLDataset(env_name)
d4rl_dataloader = DataLoader(d4rl_dataset, batch_size=2048, shuffle=True, num_workers=16)

# agent and the model for agent
model = IDQLDiffusion(d4rl_dataset.a_dim, d4rl_dataset.a_dim, d4rl_dataset.s_dim, time_embeding='learned', device=device).to(device)
ddpm_policy = DDPM_BC(model, schedule='vp', num_timesteps=5)
v_model = MLPV(d4rl_dataset.s_dim, 1)
qs_model = MLPQs(d4rl_dataset.s_dim + d4rl_dataset.a_dim, 1, ensemble_num=2)
idql_agent = IDQL_Agent(ddpm_policy, v_model, qs_model, num_sample=64).to(device)

# logger
wandb_logger = WandbLogger(project_name='name', run_name=env_name, config={"env_name": env_name})

# evaluator
evaluator = D4RLEval(env_name, d4rl_dataset.data_statistics, wandb_logger, 10, 25000)

# trainer
test_trainer = RLTrainer(idql_agent, d4rl_dataloader, d4rl_dataloader, wandb_logger, evaluator, int(1e+6), lr=3e-4, device=device)
test_trainer.train_steps()    
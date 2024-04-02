import torch
import numpy as np
from torch.utils.data import DataLoader

from Net.my_model.diffusion_model import MLPDiffusion, IDQLDiffusion
from Net.my_model.critic_model import MLPV, MLPQs
from Agent.ddpm_bc import DDPM_BC, IDQL_Agent

from utils.dataset.d4rl_dataloader import D4RLDataset
from Trainer.trainer import RLTrainer

from utils.evaluation.d4rl_eval import D4RLEval
from utils.logger.wandb_log import WandbLogger

import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
       # customize your argparser
       parser = argparse.ArgumentParser(description='An example to use this model')
       parser.add_argument('--device', default='cuda', help='cuda or cpu')
       parser.add_argument('--project_name', default='IDQL_pytorch_example', help='your project name')
       parser.add_argument('--env_name', default='hopper-medium-expert-v2', help='choose your mujoco env')
       parser.add_argument('--steps', default=1.5e+6, type=float, help='train steps')
       parser.add_argument("--seed", default=42, type=int)  # Sets Gym, PyTorch and Numpy seeds
       parser.add_argument("--expectile", default=0.7, type=float, help="expectile value in IQL")
       parser.add_argument('--batch_size', default=2048, type=int)
       parser.add_argument('--num_samples', default=64, type=int, help='evaluation sample action nums')
       parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
       parser.add_argument('--T', default=5, type=int, help='maximize diffusion time steps')
       parser.add_argument('--time_embed', default='learned', type=str, help='learned or fixed type time embedding')
       parser.add_argument('--beta', default='vp', type=str, help='noise schedule')
       
       args = parser.parse_args()

       # seed
       seed = args.seed
       np.random.seed(seed)
       torch.manual_seed(seed)
       
       # dataset and dataloader
       env_name = args.env_name
       d4rl_dataset = D4RLDataset(env_name)
       d4rl_dataloader = DataLoader(d4rl_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)

       # agent and the model for agent
       ## ddpm policy
       policy_model = IDQLDiffusion(d4rl_dataset.a_dim, d4rl_dataset.a_dim, d4rl_dataset.s_dim, time_embeding=args.time_embed, device=device).to(device)
       v_model = MLPV(d4rl_dataset.s_dim, 1)
       qs_model = MLPQs(d4rl_dataset.s_dim + d4rl_dataset.a_dim, 1, ensemble_num=2)
       
       ## feed v, q and policy into IDQL agent
       idql_agent = IDQL_Agent(policy_model, v_model, qs_model, schedule=args.beta, num_timesteps=args.T, num_sample=args.num_samples, expectile=args.expectile).to(device)
       
       # logger
       wandb_logger = WandbLogger(project_name=args.project_name, run_name=env_name, args=args)

       # evaluator
       evaluator = D4RLEval(env_name, d4rl_dataset.data_statistics, wandb_logger, 50, 250000, seed=seed)

       # trainer
       test_trainer = RLTrainer(idql_agent, d4rl_dataloader, d4rl_dataloader, wandb_logger, evaluator, int(args.steps), lr=args.lr, device=device)
       test_trainer.train_steps()

if __name__ == '__main__':
       main()
import torch
import numpy as np
from torch.utils.data import DataLoader

from BearRobot.Net.my_model.critic_model import MLPV, MLPQs
from BearRobot.Net.my_model.MLPpolicy import MLPPi
from BearRobot.Agent.TD3 import TD3_Agent

from BearRobot.utils.dataset.onlineRL_dataloader import OnlineReplaybuffer
from BearRobot.Trainer.online_trainer import RLTrainer

from BearRobot.utils.evaluation.d4rl_eval import D4RLEval
from BearRobot.utils.logger.wandb_log import WandbLogger

import gym

import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
       # customize your argparser
       parser = argparse.ArgumentParser(description='An example to use this model')
       parser.add_argument('--device', default='cuda', help='cuda or cpu')
       parser.add_argument('--project_name', default='TD3_pytorch_example', help='your project name')
       parser.add_argument('--env_name', default='hopper-medium-expert-v2', help='choose your mujoco env')
       parser.add_argument('--steps', default=1e+6, type=float, help='train steps')
       parser.add_argument("--seed", default=42, type=int)  # Sets Gym, PyTorch and Numpy seeds
       parser.add_argument('--batch_size', default=256, type=int)
       parser.add_argument('--num_samples', default=64, type=int, help='evaluation sample action nums')
       parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
       
       args = parser.parse_args()

       # seed
       seed = args.seed
       np.random.seed(seed)
       torch.manual_seed(seed)
       
       # dataset and dataloader
       env_name = args.env_name
       env = gym.make(env_name)
       replay_buffer = OnlineReplaybuffer(env_name, max_size=int(args.steps), batch_size=args.batch_size)

       # agent and the model for agent
       policy_model = MLPPi(replay_buffer.s_dim, replay_buffer.a_dim, device=device).to(device)
       qs_model = MLPQs(replay_buffer.s_dim + replay_buffer.a_dim, 1, ensemble_num=2)
       
       ## feed q and policy into td3 agent
       td3_agent = TD3_Agent(policy_model, None, qs_model).to(device)
       
       # logger
       wandb_logger = WandbLogger(project_name=args.project_name, run_name=env_name, args=args)

       # evaluator
       evaluator = D4RLEval(env_name, replay_buffer.data_statistics, wandb_logger, 10, 5000, seed=seed)

       # trainer
       test_trainer = RLTrainer(td3_agent, replay_buffer, env, wandb_logger, evaluator, int(args.steps), lr=args.lr, device=device)
       test_trainer.train_steps()

if __name__ == '__main__':
       main()
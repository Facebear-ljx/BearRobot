import torch
import numpy as np
from torch.utils.data import DataLoader

from Net.my_model.RT_model import RT1Model
from Agent.RT_agent import RT1Agent

from utils.dataset.dataloader import RT1Dataset
from Trainer.trainer import BCTrainer

from utils.logger.wandb_log import WandbLogger

import argparse
from utils.net.initialization import boolean

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
       # customize your argparser
       parser = argparse.ArgumentParser(description='An example to use this model')
       parser.add_argument('--device', default='cuda', help='cuda or cpu')
       parser.add_argument('--project_name', default='RT1_pytorch_example', help='your project name')
       parser.add_argument('--dataset_name', default='bridge', help='choose your mujoco env')
       parser.add_argument('--img_size', default=128, type=int, help='image size')
       parser.add_argument('--frames', default=3, type=int, help='frames num input to RT1')
       parser.add_argument('--visual_pretrain', default=True, type=boolean, help='whether use visual pretrain')
       parser.add_argument('--steps', default=1000, type=float, help='train steps')
       parser.add_argument("--seed", default=42, type=int)  # Sets PyTorch and Numpy seeds
       parser.add_argument('--batch_size', default=64, type=int)
       parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
       
       args = parser.parse_args()

       # seed
       seed = args.seed
       np.random.seed(seed)
       torch.manual_seed(seed)
       
       # dataset and dataloader
       rt1dataset = RT1Dataset(frames=args.frames)
       rt1dataloader = DataLoader(rt1dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

       # agent and the model for agent
       rt1model = RT1Model(img_size=args.img_size, device=device, vision_pretrain=args.visual_pretrain).to(device)
       rt1agent = RT1Agent(rt1model)
       
       # logger
       wandb_logger = WandbLogger(project_name=args.project_name, run_name=args.dataset_name, args=args)

       # trainer
       test_trainer = BCTrainer(rt1agent, rt1dataloader, rt1dataloader, wandb_logger, None, num_steps=int(args.steps), lr=args.lr, device=device)
       test_trainer.train_steps()

if __name__ == '__main__':
       main()
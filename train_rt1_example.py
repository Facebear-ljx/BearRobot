import random
import torch
import numpy as np
from torch.utils.data import DataLoader

from Net.my_model.RT_model import RT1Model
from Agent.RT_agent import RT1Agent

from utils.dataset.dataloader import RT1DataLoader
from utils.logger.wandb_log import WandbLogger
from utils.net.initialization import boolean
from utils import ddp
from Trainer.trainer import BCTrainer
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.multiprocessing as mp
import argparse


def get_args():
       # customize your argparser
       parser = argparse.ArgumentParser(description='An example to use this model')
       parser.add_argument('--device', default='cuda', help='cuda or cpu')
       parser.add_argument('--project_name', default='RT1_pytorch_example', help='your project name')
       parser.add_argument('--dataset_name', default='bridge', help='choose your mujoco env')
       parser.add_argument('--img_size', default=128, type=int, help='image size')
       parser.add_argument('--frames', default=3, type=int, help='frames num input to RT1')
       parser.add_argument('--visual_pretrain', default=True, type=boolean, help='whether use visual pretrain')
       parser.add_argument('--steps', default=10000, type=float, help='train steps')
       parser.add_argument("--seed", default=42, type=int)  # Sets PyTorch and Numpy seeds
       parser.add_argument('--batch_size', default=128, type=int)
       parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')

       # DataLoader parameters
       parser.add_argument('--num_workers', default=8, type=int)
       parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
       parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
       parser.set_defaults(pin_mem=True)
       
       # distributed training parameters
       parser.add_argument('--world_size', default=2, type=int, help='number of distributed processes')
       parser.add_argument('--port', default=22323, type=int, help='port')
       args = parser.parse_args()    
       return args   


def main(rank: int, world_size: int, args):
       wandb_logger = WandbLogger(project_name=args.project_name, run_name=args.dataset_name, args=args, rank=rank) 
       
       # init ddp
       ddp.ddp_setup(rank, world_size, args)
       
       # dataset and dataloader
       rt1dataloader = RT1DataLoader(
              frames=args.frames,
              batch_size=args.batch_size, 
              num_workers=args.num_workers,
              pin_mem=args.pin_mem,
       )

       # agent and the model for agent
       rt1model = RT1Model(img_size=args.img_size, device=rank, vision_pretrain=args.visual_pretrain).to(rank)
       rt1agent = RT1Agent(rt1model)      

       # trainer
       test_trainer = BCTrainer(rt1agent, rt1dataloader, rt1dataloader, wandb_logger, None, num_steps=int(args.steps), lr=args.lr, device=rank, args=args)
       test_trainer.train_steps()


if __name__ == '__main__':
       # get log
       args = get_args()
       device = torch.device(args.device)
       
       # seed
       seed = args.seed + ddp.get_rank()
       np.random.seed(seed)
       torch.manual_seed(seed)
       random.seed(seed)
       
       mp.spawn(main, args=(args.world_size, args), nprocs=args.world_size)
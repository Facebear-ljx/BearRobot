import os
from datetime import datetime

import random
import torch
import numpy as np
from torch.utils.data import DataLoader

from BearRobot.Net.my_model.RT_model import RT1Model
from BearRobot.Agent.RT_agent import RT1Agent

from BearRobot.utils.dataset.dataloader import RT1DataLoader, RT1ValDataLoader, AIRKitchenDataLoader, AIRKitchenValDataLoader
from BearRobot.utils.logger.tb_log import TensorBoardLogger as Logger
from BearRobot.utils.net.initialization import boolean
from BearRobot.utils import ddp
from BearRobot.Trainer.trainer import BCTrainer
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
       parser.add_argument('--steps', default=int(1e+6), type=float, help='train steps')
       parser.add_argument('--val_freq', default=int(5e+3), type=float, help='val frequency')
       parser.add_argument("--seed", default=42, type=int)  # Sets PyTorch and Numpy seeds
       parser.add_argument('--batch_size', default=128, type=int)
       parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
       
       # save log parameters
       parser.add_argument('--save', default=True, type=boolean, help='save ckpt or not')
       parser.add_argument('--save_path', default='experiments/rt1', type=str, help='ckpt save path')
       parser.add_argument('--save_freq', default=int(1e+4), type=int, help='save ckpt frequency')
       parser.add_argument('--log_path', default='experiments/rt1', type=str, help='ckpt save path')
       parser.add_argument('--resume', default=None, type=str, help='resume path')
       parser.add_argument('--wandb', default=False, type=boolean, help='use wandb or not')
       
       # DataLoader parameters
       parser.add_argument('--num_workers', default=8, type=int)
       parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
       parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
       parser.set_defaults(pin_mem=True)
       
       # distributed training parameters
       parser.add_argument('--ddp', default=False, type=boolean, help='use ddp or not')
       parser.add_argument('--world_size', default=3, type=int, help='number of distributed processes')
       parser.add_argument('--port', default='22323', type=str, help='port')
       args = parser.parse_args()    
       return args   


def main(args):
       # seed
       seed = args.seed + ddp.get_rank()
       np.random.seed(seed)
       torch.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
       random.seed(seed)
       torch.backends.cudnn.deterministic = True
       
       # init ddp
       global_rank, rank, _ = ddp.ddp_setup_universal(True, args)
       
       # save 
       if args.save and global_rank==0:
              # your ckpt save path
              save_path = args.save_path
              if not os.path.exists(save_path):
                     os.makedirs(save_path)
       else:
              save_path = None

       # logger
       wandb_logger = Logger(args.project_name, args.dataset_name, args, save_path=args.log_path, rank=global_rank) 

       # dataset and dataloader
       view_list = ['D435_image', 'wrist_image']
       rt1dataloader, statistics = AIRKitchenDataLoader(
              base_dir='',
              datalist='/home/dodo/ljx/BearRobot/data/airkitchen/AIR-toykitchen-ac-blur.json',
              view_list=view_list,
              img_size=args.img_size,
              frames=args.frames,
              discretize_actions=True,
              batch_size=args.batch_size, 
              num_workers=args.num_workers,
              pin_mem=args.pin_mem,
              ac_num=1,
       )
       
       val_g_dataloader = AIRKitchenValDataLoader(
              datalist='/home/dodo/ljx/BearRobot/data/airkitchen/AIR-toykitchen-ac_machine_g.json',
              view_list=view_list,
              img_size=args.img_size,
              frames=args.frames,
              discretize_actions=True,
              batch_size=args.batch_size, 
              num_workers=args.num_workers,
              pin_mem=args.pin_mem,
              ac_num=1,
              statistics=statistics              
       )

       # agent and the model for agent
       rt1model = RT1Model(img_size=args.img_size, device=rank, vision_pretrain=args.visual_pretrain).to(rank)
       rt1agent = RT1Agent(rt1model)      

       # trainer
       test_trainer = BCTrainer(rt1agent, 
                                rt1dataloader, 
                                val_g_dataloader, 
                                wandb_logger, 
                                None, 
                                num_steps=int(args.steps), 
                                lr=args.lr, 
                                device=rank,
                                global_rank=global_rank,
                                save=args.save,
                                save_freq=args.save_freq, 
                                save_path=save_path,
                                resume_path=args.resume,
                                args=args
                     )
       test_trainer.train_steps()


if __name__ == '__main__':
       # get log
       args = get_args()
       main(args)
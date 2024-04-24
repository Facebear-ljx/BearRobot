import os
from datetime import datetime

import random
import torch
import numpy as np

from Net.my_model.ACT_model import ACTModel
from Agent.ACT import ACTAgent

from utils.dataset.dataloader import AIRKitchenDataLoader, AIRKitchenValDataLoader
from utils.logger.tb_log import TensorBoardLogger as Logger
from utils.net.initialization import boolean
from utils import ddp
from Trainer.trainer import BCTrainer
from config.basic_args import basic_args, diffusion_args
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.multiprocessing as mp
import argparse
import json


def get_args():
       parser = basic_args()
       # customize your argparser
       parser.add_argument('--img_size', default=0, type=int, help='image size, 0 means use the default size')
       parser.add_argument('--frames', default=1, type=int, help='frames num input to the visual encoder')
       parser.add_argument('--visual_encoder', default='resnet18', type=str, help='visual encoder backbone, support resnet 18/34/50')
       parser.add_argument('--visual_pretrain', default=True, type=boolean, help='whether use visual pretrain')
       parser.add_argument('--ft_vision', default=True, type=boolean, help='whether tune the visual encoder')
       
       parser.add_argument('--ac_num', default=4, type=int, help='action trunking number')
       parser.add_argument('--norm', default="minmax", type=str, help='whether norm the action or not')
       parser.add_argument('--discretize_actions', default=False, type=boolean, help='whether discretize_actions the action or not')
       
       parser.add_argument('--hidden_dim', default=512, type=int, help='hidden dim for decoder MLP')
       
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
              discretize_actions=args.discretize_actions,
              norm=args.norm,
              batch_size=args.batch_size, 
              num_workers=args.num_workers,
              pin_mem=args.pin_mem,
              ac_num=4,
       )
       
       val_g_dataloader = AIRKitchenValDataLoader(
              datalist='/home/dodo/ljx/BearRobot/data/airkitchen/AIR-toykitchen-ac_machine_g.json',
              view_list=view_list,
              img_size=args.img_size,
              frames=args.frames,
              discretize_actions=args.discretize_actions,
              norm=args.norm,
              batch_size=args.batch_size, 
              num_workers=args.num_workers,
              pin_mem=args.pin_mem,
              ac_num=4,
              statistics=statistics              
       )

       val_b_dataloader = AIRKitchenValDataLoader(
              datalist='/home/dodo/ljx/BearRobot/data/airkitchen/AIR-toykitchen-ac_machine_b.json',
              view_list=view_list,
              img_size=args.img_size,
              frames=args.frames,
              discretize_actions=args.discretize_actions,
              norm=args.norm,
              batch_size=args.batch_size, 
              num_workers=args.num_workers,
              pin_mem=args.pin_mem,
              ac_num=4,
              statistics=statistics              
       )


       with open(os.path.join(args.save_path, 'statistics.json'), 'w') as f:
              json.dump(statistics, f)

       # agent and the model for agent
       policy_model = ACTModel(output_dim=7,
                               ac_num=args.ac_num,
                               hidden_dim=args.hidden_dim,
                               vision_encoder=args.visual_encoder,
                               vision_pretrained=args.visual_pretrain,
                               ft_vision=args.ft_vision,
                               num_encoder_layers=6,
                               num_decoder_layers=6,
                               device=rank).to(rank)
       agent = ACTAgent(policy=policy_model)      

       # trainer
       test_trainer = BCTrainer(agent, 
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
import os
from datetime import datetime

import random
import torch
import numpy as np

from Net.my_model.diffusion_model import VisualDiffusion
from Agent.ddpm_bc import VLDDPM_BC

from utils.dataset.dataloader import RT1DataLoader, RT1ValDataLoader, AIRKitchenDataLoader
from utils.logger.tb_log import TensorBoardLogger as Logger
from utils.net.initialization import boolean
from utils import ddp
from Trainer.trainer import BCTrainer
from config.basic_args import basic_args, diffusion_args
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.multiprocessing as mp
import argparse


def get_args():
       parser = basic_args()
       # customize your argparser
       parser.add_argument('--img_size', default=224, type=int, help='image size')
       parser.add_argument('--frames', default=1, type=int, help='frames num input to the visual encoder')
       parser.add_argument('--visual_encoder', default='resnet34', type=str, help='visual encoder backbone, support resnet 18/34/50')
       parser.add_argument('--visual_pretrain', default=False, type=boolean, help='whether use visual pretrain')
       parser.add_argument('--ft_vision', default=False, type=boolean, help='whether tune the visual encoder')
       
       parser.add_argument('--ac_num', default=4, type=int, help='action trunking number')
       parser.add_argument('--norm', default="minmax", type=str, help='whether norm the action or not')
       parser.add_argument('--discretize_actions', default=False, type=boolean, help='whether discretize_actions the action or not')
       
       parser.add_argument('--num_blocks', default=3, type=int, help='num blocks for decoder MLP')
       parser.add_argument('--hidden_dim', default=256, type=int, help='hidden dim for decoder MLP')
       parser.add_argument('--norm_type', default="bn", type=str, help='normalization type')
       parser.add_argument('--pooling_type', default="avg", type=str, help='pooling type')
       parser.add_argument('--add_spatial_coordinates', default=False, type=boolean, help='add spatial coordinates to the image')
       
       parser = diffusion_args(parser)
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
       rt1dataloader = AIRKitchenDataLoader(
              base_dir='',
              datalist='/home/dodo/ljx/BearRobot/data/airkitchen/AIR-toykitchen-ac.json',
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

       # agent and the model for agent
       visual_diffusion_policy = VisualDiffusion(img_size=args.img_size,
                                                 view_num=len(view_list), 
                                                 output_dim=int(7 * args.ac_num),
                                                 num_blocks=args.num_blocks,
                                                 hidden_dim=args.hidden_dim,
                                                 time_embeding=args.time_embed,
                                                 time_dim=args.time_dim,
                                                 time_hidden_dim=args.time_hidden_dim,
                                                 vision_encoder=args.visual_encoder,
                                                 vision_pretrained=args.visual_pretrain,
                                                 ft_vision=args.ft_vision,
                                                 norm_type=args.norm_type,
                                                 pooling_type=args.pooling_type,
                                                 add_spatial_coordinates=args.add_spatial_coordinates,
                                                 device=rank).to(rank)
       agent = VLDDPM_BC(policy=visual_diffusion_policy,
                         schedule=args.beta,
                         num_timesteps=args.T,
                         text_encoder="t5")      

       # trainer
       test_trainer = BCTrainer(agent, 
                                rt1dataloader, 
                                rt1dataloader, 
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
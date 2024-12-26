import os
from datetime import datetime

import random
import torch
import numpy as np

from BearRobot.Net.my_model.diffusion_model import VisualDiffusion
from BearRobot.Agent.ddpm_bc import VLDDPM_BC

from BearRobot.utils.dataset.dataloader import RT1DataLoader, RT1ValDataLoader, AIRKitchenDataLoader, AIRKitchenValDataLoader
from BearRobot.utils.logger.tb_log import TensorBoardLogger as Logger
from BearRobot.utils.net.initialization import boolean
from BearRobot.utils import ddp
from BearRobot.Trainer.trainer import BCTrainer
from BearRobot.config.basic_args import basic_args, diffusion_args
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.multiprocessing as mp
import argparse
import json


def get_args():
       parser = basic_args()
       # customize your argparser
       parser.add_argument('--img_size', default=224, type=int, help='image size')
       parser.add_argument('--frames', default=1, type=int, help='frames num input to the visual encoder')
       parser.add_argument('--visual_encoder', default='resnet34', type=str, help='visual encoder backbone, support resnet 18/34/50')
       parser.add_argument('--visual_pretrain', default=True, type=boolean, help='whether use visual pretrain')
       parser.add_argument('--ft_vision', default=True, type=boolean, help='whether tune the visual encoder')
       
       parser.add_argument('--ac_num', default=4, type=int, help='action trunking number')
       parser.add_argument('--norm', default="minmax", type=str, help='whether norm the action or not')
       parser.add_argument('--discretize_actions', default=False, type=boolean, help='whether discretize_actions the action or not')
       parser.add_argument('--s_dim', default=9, type=int, help='qpos dim, 0 means dont use qpos')
       parser.add_argument('--encode_s', default=False, type=boolean, help='whether encode the state (qpos) or not')
       parser.add_argument('--encode_a', default=False, type=boolean, help='whether encode the action or not')
       
       parser.add_argument('--num_blocks', default=3, type=int, help='num blocks for decoder MLP')
       parser.add_argument('--hidden_dim', default=256, type=int, help='hidden dim for decoder MLP')
       parser.add_argument('--norm_type', default="bn", type=str, help='normalization type')
       parser.add_argument('--pooling_type', default="avg", type=str, help='pooling type')
       parser.add_argument('--add_spatial_coordinates', default=False, type=boolean, help='add spatial coordinates to the image')
       parser.add_argument('--film_fusion', default=False, type=boolean, help='add film condition to the decoder')
       
       parser.add_argument('--view_list', default=['top_image', 'wrist_image', 'side_image'], nargs='+', help='image view list')
       parser.add_argument('--datalist', default=None, nargs='+', help='datalist path')
       
       parser = diffusion_args(parser)
       args = parser.parse_args()    
       return args   


def main(args):
       kwargs = vars(args)
       # seed
       seed = args.seed + ddp.get_rank()
       np.random.seed(seed)
       torch.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
       random.seed(seed)
       torch.backends.cudnn.deterministic = True
       
       # init ddp
       global_rank, rank, _ = ddp.ddp_setup_universal(True, args)
       kwargs['device'] = rank

       import torchvision.transforms as T
       from PIL import Image
       transform_list  = [
              T.RandomResizedCrop(args.img_size, scale=(0.75, 1), interpolation=Image.BICUBIC),
              T.ColorJitter(0.2, [0.8, 1.2], [0.8, 1.2], 0.1),
              T.ToTensor(),
              T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       ]

       # dataset and dataloader
       print("-"*88)
       print("image_view:", kwargs['view_list'])
       print("datalist_path:", kwargs['datalist'])
       dataloader, statistics = AIRKitchenDataLoader(
              base_dir='',
              transform_list=transform_list,
              **kwargs
       )

       # save 
       if args.save and global_rank==0:
              # your ckpt save path
              save_path = args.save_path
              if not os.path.exists(save_path):
                     os.makedirs(save_path)
                     print("save path {} created".format(save_path))
              
              # save the statistics for training dataset
              with open(os.path.join(save_path, 'statistics.json'), 'w') as f:
                     json.dump(statistics, f)
                     print("statistics saved to {}".format(os.path.join(save_path, 'statistics.json')))
       else:
              save_path = None

        # logger
       wandb_logger = Logger(args.project_name, args.dataset_name, args, save_path=args.log_path, rank=global_rank) 
       print("-"*88)
       
       print("(TRAINING): INITIALIZE AGENT")
       
       # agent and the model for agent
       visual_diffusion_policy = VisualDiffusion(view_num=len(kwargs['view_list']), 
                                                 output_dim=int(7 * args.ac_num),
                                                 **kwargs).to(rank)
       agent = VLDDPM_BC(policy=visual_diffusion_policy, **kwargs)      
       agent.get_statistics(os.path.join(args.save_path, 'statistics.json'))
       agent.get_transform(img_size=0, transform_list=transform_list)
       print("(TRAINING): AGENT INITIALIZED DONE")
       # trainer
       test_trainer = BCTrainer(agent, 
                                dataloader, 
                                dataloader, 
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
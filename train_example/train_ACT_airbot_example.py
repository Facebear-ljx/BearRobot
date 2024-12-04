import os
from datetime import datetime

import random
import torch
import numpy as np

from BearRobot.Net.my_model.ACT_model import ACTModel
from BearRobot.Agent.ACT import ACTAgent

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
       parser.add_argument('--visual_encoder', default='resnet18', type=str, help='visual encoder backbone, support resnet 18/34/50')
       parser.add_argument('--visual_pretrain', default=True, type=boolean, help='whether use visual pretrain')
       parser.add_argument('--ft_vision', default=True, type=boolean, help='whether tune the visual encoder')
       
       parser.add_argument('--ac_num', default=4, type=int, help='action trunking number')
       parser.add_argument('--norm', default="mean", type=str, help='whether norm the action or not')
       parser.add_argument('--discretize_actions', default=False, type=boolean, help='whether discretize_actions the action or not')
       parser.add_argument('--s_dim', default=7, type=int, help='qpos dim, 0 means dont use qpos')
       
       parser.add_argument('--hidden_dim', default=512, type=int, help='hidden dim for transformer hidden state')
       parser.add_argument('--dim_feedforward', default=2048, type=int, help='hidden dim transformer layer')
       parser.add_argument('--num_encoder_layers', default=6, type=int, help='number of transformer encoder layers')
       parser.add_argument('--num_decoder_layers', default=6, type=int, help='number of transformer decoder layers')
       parser.add_argument('--kl_weight', default=10, type=float, help='KL-divergence weight')    
       parser.add_argument('--loss_type', default='huber', type=str, help='action loss, l1, l2, huber')   

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
       view_list = ['top_image', 'wrist_image', 'side_image']
       dataloader, statistics = AIRKitchenDataLoader(
              base_dir='',
              datalist=['/home/dodo/ljx/BearRobot/data/airbot/newair_rel_eef_1203.json'],
              view_list=view_list,
              transform_list=transform_list,
              **kwargs
       )

       # save 
       if args.save and global_rank==0:
              # your ckpt save path
              save_path = args.save_path
              if not os.path.exists(save_path):
                     os.makedirs(save_path)
              
              # save the statistics for training dataset
              with open(os.path.join(save_path, 'statistics.json'), 'w') as f:
                     json.dump(statistics, f)
       else:
              save_path = None

        # logger
       wandb_logger = Logger(args.project_name, args.dataset_name, args, save_path=args.log_path, rank=global_rank) 

       # agent and the model for agent
       model = ACTModel(output_dim=7,
                     **kwargs).to(rank)
       agent = ACTAgent(policy=model, **kwargs)
       agent.get_statistics(os.path.join(args.save_path, 'statistics.json'))
       agent.get_transform(img_size=args.img_size, transform_list=transform_list)

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
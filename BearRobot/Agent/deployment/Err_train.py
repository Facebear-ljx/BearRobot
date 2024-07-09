import os
from datetime import datetime

import sys 
import random
import torch
import numpy as np

from Err_ddpm_bc import VLDDPM_BC, IDQL_Agent
from Err_dataloader import AIRKitchenDataLoader_err
from Err_trainer import RLTrainer

from BearRobot.Net.my_model.diffusion_model import VisualDiffusion
from BearRobot.utils.logger.tb_log import TensorBoardLogger as Logger
from BearRobot.utils.net.initialization import boolean
from BearRobot.utils import ddp
from BearRobot.utils.evaluation.mp_libero_eval import LIBEROEval
from BearRobot.config.basic_args import basic_args, diffusion_args

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import argparse
import json

def get_args():
       parser = basic_args()
       # customize your argparser
       parser.add_argument('--img_size', default=128, type=int, help='image size')
       parser.add_argument('--frames', default=1, type=int, help='frames num input to the visual encoder')
       parser.add_argument('--visual_encoder', default='resnet34', type=str, help='visual encoder backbone, support resnet 18/34/50')
       parser.add_argument('--visual_pretrain', default=False, type=boolean, help='whether use visual pretrain')
       parser.add_argument('--ft_vision', default=False, type=boolean, help='whether tune the visual encoder')
       parser.add_argument('--text_encoder', default='DecisionNCE-T', type=str, help='language encoder, support T5, DecisionNCE-T, DecisionNCE-P')
       
       parser.add_argument('--ac_num', default=4, type=int, help='action trunking number')
       parser.add_argument('--norm', default="minmax", type=str, help='whether norm the action or not')
       parser.add_argument('--discretize_actions', default=False, type=boolean, help='whether discretize_actions the action or not')
       parser.add_argument('--s_dim', default=9, type=int, help='qpos dim, 0 means dont use qpos')# remember to change the s_dim
       parser.add_argument('--encode_s', default=False, type=boolean, help='whether encode the state (qpos) or not')
       parser.add_argument('--encode_a', default=False, type=boolean, help='whether encode the action or not')
       
       parser.add_argument('--num_blocks', default=3, type=int, help='num blocks for decoder MLP')
       parser.add_argument('--hidden_dim', default=256, type=int, help='hidden dim for decoder MLP')
       parser.add_argument('--norm_type', default="bn", type=str, help='normalization type')
       parser.add_argument('--pooling_type', default="avg", type=str, help='pooling type')
       parser.add_argument('--add_spatial_coordinates', default=False, type=boolean, help='add spatial coordinates to the image')
       parser.add_argument('--film_fusion', default=False, type=boolean, help='add film condition to the decoder')

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

    # dataset and dataloader
    view_list = ['D435_image', 'wrist_image']

    img_goal = True if kwargs['text_encoder'] == 'DecisionNCE-V' else False
    rt1dataloader, statistics = AIRKitchenDataLoader_err(
        base_dir='/home/dodo/ljx/BearRobot/data/libero/dataset/',
        datalist=['/home/dodo/ljx/BearRobot/data/libero/libero_goal-ac.json'],
        view_list=view_list,
        img_goal=img_goal,
        **kwargs
    )

    # save 
    if args.save and global_rank == 0:
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
    visual_diffusion_policy = VisualDiffusion(
        view_num=len(view_list),
        output_dim=int(7 * args.ac_num),
        **kwargs
    ).to(rank)
    agent = IDQL_Agent(policy_model=visual_diffusion_policy, **kwargs)
    agent.get_statistics(os.path.join(args.save_path, 'statistics.json'))
    agent.get_transform(img_size=0)

    # trainer
    test_trainer = RLTrainer(
        agent=agent,
        train_dataloader=rt1dataloader,
        val_dataloader=None,  # Add the validation dataloader if you have one
        logger=wandb_logger,
        evaluator=None,  # Add the evaluator if you have one
        num_steps=int(args.steps),
        lr=1e-4,
        policy_ema=1e-3,
        critic_ema=5e-3,
        optimizer='adam',
        device="cuda",
        save=args.save,
        save_freq=args.save_freq, 
        save_path=save_path,
        resume_path=args.resume,
    )
    test_trainer.train_steps()

if __name__ == '__main__':
    args = get_args()
    main(args)

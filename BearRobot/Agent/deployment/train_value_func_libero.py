import os
from datetime import datetime

import sys 
import random
import torch
import numpy as np

from BearRobot.Net.my_model.diffusion_model import VisualDiffusion
from Err_ddpm_bc import VLDDPM_BC,IDQL_Agent

from BearRobot.utils.dataset.dataloader import AIRKitchenDataLoader_err
from BearRobot.utils.logger.tb_log import TensorBoardLogger as Logger
from BearRobot.utils.net.initialization import boolean
from BearRobot.utils import ddp
from BearRobot.utils.evaluation.mp_libero_eval import LIBEROEval
from Err_trainer import RLTrainer
from BearRobot.config.basic_args import basic_args, diffusion_args
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.multiprocessing as mp
import argparse
import json


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
    agent = IDQL_Agent(policy=visual_diffusion_policy, K=60 , **kwargs)
    agent.get_statistics(os.path.join(args.save_path, 'statistics.json'))
    agent.get_transform(img_size=0)

    # trainer
    test_trainer = RLTrainer(
        agent=agent,
        train_dataloader=rt1dataloader,
        val_dataloader=None,  # Add the validation dataloader if you have one
        logger=wandb_logger,
        evaluator=None,  # Add the evaluator if you have one
        num_steps=args.num_steps,
        lr=1e-4,
        policy_ema=1e-3,
        critic_ema=5e-3,
        optimizer='adam',
        device="cuda",
    )
    test_trainer.train_steps()

if __name__ == '__main__':
    # get log
    args = get_args()
    main(args)

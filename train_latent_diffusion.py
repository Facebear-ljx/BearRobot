import random
import torch
import numpy as np
from torch.utils.data import DataLoader

from Net.my_model.diffusion_model import IDQLDiffusion
from Net.encoder.DecisionNCE import DecisionNCE_encoder
from Agent.ddpm_bc import DDPM_BC_latent

from utils.dataset.dataloader import VideoPredictDataLoader
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
       parser.add_argument('--project_name', default='latent_diffusion_pytorch_example', help='your project name')
       parser.add_argument('--dataset_name', default='all', help='choose your mujoco env')
       parser.add_argument('--img_size', default=128, type=int, help='image size')
       parser.add_argument('--frames', default=3, type=int, help='history frames num')
       parser.add_argument('--skip_frame', default=5, type=int, help='skip number')
       parser.add_argument('--visual_pretrain', default=True, type=boolean, help='whether use visual pretrain')
       parser.add_argument('--steps', default=int(1e+6), type=float, help='train steps')
       parser.add_argument("--seed", default=42, type=int)  # Sets PyTorch and Numpy seeds
       parser.add_argument('--batch_size', default=512, type=int)
       parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
       parser.add_argument('--save', default=True, type=boolean, help='save ckpt or not')
       parser.add_argument('--save_freq', default=int(1e+4), type=int, help='save ckpt frequency')
       parser.add_argument('--resume', default="None", type=str, help='resume path')

       # diffusion model
       parser.add_argument('--num_blocks', default=6, type=int, help='image size')
       parser.add_argument('--hidden_dim', default=1024, type=int, help='image size')
       parser.add_argument('--T', default=100, type=int, help='maximize diffusion time steps')
       parser.add_argument('--time_embed', default='learned', type=str, help='learned or fixed type time embedding')
       parser.add_argument('--beta', default='vp', type=str, help='noise schedule')

       # encoder type
       parser.add_argument("--encoder", default="DecisionNCE-T", type=str, help="choose your encoder")
       
       # DataLoader parameters
       parser.add_argument('--num_workers', default=8, type=int)
       parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
       parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
       parser.set_defaults(pin_mem=True)
       
       # distributed training parameters
       parser.add_argument('--world_size', default=4, type=int, help='number of distributed processes')
       parser.add_argument('--port', default="11111", type=str, help='port for ddp')
       args = parser.parse_args()    
       return args   


def main(rank: int, world_size: int, save_path: str, args):
       # wandb logger
       wandb_logger = WandbLogger(project_name=args.project_name, run_name=args.dataset_name, args=args, rank=rank) 
       
       # init ddp
       ddp.ddp_setup(rank, world_size, True, args.port)
       
       # dataset and dataloader
       vpdataloader = VideoPredictDataLoader(
              frames=args.frames,
              skip_frame=args.skip_frame,
              batch_size=args.batch_size, 
              num_workers=args.num_workers,
              pin_mem=args.pin_mem,
       )
       
       # agent and the model for agent
       encoder = DecisionNCE_encoder(args.encoder, device=rank)
       cond_dim = encoder.v_dim * args.frames + encoder.l_dim
       input_dim = encoder.v_dim
       output_dim = encoder.v_dim
       vpmodel = IDQLDiffusion(input_dim, 
                               output_dim, 
                               cond_dim, 
                               time_dim=args.hidden_dim,
                               time_embeding=args.time_embed, 
                               num_blocks=args.num_blocks,
                               hidden_dim=args.hidden_dim,
                               device=rank).to(rank)
       vpagent = DDPM_BC_latent(vpmodel, schedule=args.beta, num_timesteps=args.T, mmencoder=encoder)      

       # # trainer
       test_trainer = BCTrainer(vpagent, 
                                vpdataloader, 
                                vpdataloader, 
                                wandb_logger, 
                                None, 
                                num_steps=int(args.steps), 
                                lr=args.lr, 
                                device=rank,
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
       device = torch.device(args.device)
       
       # your ckpt save path
       import os
       from datetime import datetime

       previous_dir = os.getcwd()
       base_dir = "experiments"
       project_name = args.project_name
       data_name = args.dataset_name
       time = datetime.now()
       time = time.strftime("%Y-%m-%d %H:%M:%S")
       save_path = f"{previous_dir}/{base_dir}/{project_name}/{data_name}/{time}"
       if not os.path.exists(save_path):
              os.makedirs(save_path)
       
       # seed
       seed = args.seed + ddp.get_rank()
       np.random.seed(seed)
       torch.manual_seed(seed)
       random.seed(seed)

       mp.spawn(main, args=(args.world_size, save_path, args), nprocs=args.world_size)
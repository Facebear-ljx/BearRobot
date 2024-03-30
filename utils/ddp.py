import os
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group



def ddp_setup(rank: int, world_size: int, verbose: bool=False, port: str="12355"):
       """
       Args:
       rank: Unique identifier of each process
       world_size: Total number of processes
       """
       os.environ["MASTER_ADDR"] = "localhost"
       os.environ["MASTER_PORT"] = port
       torch.cuda.set_device(rank)
       init_process_group(backend="nccl", rank=rank, world_size=world_size)
       torch.distributed.barrier()
       if verbose:
              setup_for_distributed(rank == 0)
       print(f"{rank}"*88)


def init_distributed_mode(args, verbose=False):
       print('1'*88)
       if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
              args.rank = int(os.environ["RANK"])  # global gpu id
              args.world_size = int(os.environ['WORLD_SIZE']) # process num
              args.gpu = int(os.environ['LOCAL_RANK']) # 0~7 local gpu id
              os.environ['MASTER_PORT'] = "12355"
              os.environ['MASTER_ADDR'] = 'localhost'
              print("2"*88)
       else:
              print('Not using distributed mode')
              args.distributed = False
              return
       print("3"*88)
       args.distributed = True

       torch.cuda.set_device(args.rank)
       args.dist_backend = 'nccl'
       print('| distributed init (rank {}): {}, gpu {}'.format(
              args.rank, args.dist_url, args.gpu), flush=True)
       init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.rank)
       print("4"*88)
       torch.distributed.barrier()
       if verbose:
              setup_for_distributed(args.rank == 0)
       print("5"*88)
              

def setup_for_distributed(is_master):
       """
       This function disables printing when not in master process
       """
       import builtins as __builtin__
       builtin_print = __builtin__.print

       def print(*args, **kwargs):
              force = kwargs.pop('force', False)
              if is_master or force:
                     builtin_print(*args, **kwargs)

       __builtin__.print = print


def is_dist_avail_and_initialized():
       if not dist.is_available():
              return False
       if not dist.is_initialized():
              return False
       return True

   
def get_world_size():
       if not is_dist_avail_and_initialized():
              return 1
       return dist.get_world_size()


def get_rank():
       if not is_dist_avail_and_initialized():
              return 0
       return dist.get_rank()


def is_main_process():
       return get_rank() == 0
from Err_eval_lib import LIBEROEval_err, build_visual_diffsuion_err
from BearRobot.utils.logger.tb_log import TensorBoardLogger
from BearRobot.config.basic_args import basic_args, diffusion_args
from BearRobot.utils.net.initialization import boolean
import argparse


def get_args():
        parser = argparse.ArgumentParser(description='args for err evaluation')
        
        # crucial eval parameters
        parser.add_argument('--k', default=0.2, type=float, help='number of steps for err recognition')
        parser.add_argument('--ckpt_path', default='', type=str, help='path to the checkpoint')
        parser.add_argument('--statistic_path', default='BearRobot/Agent/deployment/experiments/libero/libero_goal/trial_err_03/latest.pth', type=str, help='path to the statistics')
        parser.add_argument('--num_episodes', default=10, type=int, help='number of episodes to evaluate')
        parser.add_argument('--eval_horizon', default=300, type=int, help='horizon for evaluation')
            
        args = parser.parse_args()    
        return args   
    
   
def main(args):
    
    policy = build_visual_diffsuion_err(args.ckpt_path, args.statistic_path, args.k, args.num_episodes)
    evaluator = LIBEROEval_err(task_suite_name='libero_goal', data_statistics=None, k=args.k, eval_horizon=args.eval_horizon, num_episodes=args.num_episodes,checkpoint_path=args.ckpt_path)
    evaluator.eval_episodes(policy, 0, save_path=args.statistic_path[:-15])

if __name__ == '__main__':
    args = get_args()
    main(args)
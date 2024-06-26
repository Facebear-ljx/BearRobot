from BearRobot.utils.evaluation.mp_libero_eval import LIBEROEval
from BearRobot.utils.logger.tb_log import TensorBoardLogger
from BearRobot.Agent import build_visual_diffsuion, build_visual_diffusion_mmpretrain

def main():
# logger = TensorBoardLogger(project_name='test', run_name='test', )
    ckpt_path = '/home/dodo/ljx/BearRobot/experiments/libero/libero_goal/diffusion_dnce/test_0612_eval/latest.pth'
    statistic_path = '/home/dodo/ljx/BearRobot/experiments/libero/libero_goal/diffusion_dnce/test_0612_eval/statistics.json'

    policy = build_visual_diffusion_mmpretrain(ckpt_path, statistic_path)
    evaluator = LIBEROEval(task_suite_name='libero_goal', data_statistics=None, eval_horizon=300, num_episodes=10)

    evaluator.eval_episodes(policy, 0, save_path='/home/dodo/ljx/BearRobot/experiments/libero/libero_goal/diffusion_dnce/test_0612_eval/')

if __name__ == '__main__':
    main()
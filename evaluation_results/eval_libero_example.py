from BearRobot.utils.evaluation.mp_libero_eval import LIBEROEval
from BearRobot.utils.logger.tb_log import TensorBoardLogger
from BearRobot.Agent import build_visual_diffsuion

def main():
# logger = TensorBoardLogger(project_name='test', run_name='test', )
    ckpt_path = '/home/dodo/ljx/BearRobot/experiments/libero/libero30/diffusion/resnet34_wstate_test/439999_0.6052184104919434.pth'
    statistic_path = '/home/dodo/ljx/BearRobot/experiments/libero/libero30/diffusion/resnet34_wstate_test/statistics.json'

    policy = build_visual_diffsuion(ckpt_path, statistic_path)
    evaluator = LIBEROEval(task_suite_name='libero_goal', data_statistics=None, eval_horizon=300, num_episodes=10)

    evaluator.eval_episodes(policy, 0)

if __name__ == '__main__':
    main()
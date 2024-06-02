from BearRobot.utils.evaluation.libero_eval import LIBEROEval
from BearRobot.utils.logger.tb_log import TensorBoardLogger
from BearRobot.Agent import build_visual_diffsuion

# logger = TensorBoardLogger(project_name='test', run_name='test', )
ckpt_path = '/home/dodo/ljx/BearRobot/experiments/libero/diffusion/test/latest.pth'
statistic_path = '/home/dodo/ljx/BearRobot/experiments/libero/diffusion/test/statistics.json'

policy = build_visual_diffsuion(ckpt_path, statistic_path)
evaluator = LIBEROEval(task_suite_name='libero_goal', data_statistics=None, eval_horizon=300)

evaluator.eval_episodes(policy, 0)
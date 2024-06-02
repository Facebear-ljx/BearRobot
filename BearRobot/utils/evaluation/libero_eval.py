import os

import torch
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from BearRobot.utils.evaluation.base_eval import BaseEval
from BearRobot.Agent.base_agent import BaseAgent
from BearRobot.utils.logger.base_log import BaseLogger

EPS = 1e-5
benchmark_dict = benchmark.get_benchmark_dict()

class LIBEROEval(BaseEval):
       def __init__(
              self,
              task_suite_name: str, # can choose libero_spatial, libero_spatial, libero_object, libero_100, all, etc.
              obs_key: list=['agentview_image', 'robot0_eye_in_hand_image'],
              data_statistics: dict=None,
              logger: BaseLogger=None,
              num_episodes: int=10,
              eval_freq: int=10,
              seed: int=42,
              rank: int=0,
       ):
              super(BaseEval, self).__init__()   
              self.task_suite_name = task_suite_name
              self.task_suite = benchmark_dict[task_suite_name]()
              self.obs_key = obs_key
              self.data_statistics = data_statistics
              
              self.num_episodes = num_episodes
              self.eval_freq = eval_freq
              self.logger = logger
              self.seed = seed
              
              self.rank = rank
              if self.rank == 0:
                     self._init_env()
       
       def _init_env(self):
              self.env = []
              for task_id in range(len(self.task_suite.tasks)):
                     task = self.task_suite.get_task(task_id)
                     task_name = task.name
                     task_description = task.language
                     task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
                     print(f"[info] retrieving task {task_id} from suite {self.task_suite_name}, the " + \
                            f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

                     # step over the environment
                     env_args = {
                            "bddl_file_name": task_bddl_file,
                            "camera_heights": 128,
                            "camera_widths": 128
                     }
                     env = OffScreenRenderEnv(**env_args)
                     env.seed(self.seed + 100)
                     env.reset()
                     init_states = self.task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
                     init_state_id = 0
                     obs = env.set_init_state(init_states[init_state_id])
                     
                     temp_dict = {}
                     temp_dict['env'] = env
                     temp_dict['obs'] = obs
                     
                     self.env.append(temp_dict)
       
       def _log_results(self, metrics: dict, steps: int):
              if self.logger is None:
                     # just print out and pass
                     print(metrics)
                     pass # TODO, implement the print function
              else:
                     # log the results to the logger
                     self.logger.log_metrics(metrics, steps)
       
       def _rollout(self, policy: BaseAgent):
              """
              rollout one episode and return the episode return
              """
              ep_rews = 0
              for env in self.env:
                     obs = env['obs']
                     for _ in range(200):
                            # get image
                            agent_view, wrist_view = obs['agentview_image'], obs['robot0_eye_in_hand_image']
                            
                            # get action and denorm
                            action = policy.get_action([agent_view, wrist_view], env['env'].language_instruction)[0]  # TODO, use temporal ensembling
                            action = action.reshape(-1)
                            
                            # step
                            obs, reward, done, info = env['env'].step(action)
                            ep_rews += reward
                            if done:
                                   break
              avg_succ_rate = ep_rews / len(self.env) / self.num_episodes
              print("Average success rate:", avg_succ_rate)
              return avg_succ_rate
       
       def eval_episodes(self, policy: BaseAgent, steps: int):
              """
              rollout several episodes and log the mean episode return
              """
              rews = []
              policy.eval()
              for _ in range(self.num_episodes):
                     rews.append(self._rollout(policy))
              eval_rewards = sum(rews) / len(rews)
              metrics = {"eval/rewards": eval_rewards}
              self._log_results(metrics, steps)
              return eval_rewards
              
              

              
       
       
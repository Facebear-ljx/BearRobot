import os

import torch
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv

from BearRobot.utils.evaluation.base_eval import BaseEval
from BearRobot.Agent.base_agent import BaseAgent
from BearRobot.utils.logger.base_log import BaseLogger

from tqdm import tqdm

import multiprocessing
from functools import partial
import imageio

EPS = 1e-5
benchmark_dict = benchmark.get_benchmark_dict()

class LIBEROEval(BaseEval):
       def __init__(
              self,
              task_suite_name: str, # can choose libero_spatial, libero_spatial, libero_object, libero_100, all, etc.
              obs_key: list=['agentview_image', 'robot0_eye_in_hand_image'],
              data_statistics: dict=None,
              logger: BaseLogger=None,
              eval_horizon: int=600,
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
              
              self.eval_horizon = eval_horizon
              self.num_episodes = num_episodes
              self.eval_freq = eval_freq
              self.logger = logger
              self.seed = seed
              
              self.rank = rank
              if self.rank == 0:
                     # self._init_env()
                     self._make_dir()
                     
       def _make_dir(self):
              task_suite_name = self.task_suite_name
              path = 'evaluation_results/results/libero/' + task_suite_name
              if not os.path.exists(path):
                     os.makedirs(path)
              self.base_dir = path
       
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
                     
                     temp_dict = {}
                     temp_dict['env'] = env
                     temp_dict['task_id'] = task_id
                     
                     self.env.append(temp_dict)
       
       def _log_results(self, metrics: dict, steps: int):
              if self.logger is None:
                     # just print out and pass
                     print(metrics)
                     pass # TODO, implement the print function
              else:
                     # log the results to the logger
                     self.logger.log_metrics(metrics, steps)
       
       def _rollout(task_id, policy: BaseAgent, task_suite, task_suite_name, base_dir, seed=0, num_episodes=10, eval_horizon=300):
              """
              rollout several episodes and return the episodes mean success rate
              """
              task = task_suite.get_task(task_id)
              task_description = task.language
              task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
              print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
                     f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

              # step over the environment
              env_args = {
                     "bddl_file_name": task_bddl_file,
                     "camera_heights": 128,
                     "camera_widths": 128
              }
              env = OffScreenRenderEnv(**env_args)
              env.seed(seed + 100)

              obs = env.reset()
              init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
              init_state_id = env['task_id'] % init_states.shape[0]
              obs = env.set_init_state(init_states[init_state_id])
              lang = env.language_instruction
              
              for i in range(num_episodes):
                     ep_rews = 0
                     images = []
                     policy._init_action_chunking(eval_horizon)
                     for t in tqdm(range(eval_horizon), desc=f'{lang}'):
                            # get image
                            images.append(np.flip(np.flip(obs["agentview_image"], 0), 1))
                            agent_view, wrist_view = obs['agentview_image'], obs['robot0_eye_in_hand_image']
                            
                            gripper_qpos = obs['robot0_gripper_qpos']
                            eef_pos = obs['robot0_eef_pos']
                            eef_quat = obs['robot0_eef_quat']
                            state = np.concatenate([gripper_qpos, eef_pos, eef_quat], axis=-1)
                            
                            # get action and denorm
                            action = policy.get_action([agent_view, wrist_view], env.language_instruction, state=state, t=t, k=0.25)
                            action = action.reshape(-1)
                            
                            # step
                            obs, reward, done, info = env.step(action)
                            ep_rews += reward
                            if done:
                                   break
                     save_path = f'{base_dir}/{lang}.mp4'

              imageio.mimsave(save_path, images, fps=50)
              avg_succ_rate = ep_rews / num_episodes
              print("Average success rate:", avg_succ_rate)
              return avg_succ_rate
       
       def eval_episodes(self, policy: BaseAgent, steps: int):
              """
              rollout several episodes and log the mean episode return
              """
              policy.eval()
              
              # multiprocess evaluation
              rollout_patial = partial(self._rollout, 
                                       policy=policy, 
                                       task_suite=self.task_suite, 
                                       task_suite_name=self.task_suite_name, 
                                       base_dir=self.base_dir, 
                                       seed=self.seed, 
                                       num_episodes=self.num_episodes, 
                                       eval_horizon=self.eval_horizon)
              pool = multiprocessing.Pool(processes=len(self.task_suite.tasks))
              rews = pool.map(rollout_patial, list(range(len(self.task_suite.tasks))))
              pool.close()
              pool.join()

              eval_rewards = sum(rews) / len(rews)
              metrics = {"eval/rewards": eval_rewards}
              self._log_results(metrics, steps)
              return eval_rewards
              
       
       def close_env(self):
              for env in self.env:
                     env['env'].close()
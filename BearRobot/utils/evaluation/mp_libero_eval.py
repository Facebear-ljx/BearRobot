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
              obs_key: list=['agentview_image', 'robot0_eye_in_hand_image', 'robot0_gripper_qpos', 'robot0_eef_pos', 'robot0_eef_quat'],
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
                     self._make_dir()
                     
       def _make_dir(self):
              task_suite_name = self.task_suite_name
              path = 'evaluation_results/results/libero/' + task_suite_name
              if not os.path.exists(path):
                     os.makedirs(path)
              self.base_dir = path
       
       def _init_env(self, task_id: int=0):
              # get task information and env args
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
              
              # init thesubprocess vector environment
              env_num = self.num_episodes
              env = SubprocVectorEnv(
                     [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
              )
              
              # environment reset 
              env.seed(self.seed + 100)
              env.reset()
              init_states = self.task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
              init_state_id = np.arange(self.num_episodes) % init_states.shape[0]
              obs = env.set_init_state(init_states[init_state_id])
              
              # return the environment
              env_dict = {}
              env_dict['env'] = env
              env_dict['language_instruction'] = task_description
              env_dict['obs'] = obs
              
              return env_dict
       
       def _log_results(self, metrics: dict, steps: int):
              if self.logger is None:
                     # just print out and pass
                     print(metrics)
                     pass # TODO, implement the print function
              else:
                     # log the results to the logger
                     self.logger.log_metrics(metrics, steps)
       
       def raw_obs_to_stacked_obs(self, obs, lang):
              env_num = len(obs)
              
              data = {
                     "obs": {},
                     "lang": lang,
              }
              
              for key in self.obs_key:
                     data["obs"][key] = []
                     
              for i in range(env_num):
                     for key in self.obs_key:
                            data['obs'][key].append(obs[i][key])
              
              for key in data['obs']:
                     data['obs'][key] = np.stack(data['obs'][key])
              
              return data     
       
       def np_image_to_tensor(self, image: np.array):
              B, H, W, C = image.shape
              image = image / 255. if image.max() > 10 else image
              image = torch.from_numpy(image).permute(0, 3, 1, 2).to(torch.float32)  # B, C, H, W
              
              norm_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).repeat(B, 1, 1, 1)  # B, C, 1, 1
              norm_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).repeat(B, 1, 1, 1)
              
              image = (image - norm_mean) / norm_std
              return image  # B, C, H, W

       def _rollout(self, policy: BaseAgent, task_id: int=0):
              """
              rollout one episode and return the episode return
              """
              env = self._init_env(task_id)
              lang = env['language_instruction']
              obs = env['obs']
              policy._init_action_chunking(self.eval_horizon, self.num_episodes)
              
              images = []
              for t in tqdm(range(self.eval_horizon), desc=f'{lang}'):
                     # data transform
                     ## image
                     data = self.raw_obs_to_stacked_obs(obs, lang)
                     obs, lang = data['obs'], data['lang']
                     agent_view = self.np_image_to_tensor(obs['agentview_image']).unsqueeze(1)
                     wrist_view = self.np_image_to_tensor(obs['robot0_eye_in_hand_image']).unsqueeze(1)
                     image_input = torch.cat([agent_view, wrist_view], dim=1).unsqueeze(1)
                     
                     ### record the video
                     B, H, W, C = obs["agentview_image"].shape
                     images.append(np.flip(np.flip(obs["agentview_image"], 1), 2).reshape(B * H, W, C))
                     
                     ## proprio
                     gripper_qpos = obs['robot0_gripper_qpos']
                     eef_pos = obs['robot0_eef_pos']
                     eef_quat = obs['robot0_eef_quat']
                     state = np.concatenate([gripper_qpos, eef_pos, eef_quat], axis=-1)
                     
                     # get action
                     action = policy.get_action(image_input, lang, state=state, t=t, k=0.25)
                     action = action.reshape(self.num_episodes, -1)
                     
                     # step
                     obs, reward, done, info = env['env'].step(action)
                     if done.all():
                            break
              save_path = f'{self.base_dir}/{lang}.mp4'
              self._save_video(save_path, images, done, fps=30)
              # imageio.mimsave(save_path, images, fps=30)
              
              num_success = 0
              for k in range(self.num_episodes):
                     num_success += int(done[k])
              avg_succ_rate = num_success / self.num_episodes
              print("Average success rate:", avg_succ_rate)
              
              env['env'].close()
              return avg_succ_rate
       
       def _save_video(self, save_path: str, images: list, done: list, fps=30): 
              imageio.mimsave(save_path, images, fps=fps)
              
       
       def eval_episodes(self, policy: BaseAgent, steps: int):
              """
              rollout several episodes and log the mean episode return
              """
              rews = []
              policy.eval()
              # for _ in tqdm(range(self.num_episodes), desc="Evaluating..."):
              for task_id in tqdm(range(len(self.task_suite.tasks)), desc="Evaluating..."):
                     rews.append(self._rollout(policy, task_id))
              eval_rewards = sum(rews) / len(rews)
              metrics = {"eval/succ_rate": eval_rewards}
              self._log_results(metrics, steps)
              return eval_rewards
              
       
       def close_env(self):
              for env in self.env:
                     env['env'].close()
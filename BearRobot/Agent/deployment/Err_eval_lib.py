import os

import torch
import torchvision.transforms as transforms
import numpy as np
import random

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv

from BearRobot.utils.evaluation.base_eval import BaseEval
from BearRobot.Agent.base_agent import BaseAgent
from BearRobot.utils.logger.base_log import BaseLogger
from BearRobot.utils.dataset.dataloader import openimage
from data.libero.data_process import demo2frames
from data.libero.data_process import get_libero_frame
from V_network import V_model #这个环境里pip install -e .是原版bearobot库，所以V_network直接写相对路径，不然找不到

from tqdm import tqdm
import json

import multiprocessing
from functools import partial
import imageio
from collections import deque

EPS = 1e-5
benchmark_dict = benchmark.get_benchmark_dict()
frame_length_dict = demo2frames.frame_counts_dict()

def has_normalize(transform):
       if isinstance(transform, transforms.Compose):
              for t in transform.transforms:
                     if isinstance(t, transforms.Normalize):
                            return True
       return False


class LIBEROEval_err(BaseEval):
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
              checkpoint_path: str = None,
              k: float = 0.2,
       ):
              super(BaseEval, self).__init__()   
              self.task_suite_name = task_suite_name
              self.task_suite = benchmark_dict[task_suite_name]()
              self.obs_key = obs_key
              self.data_statistics = data_statistics
              self.state_dim = 9
              self.eval_horizon = eval_horizon
              self.num_episodes = num_episodes
              self.eval_freq = eval_freq
              self.logger = logger
              self.seed = seed
              self.s_avoid =  [ False for _ in range(self.num_episodes)]
              self.rank = rank
              self.v_func = V_model(state_dim=9)
              self.k = k
              self.device = 'cuda' 
              
              
       def _make_dir(self, save_path):
              if self.rank == 0:
                     task_suite_name = self.task_suite_name
                     path = os.path.join(save_path, task_suite_name)
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
              
              ### sample one begin and end image frame to construct image goal
              # Generating paths for img_begin and img_end
              task_name = task_description.replace(" ", "_") + "_demo"
              demo_paths = demo2frames.get_demos_for_task(task_name, frame_length_dict)
              demo_path = random.choice(demo_paths)

              # eval with the beginning frame and the endding frame
              env_dict = {}
              transform = transforms.ToTensor()
              base_dir='/home/dodo/ljx/BearRobot/data/libero/dataset/'
              env_dict['img_begin'] = transform(openimage(os.path.join(base_dir, "libero/data_jpg/", demo_path, "image0/0.jpg")))
              end_idx = frame_length_dict[demo_path] - 1 
              env_dict['img_end'] = transform(openimage(os.path.join(base_dir, "libero/data_jpg/", demo_path, f"image0/{end_idx}.jpg")))

              # return the environment
              env_dict['env'] = env
              env_dict['language_instruction'] = task_description
              env_dict['obs'] = obs
              
              return env_dict
       
       def _log_results(self, metrics: dict, steps: int):
              if self.logger is None:
                     # just print out and save the results and pass
                     print(metrics)
                     save_name = os.path.join(self.base_dir, 'results.json')
                     with open(save_name, 'a+') as f:
                            line = json.dumps(metrics)
                            f.write(line+'\n')
              else:
                     # log the results to the logger
                     self.logger.log_metrics(metrics, steps)
                     self.logger.save_metrics(metrics, steps, self.base_dir)
       
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
       
       def np_image_to_tensor(self, image: np.array, normalize_img: bool):
              B, H, W, C = image.shape
              image = image / 255. if image.max() > 10 else image
              image = torch.from_numpy(image).permute(0, 3, 1, 2).to(torch.float32)  # B, C, H, W
              
              if normalize_img:
                     norm_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).repeat(B, 1, 1, 1)  # B, C, 1, 1
                     norm_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).repeat(B, 1, 1, 1)
                     
                     image = (image - norm_mean) / norm_std
              return image  # B, C, H, W

       def _rollout(self, policy: BaseAgent, task_id: int=0, img_goal=False):
              """
              rollout one episode and return the episode return
              """
              env = self._init_env(task_id)
              self._env = env
              lang = env['language_instruction']
              obs = env['obs']
              img_begin = env['img_begin']
              img_end = env['img_end']
              #注释了，跑一下脚本看看效果
              # v_func = self.v_func()
              k = self.k
              policy.policy._init_action_chunking(self.eval_horizon, self.num_episodes)
              
              images = []
              buffer = []
              T_buffer = []
              s_avoid = self.s_avoid
              
              current_times = torch.zeros(self.num_episodes).to(self.device)
              estimate_T = torch.zeros(self.num_episodes).to(self.device)
              count_T = torch.zeros(self.num_episodes).to(self.device)
              traj = deque(maxlen=100)
              traj_list = [ [] for _ in range(self.num_episodes)]
              self.recover_flag = [ False for _ in range(self.num_episodes)]
                     
              for t in tqdm(range(self.eval_horizon), desc=f'{lang}'):
                     # time ++
                     current_times += 1
                     
                     # data transform
                     ## image
                     data = self.raw_obs_to_stacked_obs(obs, lang)
                     obs, lang = data['obs'], data['lang']
                     
                     normalize_img = has_normalize(policy.transform)
                     agent_view = self.np_image_to_tensor(obs['agentview_image'], normalize_img).unsqueeze(1)
                     wrist_view = self.np_image_to_tensor(obs['robot0_eye_in_hand_image'], normalize_img).unsqueeze(1)
                     image_input = torch.cat([agent_view, wrist_view], dim=1).unsqueeze(1).to(self.device)
                     
                     ### record the video
                     B, H, W, C = obs["agentview_image"].shape
                     images.append(np.flip(np.flip(obs["agentview_image"], 1), 2).reshape(B * H, W, C))
                     img_recovery = np.flip(np.flip(obs["agentview_image"], 1), 2).reshape(B * H, W, C)
                     
                     ## proprio
                     gripper_qpos = obs['robot0_gripper_qpos']
                     eef_pos = obs['robot0_eef_pos']
                     eef_quat = obs['robot0_eef_quat']
                     
                     # add images and proprio(state) to buffer
                     try:
                            s_dim = policy.s_dim
                     except:
                            s_dim = policy.policy.module.s_dim
                     state = np.concatenate([gripper_qpos, eef_pos, eef_quat], axis=-1) if s_dim > 0 else None
                     
                     # process state
                     state = torch.from_numpy(state.astype(np.float32)).view(-1, policy.s_dim) if state is not None else None
                     state = ((state - policy.s_mean) / policy.s_std).to('cuda') if state is not None else None
              
                     # get action using img_begin and img_end embedding difference
                     if img_goal:
                            # using the current frame as the img_begin   # TODO delete this if not useful
                            img_begin = torch.from_numpy(obs['agentview_image']).permute(0, 3, 1, 2) / 255
                            
                            action = policy.get_action(image_input, None, state=state, current_time=current_times, t=t, k=0.25, img_begin=img_begin, img_end = img_end, img_goal=img_goal,s_avoid = s_avoid)
                     else:
                            action = policy.get_action(image_input, lang, state=state, current_time=current_times, t=t, k=0.25, s_avoid = s_avoid) 
                     # reshape
                     action = action.reshape(self.num_episodes, -1)
                     
                     # step
                     obs, reward, done, info = env['env'].step(action)
                     if done.all():
                            break
                     
                     # ----------------------Trial && Error part--------------------------              
                     # resize images
                     imgs = image_input.squeeze(1) # [B, V, C, H, W]
                     imgs_tuple = torch.chunk(imgs, 2, dim=1)
                     image_1 = imgs_tuple[0].squeeze(1) # [B, C, H, W] 
                     image_2 = imgs_tuple[1].squeeze(1) # [B, C, H, W]
                     
                     # add traj
                     b = policy.v_model(image_1, image_2, state).argmax(dim=-1)
                     for i in range(self.num_episodes):
                            traj_list[i].append(b[i])
                     
                     # estimate T
                     count_T += 1
                     for i in range(self.num_episodes):
                            if b[i]>0:
                                   T = current_times[i] / (b[i] * 0.02)
                                   estimate_T[i] = estimate_T[i] + (T - estimate_T[i]) / count_T[i]
                                   if T>200:
                                          print(f"episode {i}, T is {T}")

                     # before 30 frames, no recovery
                     if count_T[0].item() < 30:
                            continue
                            
                     # judgement for recovery
                     for i in range(self.num_episodes):
                            current_t = current_times[i].item() - 1
                            k_value = round(estimate_T[i].item() * self.k)
                            t_k = int(current_t - k_value)
                            
                            if t_k < 0:
                                   continue # too early
       
                            t_b = traj_list[i][-1]
                            t_k_b = traj_list[i][t_k]
                            if (t_b - t_k_b)*0.02 > k:
                                   print("traj[0] ",traj_list[0])
                                   print(f"Recovery happens at episode {i}\n t_b = {t_b}  t={current_t}  \
                                          t_k_b = {t_k_b}  t_k={t_k}  estimate_T={estimate_T[i]}")
                                   self.recover_flag[i] = True
                                   current_times[i] = 0 # curren_times reset
                                   traj_list[i].clear() # traj reset
                                   # count_T && estimate_T reserve
                                   
                     
                     # visualize recovery
                     if any(self.recover_flag):
    
                            for i in range(B):
                                   # 如果对应的 recover_flag 为 True，将最上方 H*W 的图片变为纯红色
                                   if self.recover_flag[i]:
                                          img_recovery[i * H:(i + 1) * H, :W] = [255, 0, 0]

                            # keep
                            for i in range(30): 
                                   images.append(img_recovery)      
                     
                     # do recovery
                     self.recover(state=state, task_id=task_id)   
          
                     # if torch.all(estimate_T>0):
                     #        k_index = torch.round(k * estimate_T)
                     #        t_minus_k = current_times - k_index
                     #        t_minus_k = torch.where(t_minus_k < 0, current_times, t_minus_k).tolist()
                     #        t_k_b = traj[-k_index]
                            # t_b
                            
                     #注释了，跑一下脚本看看效果
                     #get and compare v to decide recover or not
                     # vt = v_func(agent_view,wrist_view,state) # return a distribution of frame process
                     # T_buffer.append(t/torch.argmax(vt))
                     # buffer.append((agent_view, wrist_view, state))
                     
                     # if t < round(k * np.mean(T_buffer)): 
                     #        vt_k = v_func(buffer[0][0],buffer[0][1],buffer[0][2])
                     # else:
                     #        index = t - round(k * np.mean(T_buffer))
                     #        if index < 0:
                     #               index = 0  # In case the index is less than 0
                     #               vt_k = v_func(buffer[index][0], buffer[index][1], buffer[index][2])
                            
                     # v_diff = vt - vt_k
                     
                     # if v_diff < round(k * np.mean(T_buffer)):
                     #        s_avoid = self.recover(s_avoid, state,task_id)
                     
              
              save_path = f'{self.base_dir}/{lang}.mp4' #change this 
              self._save_video(save_path, images, done, fps=30)
              
              num_success = 0
              for k in range(self.num_episodes):
                     num_success += int(done[k])
              avg_succ_rate = num_success / self.num_episodes
             
              metrics = {f'sim/{self.task_suite_name}/{lang}': avg_succ_rate}
              self._log_results(metrics, self.step)
              
              env['env'].close()
              return avg_succ_rate

       def recover(self, state, task_id: int=0):
              for i in range(self.num_episodes):
                     if self.recover_flag[i]:
                            self.s_avoid[i] = True
                            self.recover_flag[i] = False
                            # print(f"!!!Episode index {i} recover!!!")
                            
                            # reset method 1
                            self._env['env'].reset(i) 
                            
                            # reset method 2
                            # init_states = self.task_suite.get_task_init_states(task_id) 
                            # init_state_id = np.arange(self.num_episodes) % init_states.shape[0]
                            # self._env['env'][i].set_init_state(init_states[init_state_id][i])
              
              return self.s_avoid
       
       
       def _save_video(self, save_path: str, images: list, done: list, fps=30): 
              imageio.mimsave(save_path, images, fps=fps)
              
       
       def eval_episodes(self, policy: BaseAgent, steps: int, save_path: str, img_goal=False):
              """
              rollout several episodes and log the mean episode return
              """
              self._make_dir(save_path)
              self.step = steps
              
              rews = []
              policy.eval()
              # for _ in tqdm(range(self.num_episodes), desc="Evaluating..."):
              for task_id in tqdm(range(len(self.task_suite.tasks)), desc="Evaluating..."):
                     rews.append(self._rollout(policy, task_id, img_goal))
              eval_rewards = sum(rews) / len(rews)
              metrics = {f'sim/{self.task_suite_name}/all': eval_rewards}
              self._log_results(metrics, self.step)
              return eval_rewards
              
       
       def close_env(self):
              for env in self.env:
                     env['env'].close()
                     

from mmengine import fileio
import io
import os
import json
import yaml

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from BearRobot.Agent import *
from BearRobot.Agent.ACT import ACTAgent 
from BearRobot.Net.my_model.diffusion_model import VisualDiffusion, VisualDiffusion_pretrain
from BearRobot.Net.my_model.ACT_model import ACTModel
from Err_ddpm_bc import IDQL_Agent

def openjson(path):
       value  = fileio.get_text(path)
       dict = json.loads(value)
       return dict


def convert_str(item):
        try:
                return int(item)
        except:
                try:
                        return float(item)
                except:
                        return item


def wandb_args2dict(ckpt_path, wandb_name: str=None):
        if wandb_name is None:
                wandb_name = 'latest-run'
        try:
                wandb_path = os.path.join('/'.join(ckpt_path.split('.pth')[0].split('/')[:-1]), f'wandb/{wandb_name}/files/wandb-metadata.json')
                meta_data = openjson(wandb_path)
                args = [convert_str(arg.split('--')[-1]) for arg in meta_data['args']]
                config_dict = dict(zip(args[::2], args[1::2]))
                print("-------load meta data from wandb succ!------------")
                print_dict = json.dumps(config_dict, indent=4, sort_keys=True)
                print(print_dict)
                return config_dict
        except:
                print("Automatically load wandb meta data fail, please provide your meta data mannually")
                return {}


def wandb_yaml2dict(ckpt_path, wandb_name: str=None, wandb_path: str=None):
        if wandb_name is None:
                wandb_name = 'latest-run'
        try:
                if wandb_path is None:
                        wandb_path = os.path.join('/'.join(ckpt_path.split('.pth')[0].split('/')[:-1]), f'wandb/{wandb_name}/files/config.yaml')
                with open(wandb_path, 'r') as stream:
                        config = yaml.safe_load(stream)
                del config['wandb_version']
                del config['_wandb']
                config_dict = {key: value['value'] for key, value in config.items()}
                print("-------load meta data from wandb succ!------------")
                print_dict = json.dumps(config_dict, indent=4, sort_keys=True)
                print(print_dict)
                return config_dict
        except:
                print("Automatically load wandb meta data fail, please provide your meta data mannually")
                return {}


def load_ckpt(agent, ckpt_path):
       from collections import OrderedDict

       ckpt = fileio.get(ckpt_path)
       ckpt_policy = fileio.get("/home/dodo/ljx/BearRobot/experiments/libero/libero_goal/diffusion/resnet34_wstate_0625_10_DecisionNCE-T/latest.pth")
       with io.BytesIO(ckpt) as f:
              ckpt = torch.load(f, map_location='cuda')
       
       new_ckpt = OrderedDict()
       for key in ckpt['model'].keys():
              new_key = key.replace(".module", '')
              new_ckpt[new_key] = ckpt['model'][key]

       ckpt['model'] = new_ckpt
       agent.load_state_dict(ckpt['model'])
       
       # get policy weights
       with io.BytesIO(ckpt_policy) as f:
              ckpt_policy = torch.load(f, map_location='cuda')
       
       new_ckpt = OrderedDict()
       for key in ckpt_policy['model'].keys():
              new_key = key.replace(".module", '')
              new_ckpt[new_key] = ckpt_policy['model'][key]

       ckpt_policy['model'] = new_ckpt
       agent.policy.load_state_dict(ckpt_policy['model'])
       agent.eval()
       agent.policy.eval()
       return agent.to(0)


def build_visual_diffsuion_err(ckpt_path: str, statistics_path: str, k: float=0.2, num_episodes: int=10 ,wandb_name: str=None, wandb_path: str=None):
       kwargs = wandb_yaml2dict(ckpt_path, wandb_name, wandb_path=wandb_path)
       model = VisualDiffusion(view_num=2,
                            output_dim=7 * kwargs['ac_num'],
                            **kwargs).to(0)
       agent = IDQL_Agent(policy_model=model, num_episodes=num_episodes, k=k, **kwargs) 
       agent.policy.get_statistics(statistics_path)
       agent.policy.get_transform(kwargs['img_size'])
       agent.get_statistics(statistics_path)
       agent.get_transform(kwargs['img_size'])
       return load_ckpt(agent, ckpt_path)




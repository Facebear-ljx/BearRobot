import d4rl
import gym

from utils.evaluation.base_eval import BaseEval
from Agent.base_agent import BaseAgent

EPS = 1e-5

class D4RLEval(BaseEval):
       def __init__(
              self,
              env_name: str,
              data_statistics: dict,
       ):     
              self.env_name = env_name
              self.env = gym.make(env_name)
              self.datadata_statistics = data_statistics
       
       def _log_results(self, metrics: dict, steps: int):
              if self.logger is None:
                     # just print out and pass
                     pass # TODO, implement the print function
              else:
                     # log the results to the logger
                     self.logger.log_metrics(metrics, steps)
       
       def _rollout(self, policy: BaseAgent):
              ep_rews = 0.
              state = self.env.reset()
              while True:
                     state = (state - self.datadata_statistics['s_mean']) / (self.datadata_statistics['s_std'] + EPS)
                     action = policy.get_action(state).cpu().detach().numpy()
                     state, reward, done, _ = self.env.step(action)
                     ep_rews += reward
                     if done:
                            break
                     ep_rews = d4rl.get_normalized_score(env_name=self.env_name, score=ep_rews) * 100    
              return ep_rews          
       
       def eval_episodes(self, policy: BaseAgent, steps: int):
              rews = []
              policy.eval()
              for _ in range(self.num_episodes):
                     rews.append(self._rollout(policy))
              eval_rewards = sum(rews) / len(rews)
              metrics = {"eval/rewars": eval_rewards}
              self._log_results(metrics, steps)
              
              

              
       
       
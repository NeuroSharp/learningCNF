import os.path
import copy
import torch
import math
# from torch.distributions import Categorical
import ipdb
import random
import time
import itertools
import torch.nn.utils as tutils

from settings import *
from cadet_env import *
from rl_model import *
from qbf_data import *
from qbf_model import QbfClassifier
from utils import *
from rl_utils import *
from cadet_utils import *
from episode_reporter import *

CADET_BINARY = './cadet'
SAVE_EVERY = 50000
INVALID_ACTION_REWARDS = -100


all_episode_files = ['data/mvs.qdimacs']

settings = CnfSettings()

reporter = DqnEpisodeReporter("{}/{}".format(settings['rl_log_dir'], log_name(settings)), tensorboard=settings['report_tensorboard'])
env = CadetEnv(CADET_BINARY, **settings.hyperparameters)
exploration = LinearSchedule(400000, settings['EPS_END'])
lr_multiplier = 1.0
num_iterations = 2000000
lr_schedule = PiecewiseSchedule([
                                     (0,                   1e-4 * lr_multiplier),
                                     (num_iterations / 10, 1e-4 * lr_multiplier),
                                     (num_iterations / 2,  5e-5 * lr_multiplier),
                                ],
                                outside_value=5e-5 * lr_multiplier)  
policy = None
target_model = None
optimizer = None
total_steps = 0
time_last_episode = 0
episode_reward = 0
inference_time = []
backprop_time = []
actions_history = []


def select_action(obs, exploration, model=None, testing=False, **kwargs):    
  if testing or random.random() > exploration:
    is_random = False
    logits = model(obs, **kwargs)
    action = logits.squeeze().max(0)[1].data   # argmax when testing    
    action = action[0]
  else:
    is_random = True
    # sample randomly n times, so we do get an invalid action now and then
    for _ in range(10):
      action = np.random.randint(obs.ground.size(1))
      if action_allowed(obs,action):
        break
  return action, is_random
  
def dqn_main():
  global policy, optimizer, all_episode_files, total_steps, episode_reward, time_last_episode
  global target_model, backprop_time, inference_time

  memory = ReplayMemory(settings['replay_size'])
  policy = create_policy()
  target_model = create_policy(is_clone=True)
  copy_model_weights(policy,target_model)
  optimizer = optim.Adam(policy.parameters(), lr=1e-2)
  reporter.log_env(settings['rl_log_envs'])
  ds = QbfDataset(fnames=settings['rl_train_data'])
  all_episode_files = ds.get_files_list()
  num_param_updates = 0
  done = True

  for t in itertools.count():            
    if done:
      last_obs, env_id = new_episode(env,all_episode_files)
      cmat_pos, cmat_neg = last_obs.cmat_pos, last_obs.cmat_neg
      if settings['rl_log_all']:
        reporter.log_env(env_id)      

    begin_time = time.time()
    action, is_random = select_action(last_obs, exploration.value(total_steps), policy) 
    if not is_random:
      inference_time.append(time.time()-begin_time)
    if not action_allowed(last_obs,action):
      # print('Chose an invalid action!')
      reward = INVALID_ACTION_REWARDS
      done = True    
    else:
      try:
        env_obs = EnvObservation(*env.step(action))
        state, vars_add, vars_remove, activities, decision, clause, reward, done = env_obs
        
      except Exception as e:
        print(e)
        ipdb.set_trace()

    total_steps += 1
    episode_reward += reward
    
    # prepare the next observation
    if not done:
      obs = env.process_observation(last_obs,env_obs)
    
    else:
      s = total_steps - time_last_episode
      reporter.add_stat(env_id,s,episode_reward, total_steps)
      episode_reward = 0
      time_last_episode = total_steps
      obs = None
    
    memory.push(last_obs,action,obs,reward)
    last_obs = obs

    if t > settings['learning_starts'] and not (t % settings['learning_freq']):
      begin_time = time.time()
      batch = memory.sample(settings['batch_size'])
      states, actions, next_states, rewards = zip(*batch)
      collated_batch = collate_transitions(batch,settings=settings)
      non_final_mask = settings.ByteTensor(tuple(map(lambda s: s is not None,
                                          next_states)))

      state_action_values = policy(collated_batch.state).gather(1,Variable(collated_batch.action).view(-1,1))
      next_state_values = Variable(settings.zeros(settings['batch_size']))
      next_state_values[non_final_mask] = target_model(collated_batch.next_state).detach().max(1)[0]
      expected_state_action_values = (next_state_values * settings['gamma']) + Variable(collated_batch.reward)
      loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

      # ipdb.set_trace()
      optimizer.zero_grad()
      loss.backward()      
      torch.nn.utils.clip_grad_norm(policy.parameters(), settings['grad_norm_clipping'])
      optimizer.step()
      end_time = time.time()
      backprop_time.append(end_time-begin_time)

      num_param_updates += 1
      if num_param_updates % settings['target_update_freq'] == 0:
        print('Updating target network (step {})'.format(total_steps))
        copy_model_weights(policy,target_model)

    if not (t % 1000) and t > 0:
      print('[{}] Epsilon is {}'.format(total_steps, exploration.value(total_steps)))
      print('[{}] {} Items in memory'.format(total_steps, len(memory)))
      inference_time = inference_time[-100:]
      print('[{}] Mean forward time: {}'.format(total_steps, np.mean(inference_time)))
    if not (t % 200) and t > settings['learning_starts']:
      backprop_time = backprop_time[-100:]
      print('[{}] Mean backwards time (Batch size {}): {}'.format(total_steps, settings['batch_size'],np.mean(backprop_time)))
      reporter.report_stats()
      print(policy.invalid_bias)

    if t % SAVE_EVERY == 0 and t > settings['learning_starts']:
      torch.save(policy.state_dict(),'%s/%s_iter%d.model' % (settings['model_dir'],utils.log_name(settings), t))
    



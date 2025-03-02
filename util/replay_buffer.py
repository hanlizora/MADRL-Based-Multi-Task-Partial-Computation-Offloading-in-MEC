import copy
import torch
from util.utils import GetValueInputs, GetPolicyInputs

class ReplayBuffer():
   def __init__(self, params):
       self.device_num = params.device_num
       self.train_freq = params.train_freq
       self.train_time_slots = params.train_time_slots
       self.state_dim = params.state_dim
       self.obs_dim = params.obs_dim
       self.action_dim = params.action_dim
       self.gamma = params.gamma
       self.lamda = params.lamda
       
       self.ps = [0, 0]
       self.edge_obs = [[None for j in range(self.train_time_slots + 1)]
                              for i in range(self.train_freq)]
       self.device_obss = [[None for j in range(self.train_time_slots + 1)]
                                 for i in range(self.train_freq)]
       self.device_acts = [[None for j in range(self.train_time_slots + 1)]
                                      for i in range(self.train_freq)]
       self.device_act_logprobs = [[None for j in range(self.train_time_slots + 1)]
                                         for i in range(self.train_freq)]
       self.joint_reward = [[None for j in range(self.train_time_slots + 1)]
                                  for i in range(self.train_freq)]
       
   def store(self, edge_obs, device_obss, device_acts, device_act_logprobs, joint_reward):
       # store sampled data
       self.edge_obs[self.ps[0]][self.ps[1]] = copy.copy(edge_obs)
       self.device_obss[self.ps[0]][self.ps[1]] = copy.copy(device_obss)
       self.device_acts[self.ps[0]][self.ps[1]] = copy.copy(device_acts)
       self.device_act_logprobs[self.ps[0]][self.ps[1]] = copy.copy(device_act_logprobs)
       self.joint_reward[self.ps[0]][self.ps[1]] = copy.copy(joint_reward)
       
       # update positions
       if self.ps[1] == self.train_time_slots:
           self.ps[0] = (self.ps[0] + 1) % (self.train_freq)
       self.ps[1] = (self.ps[1] + 1) % (self.train_time_slots + 1)
       
   def get_training_data(self, value_net):
       '''GAE'''
       v_inputs = torch.zeros([self.train_freq, self.train_time_slots + 1, self.state_dim])
       for i in range(self.train_freq):
           for j in range(self.train_time_slots + 1):
               edge_obs = copy.copy(self.edge_obs[i][j]) 
               if edge_obs == None:
                   print(i, j)
               device_obss = copy.copy(self.device_obss[i][j])
               inputs = GetValueInputs(edge_obs, device_obss)
               v_inputs[i, j] = inputs
       
       with torch.no_grad():
           vs = value_net(v_inputs.reshape([-1, self.state_dim]))
       vs = vs.reshape([self.train_freq, self.train_time_slots + 1, 1])
       
       # [train_freq, train_time_slots, 1]
       rewards = torch.tensor(self.joint_reward, dtype = torch.float) \
                 [:, 0: self.train_time_slots].unsqueeze(-1)
       
       # [train_episodes, train_time_slots, 1]
       deltas = rewards + self.gamma * vs[:, 1: self.train_time_slots + 1] - \
                vs[:, 0: self.train_time_slots]
       gae = 0
       advs = torch.zeros([self.train_freq, self.train_time_slots, 1])
       for t in reversed(range(self.train_time_slots)):
           gae = deltas[:, t] + self.lamda * self.gamma * gae
           advs[:, t] = gae
       # [train_episodes, train_time_slots, 1]
       v_tags = advs + vs[:, 0: self.train_time_slots]
       # normalization
       advs = (advs - advs.mean()) / (advs.std() + 1e-5)
       
       '''training data - value network'''
       # [train_freq x train_time_slots, state_dim]
       v_inputs = v_inputs[:, 0: self.train_time_slots].reshape([-1, self.state_dim])
       # [train_freq x train_time_slots, 1]
       v_tags = v_tags.reshape([-1, 1])
       
       '''training data - policy networks'''
       p_inputs = torch.zeros([self.train_freq, self.train_time_slots, 
                               self.device_num, self.obs_dim])
       acts = torch.zeros([self.train_freq, self.train_time_slots, 
                           self.device_num, self.action_dim])
       act_logprobs = torch.zeros([self.train_freq, self.train_time_slots, 
                                   self.device_num, 1])
       for i in range(self.train_freq):
           for j in range(self.train_time_slots):
               for k in range(self.device_num):
                   obs = copy.copy(self.device_obss[i][j][k])
                   inputs = GetPolicyInputs(obs)
                   p_inputs[i, j, k] = inputs
                   acts[i, j, k] = torch.tensor(self.device_acts[i][j][k],
                                                dtype = torch.float)
                   act_logprobs[i, j, k] = torch.tensor(self.device_act_logprobs[i][j][k],
                                                        dtype = torch.float)
       # [train_freq x train_time_slots, device_num, obs_dim]
       p_inputs = p_inputs.reshape([-1, self.device_num, self.obs_dim])
       # [train_freq x train_time_slots, device_num, action_dim]
       acts = acts.reshape([-1, self.device_num, self.action_dim])
       # [train_freq x train_time_slots, device_num, 1]
       act_logprobs = act_logprobs.reshape([-1, self.device_num, 1])
       advs = advs.reshape([-1, 1])
       
       return v_inputs, v_tags, p_inputs, acts, act_logprobs, advs
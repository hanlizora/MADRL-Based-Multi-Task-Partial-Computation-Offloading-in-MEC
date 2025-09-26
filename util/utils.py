import numpy as np
import torch
import torch.nn as nn

'''calculate mean and std dynamically'''
class RunningMeanStd():
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)
            
    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)
            
class ObsScaling():
    def __init__(self, task_num, max_data_size, max_comp_dens, std_comp_freq):
        self.task_num = task_num
        # unit: Mb
        self.max_data_size = max_data_size
        # unit: Gcycles/bit
        self.max_comp_dens = max_comp_dens
        # unit: Gcycles/s 
        self.std_comp_freq = std_comp_freq
        # unit: s
        self.max_dly_cons = self.max_data_size * pow(10, 6) * self.max_comp_dens \
                            / self.std_comp_freq
                            
    def __call__(self, edge_obs, device_obss):
        edge_obs[0] = np.clip(edge_obs[0], 0, 20) / 10
        for i in range(len(device_obss)):
            device_obss[i][0] = np.clip(device_obss[i][0], 0, 20) / 10
            device_obss[i][1] = np.clip(device_obss[i][1], 0, 20) / 10
            for j in range(self.task_num):
                device_obss[i][2 + j * 3] /= self.max_data_size
                device_obss[i][2 + j * 3 + 1] /= self.max_comp_dens
                device_obss[i][2 + j * 3 + 2] /= self.max_dly_cons
                
class RewardScaling():
    def __init__(self, gamma):
        # discount factor
        self.gamma = gamma
        self.R = 0
        self.r_running_ms = RunningMeanStd(1)

    def __call__(self, reward):
        self.R = self.gamma * self.R + reward
        self.r_running_ms.update(self.R)
        reward = float(reward / (self.r_running_ms.std + 1e-8))
        
        return reward
        
    # reset 'R' when an episode is done 
    def reset(self):
        self.R = 0
        
class GaussianNoise():
    def __init__(self, action_dim, mu = 0.25, sigma = 0.5):
        self.action_dim = action_dim
        self.mu = mu
        self.sigma = sigma
        
    def sample(self):
        x = np.random.normal(self.mu, self.sigma, self.action_dim)
                
        return x

def OrthogonalInit(layer, gain = 1.0):
    for name, params in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(params, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(params, gain = gain)
            
def GetPolicyInputs(obs):
    inputs = torch.tensor(obs, dtype = torch.float).reshape([1, -1])
    
    return inputs

def GetValueInputs(edge_obs, device_obss):
    inputs = []
    
    inputs += edge_obs
    for i in range(len(device_obss)):
        inputs += device_obss[i]
    inputs = torch.tensor(inputs, dtype = torch.float).reshape([1, -1])
    
    return inputs
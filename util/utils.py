import numpy as np
import torch
from config.params import get_params

# params
params = get_params()

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
    def __init__(self):
        self.e_running_ms = RunningMeanStd(1)
        self.d_running_ms = RunningMeanStd(2 * params.device_num)
        
        self.max_data_size = params.max_data_size
        self.max_comp_dens = params.max_comp_dens
        self.max_dly_cons = self.max_data_size * pow(10, 6) * self.max_comp_dens \
                            / params.std_comp_freq
                            
    def __call__(self, edge_obs, device_obss, evaluate):
        nor_edge_obs = edge_obs[0]
        nor_device_obss = []
        for obs in device_obss:
            nor_device_obss += [obs[0], obs[1]]
        
        # whether to update mean and std
        # during evaluation, update = False
        if not evaluate:
            self.e_running_ms.update(nor_edge_obs)
            self.d_running_ms.update(nor_device_obss)
        nor_edge_obs = (nor_edge_obs - self.e_running_ms.mean) / \
                                                 (self.e_running_ms.std + 1e-8)
        nor_device_obss = (nor_device_obss - self.d_running_ms.mean) / \
                                                 (self.d_running_ms.std + 1e-8)
        
        edge_obs[0] = float(nor_edge_obs)
        for i in range(params.device_num):    
            device_obss[i][0] = float(nor_device_obss[i * 2])
            device_obss[i][1] = float(nor_device_obss[i * 2 + 1])
            for j in range(params.task_num):
                device_obss[i][2 + j * 3] /= self.max_data_size
                device_obss[i][2 + j * 3 + 1] /= self.max_comp_dens
                device_obss[i][2 + j * 3 + 2] /= self.max_dly_cons
        
class RewardScaling():
    def __init__(self):
        # discount factor
        self.gamma = params.gamma
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
        
def GetValueInputs(edge_obs, device_obss):
    inputs = []
    
    inputs += edge_obs
    for i in range(params.device_num):
        inputs += device_obss[i]
    inputs = torch.tensor(inputs, dtype = torch.float).reshape([1, -1])
    
    return inputs

def GetPolicyInputs(obs):
    inputs = torch.tensor(obs, dtype = torch.float).reshape([1, -1])
    
    return inputs
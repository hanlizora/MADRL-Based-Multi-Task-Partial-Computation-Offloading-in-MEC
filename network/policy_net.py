import torch
import torch.nn as nn
from util.utils import OrthogonalInit

class MappoPolicyNet(nn.Module):
    def __init__(self, alg_params):
        super(MappoPolicyNet, self).__init__()
        
        self.fc1 = nn.Linear(alg_params.obs_dim, alg_params.p_hid_dims[0])
        self.fc2 = nn.Linear(alg_params.p_hid_dims[0], alg_params.p_hid_dims[1])
        self.fc3 = nn.Linear(alg_params.p_hid_dims[1], alg_params.action_dim)
        self.tanh = nn.Tanh()
            
        self.log_std = nn.Parameter(torch.tensor([0] * alg_params.action_dim,
                                    dtype = torch.float))
        
        # orthogonal initialization
        if alg_params.use_orthogonal_init:
            OrthogonalInit(self.fc1)
            OrthogonalInit(self.fc2)
            OrthogonalInit(self.fc3, gain = 0.01)
        
    def forward(self, obs):
        x = self.tanh(self.fc1(obs))
        x = self.tanh(self.fc2(x))
        mean = self.tanh(self.fc3(x)) * 5 + 5
        
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        
        return mean, std
    
class MaddpgPolicyNet(nn.Module):
    def __init__(self, alg_params):
        super(MaddpgPolicyNet, self).__init__()
        
        self.fc1 = nn.Linear(alg_params.obs_dim, alg_params.p_hid_dims[0])
        self.fc2 = nn.Linear(alg_params.p_hid_dims[0], alg_params.p_hid_dims[1])
        self.fc3 = nn.Linear(alg_params.p_hid_dims[1], alg_params.action_dim)
        self.tanh = nn.Tanh()
        
        # orthogonal initialization
        if alg_params.use_orthogonal_init:
            OrthogonalInit(self.fc1)
            OrthogonalInit(self.fc2)
            OrthogonalInit(self.fc3, gain = 0.01)
    
    def forward(self, obs):
        x = self.tanh(self.fc1(obs))
        x = self.tanh(self.fc2(x))
        act = self.tanh(self.fc3(x)) + 1
        
        return act
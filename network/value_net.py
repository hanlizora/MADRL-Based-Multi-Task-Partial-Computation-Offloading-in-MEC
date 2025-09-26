import torch
import torch.nn as nn
from util.utils import OrthogonalInit

class MappoValueNet(nn.Module):
    def __init__(self, alg_params):
        super(MappoValueNet, self).__init__()
        
        self.fc1 = nn.Linear(alg_params.state_dim, alg_params.v_hid_dims[0])
        self.fc2 = nn.Linear(alg_params.v_hid_dims[0], alg_params.v_hid_dims[1])
        self.fc3 = nn.Linear(alg_params.v_hid_dims[1], 1)
        self.tanh = nn.Tanh()
        
        # orthogonal initialization
        if alg_params.use_orthogonal_init:
            OrthogonalInit(self.fc1)
            OrthogonalInit(self.fc2)
            OrthogonalInit(self.fc3)
    
    def forward(self, state):
        x = self.tanh(self.fc1(state))
        x = self.tanh(self.fc2(x))
        v = self.fc3(x)
                        
        return v
    
class MaddpgValueNet(nn.Module):
    def __init__(self, alg_params):
        super(MaddpgValueNet, self).__init__()
                
        self.fc1 = nn.Linear(alg_params.state_action_dim, alg_params.v_hid_dims[0])
        self.fc2 = nn.Linear(alg_params.v_hid_dims[0], alg_params.v_hid_dims[1])
        self.fc3 = nn.Linear(alg_params.v_hid_dims[1], 1)
        self.tanh = nn.Tanh()
                        
        # orthogonal initialization
        if alg_params.use_orthogonal_init:
            OrthogonalInit(self.fc1)
            OrthogonalInit(self.fc2)
            OrthogonalInit(self.fc3)
            
    def forward(self, state, joint_act):
        x = self.tanh(self.fc1(torch.concat([state, joint_act], dim = -1)))
        x = self.tanh(self.fc2(x))
        v = self.fc3(x)
        
        return v
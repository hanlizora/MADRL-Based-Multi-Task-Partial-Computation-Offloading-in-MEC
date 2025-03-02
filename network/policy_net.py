import torch
import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self, params):
        super(PolicyNet, self).__init__()
        
        def orthogonal_init(layer, gain = 1.0):
            for name, params in layer.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(params, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(params, gain = gain)
        
        self.fc1 = nn.Linear(params.obs_dim, params.p_hid_dims[0])
        self.fc2 = nn.Linear(params.p_hid_dims[0], params.p_hid_dims[1])
        self.fc3 = nn.Linear(params.p_hid_dims[1], params.action_dim)
        # self.fc4 = nn.Linear(params.p_hid_dims[1], params.action_dim)
        self.tanh = nn.Tanh()
        # self.softplus = nn.Softplus()
            
        self.log_std = nn.Parameter(torch.tensor([0] * params.action_dim,
                                                 dtype = torch.float))
         
        # trick: orthogonal initialization
        if params.use_orthogonal:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain = 0.01)
            # orthogonal_init(self.fc4, gain = 0.01)
        
    def forward(self, obs):
        x = self.tanh(self.fc1(obs))
        x = self.tanh(self.fc2(x))
        mean = self.tanh(self.fc3(x)) * 5 + 5
        # std = self.softplus(self.fc4(x))
        
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        
        return mean, std
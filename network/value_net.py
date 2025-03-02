import torch.nn as nn

class ValueNet(nn.Module):
    def __init__(self, params):
        super(ValueNet, self).__init__()

        def orthogonal_init(layer, gain=1.0):
            for name, params in layer.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(params, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(params, gain=gain)

        self.fc1 = nn.Linear(params.state_dim, params.v_hid_dims[0])
        self.fc2 = nn.Linear(params.v_hid_dims[0], params.v_hid_dims[1])
        self.fc3 = nn.Linear(params.v_hid_dims[1], 1)
        self.tanh = nn.Tanh()
        
        # trick: orthogonal initialization
        if params.use_orthogonal:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, state):
        x = self.tanh(self.fc1(state))
        x = self.tanh(self.fc2(x))
        v = self.fc3(x)
        
        return v
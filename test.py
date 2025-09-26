import torch
import torch.nn as nn
from torch.distributions import Categorical

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 2)
    
    def forward(self, obs):
        x = self.fc1(obs)
        logits = self.fc2(x)
        
        return logits

net = PolicyNet()
optimizer = torch.optim.Adam(net.parameters(),
                             lr = 5e-3)

'''
for params in net.parameters():
    print(params)
for name, params in net.named_parameters():
    print(name, ':', params.size())
'''

obs = torch.normal(0, 1, [2, 5])
logits = net(obs)
dist = Categorical(logits = logits)
loss = dist.entropy()
print("loss: ", loss.mean())
optimizer.zero_grad()
loss.mean().backward()

for name, params in net.named_parameters():
    if params.grad is not None:
        print(name, params.grad.norm().item())

optimizer.step()

'''
for params in net.parameters():
    print(params)
for name, paramrs in net.named_parameters():
    print(name, ':', params.size())
'''
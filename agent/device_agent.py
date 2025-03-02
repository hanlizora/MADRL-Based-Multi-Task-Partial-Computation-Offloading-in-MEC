import numpy as np
import torch
from torch.distributions import Normal
from network.policy_net import PolicyNet
from util.utils import GetPolicyInputs

class DeviceAgent():
    def __init__(self, agent_id, params):
        # agent id
        self.agent_id = agent_id
        
        # policy network
        self.p_net = PolicyNet(params)
        self.action_dim = params.action_dim
        
        # evaluation mode
        self.eval_mode = params.eval_mode

    def choose_action(self, obs, evaluate):
        if not evaluate or (evaluate and self.eval_mode == "mappo"):
            with torch.no_grad():
                p_inputs = GetPolicyInputs(obs)
                mean, std = self.p_net(p_inputs)
            if evaluate:
                act = mean.squeeze(dim = 0).tolist()
                act_logprob = None
            else:
                dist = Normal(mean, std)
                act = dist.sample()
                act = torch.clamp(act, 0, 10)
                act_logprob = dist.log_prob(act).sum(-1)
                act = act.squeeze(dim = 0).tolist()
                act_logprob = float(act_logprob)
        elif self.eval_mode == "local_comp":
                act = [0 for i in range(self.action_dim)]
                act_logprob = None
        elif self.eval_mode == "edge_comp":
                act = [10 for i in range(self.action_dim)] 
                act_logprob = None
        else:
            act = [np.random.uniform(0, 10) for i in range(self.action_dim)] 
            act_logprob = None
            
        return act, act_logprob
    
    def load_net(self, path):
        self.p_net.load_state_dict(torch.load(path))
    
    def update_net(self, params):
        self.p_net.load_state_dict(params)
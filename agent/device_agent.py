from abc import abstractmethod
import numpy as np
import torch
from torch.distributions import Normal
from network.policy_net import MappoPolicyNet, MaddpgPolicyNet
from util.utils import GetPolicyInputs, GaussianNoise

class MappoDeviceAgent():
    def __init__(self, agent_id, gen_params, alg_params):
        # agent id
        self.agent_id = agent_id
        
        # policy network
        self.p_net = MappoPolicyNet(alg_params)
        
        self.evaluate = gen_params.evaluate
        
    def choose_action(self, obs):
        p_inputs = GetPolicyInputs(obs)
        with torch.no_grad():
            mean, std = self.p_net(p_inputs)
        if self.evaluate:
            act = mean.squeeze(dim = 0).tolist()
            act_logprob = None
        else:
            dist = Normal(mean, std)
            act = dist.sample()
            act = torch.clamp(act, 0, 10)
            act_logprob = dist.log_prob(act).sum(-1)
            act = act.squeeze(dim = 0).tolist()
            act_logprob = float(act_logprob)
            
        return act, act_logprob
    
    def update_net(self, params):
        self.p_net.load_state_dict(params)
        
    def load_net(self, path):
        self.update_net(torch.load(path))
        
class MaddpgDeviceAgent():
    def __init__(self, agent_id, gen_params, alg_params):
        # agent id
        self.agent_id = agent_id
        
        # policy network
        self.p_net = MaddpgPolicyNet(alg_params)
        
        # action noise
        self.use_action_noise = alg_params.use_action_noise
        if self.use_action_noise:
            self.action_noise = GaussianNoise(alg_params.action_dim)
            
        self.evaluate = gen_params.evaluate
        
    def choose_action(self, obs):
        p_inputs = GetPolicyInputs(obs)
        with torch.no_grad():
            act = self.p_net(p_inputs).squeeze(0).tolist()
        if not self.evaluate:
            act = np.clip((act + self.action_noise.sample()), 0, 2).tolist()
        
        return act
        
    def update_net(self, params):
        self.p_net.load_state_dict(params)
        
    def load_net(self, path):
        self.update_net(torch.load(path))

class StaticDeviceAgent():
    def __init__(self, agent_id, gen_params):
        # agent id
        self.agent_id = agent_id
        self.task_num = gen_params.task_num
    
    @abstractmethod
    def choose_action(self):
        pass
    
class LocalComputingDeviceAgent(StaticDeviceAgent):
    def __init__(self, agent_id, gen_params):
        super().__init__(agent_id, gen_params)
        
    def choose_action(self):
        act = [0 for i in range(self.task_num + 1)]
        
        return act

class EdgeComputingDeviceAgent(StaticDeviceAgent):
    def __init__(self, agent_id, gen_params):
        super().__init__(agent_id, gen_params)
        
    def choose_action(self):
        act = [1 for i in range(self.task_num + 1)]
        
        return act

class RandomComputingDeviceAgent(StaticDeviceAgent):
    def __init__(self, agent_id, gen_params):
        super().__init__(agent_id, gen_params)
        
    def choose_action(self):
        act = [np.random.uniform(0, 1) for i in range(self.task_num + 1)]
        
        return act
import os
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Normal
from network.policy_net import PolicyNet
from network.value_net import ValueNet

class EdgeAgent():
    def __init__(self, params):
       self.device_num = params.device_num
        
       # train
       self.train_freq = params.train_freq
       self.train_time_slots = params.train_time_slots
       self.v_batch_size = params.v_batch_size
       self.p_batch_size = params.p_batch_size
       self.v_epochs = params.v_epochs
       self.p_epochs = params.p_epochs
       self.p_clip = params.p_clip
       self.enty_coef = params.enty_coef
       self.v_lr = params.v_lr
       self.p_lr = params.p_lr
       self.gamma = params.gamma
       # gradient clip
       self.use_grad_clip = params.use_grad_clip
       self.v_grad_clip = params.v_grad_clip
       self.p_grad_clip = params.p_grad_clip
       self.weights_dir = params.weights_dir
       # learning-rate decay
       self.use_lr_decay = params.use_lr_decay
       self.min_v_lr = params.min_v_lr
       self.min_p_lr = params.min_p_lr
       self.decay_fac = params.decay_fac
       
       # value network
       self.v_net = ValueNet(params)
       self.v_optimizer = torch.optim.Adam(self.v_net.parameters(),
                                           lr = self.v_lr)
       
       # policy networks
       self.p_nets = []
       self.p_optimizers = []
       for i in range(self.device_num):
           p_net = PolicyNet(params)
           self.p_nets.append(p_net)
           
           p_optimizer = torch.optim.Adam(p_net.parameters(),
                                          lr = self.p_lr)
           self.p_optimizers.append(p_optimizer)
           
       # load networks' weights 
       if params.load_weights:
           v_path = self.weights_dir + "v_net_params.pkl"
           self.v_net.load_state_dict(torch.load(v_path))
           
           for i in range(self.device_num):
               p_path = self.weights_dir + "p_net_params_" + str(i) + ".pkl"
               self.p_nets[i].load_state_dict(torch.load(p_path))
               
    def train_nets(self, replay_buffer):
        '''training data'''
        # v_inputs: [train_freq x train_time_slots, state_dim]
        # v_tags: [train_freq x train_time_slots, 1]
        # p_inputs: [train_freq x train_time_slots, device_num, obs_dim]
        # acts: [train_freq x train_time_slots, device_num, action_dim]
        # act_logprobs: [train_freq x train_time_slots, device_num, 1]
        # advs: [train_freq x train_time_slots, 1]
        v_inputs, v_tags, p_inputs, acts, act_logprobs, advs = \
                                    replay_buffer.get_training_data(self.v_net)
                                    
        self.train_value_net(v_inputs, v_tags)
        
        for i in range(self.device_num):
            self.train_policy_net(i, p_inputs[:, i], acts[:, i], act_logprobs[:, i], advs)
        
        if self.use_lr_decay:
            self.decay_lr()
       
    def train_value_net(self, v_inputs, v_tags):
        v_total_size = self.train_freq * self.train_time_slots
        for e in range(self.v_epochs):
            for ids in BatchSampler(SubsetRandomSampler(range(v_total_size)),
                                    self.v_batch_size, False):
                vs = self.v_net(v_inputs[ids])
                
                loss = F.mse_loss(v_tags[ids], vs)
                
                self.v_optimizer.zero_grad()
                loss.backward()
                # gradient clip
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.v_net.parameters(), 
                                                   self.v_grad_clip)
                self.v_optimizer.step()
        
    def train_policy_net(self, agent_id, p_inputs, acts, act_logprobs, advs):
        p_total_size = self.train_freq * self.train_time_slots
        
        for e in range(self.p_epochs):
            for ids in BatchSampler(SubsetRandomSampler(range(p_total_size)),
                                    self.p_batch_size, False):
                # mean: [p_batch_size, p_out_dim]
                # std: [p_batch_size, p_out_dim]
                mean, std = self.p_nets[agent_id](p_inputs[ids])
                dist = Normal(mean, std)
                # [p_batch_size]
                enty = dist.entropy().sum(-1)
                # [p_batch_size]
                new_act_logprobs = dist.log_prob(acts[ids]).sum(-1)
                # [p_batch_size]
                old_act_logprobs = act_logprobs[ids].reshape([-1])
                ratios = torch.exp(new_act_logprobs - old_act_logprobs)
                surr1 = ratios * advs[ids].reshape([-1])
                surr2 = torch.clamp(ratios, 1 - self.p_clip, 1 + self.p_clip) * \
                        advs[ids].reshape([-1])
                
                loss = -(torch.min(surr1, surr2) + self.enty_coef * enty)

                self.p_optimizers[agent_id].zero_grad()
                loss.mean().backward()
                # gradient clip
                if self.use_grad_clip:  
                    torch.nn.utils.clip_grad_norm_(self.p_nets[agent_id].parameters(),
                                                   self.p_grad_clip)
                self.p_optimizers[agent_id].step()
    
    def decay_lr(self):
       self.v_lr = max(self.v_lr * self.decay_fac, self.min_v_lr)
       for params in self.v_optimizer.param_groups:
           params['lr'] = self.v_lr
               
       self.p_lr = max(self.p_lr * self.decay_fac, self.min_p_lr)
       for i in range(self.device_num):
           for params in self.p_optimizers[i].param_groups:
               params['lr'] = self.p_lr
        
    def save_nets(self, e_id):
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
            
        torch.save(self.v_net.state_dict(), 
                   self.weights_dir + "v_net_params_" + str(e_id) + ".pkl")
        
        for i in range(self.device_num):
            torch.save(self.p_nets[i].state_dict(),
                       self.weights_dir + "p_net_params_" + str(i) + "_" + str(e_id) + ".pkl")
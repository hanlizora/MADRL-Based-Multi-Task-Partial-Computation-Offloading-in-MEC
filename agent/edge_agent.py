import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Normal
from network.value_net import MappoValueNet, MaddpgValueNet
from network.policy_net import MappoPolicyNet, MaddpgPolicyNet

class MappoEdgeAgent():
    def __init__(self, gen_params, alg_params):
        self.device_num = gen_params.device_num
        
        # training
        self.train_time_slots = alg_params.train_time_slots
        self.train_freq = alg_params.train_freq
        self.train_batch_size = alg_params.train_batch_size
        self.v_epochs = alg_params.v_epochs
        self.p_epochs = alg_params.p_epochs
        self.p_clip = alg_params.p_clip
        self.enty_coef = alg_params.enty_coef
        self.v_lr = alg_params.v_lr
        self.p_lr = alg_params.p_lr
        self.gamma = alg_params.gamma
        # gradient clip
        self.use_grad_clip = alg_params.use_grad_clip
        self.v_grad_clip = alg_params.v_grad_clip
        self.p_grad_clip = alg_params.p_grad_clip
        # learning-rate decay
        self.use_lr_decay = alg_params.use_lr_decay
        self.min_v_lr = alg_params.min_v_lr
        self.min_p_lr = alg_params.min_p_lr
        self.decay_fac = alg_params.decay_fac
        self.weights_dir = alg_params.weights_dir
        
        # value network
        self.v_net = MappoValueNet(alg_params)
        self.v_optimizer = torch.optim.Adam(self.v_net.parameters(),
                                            lr = self.v_lr)
        
        # policy networks
        self.p_nets = []
        self.p_optimizers = []
        for i in range(self.device_num):
            p_net = MappoPolicyNet(alg_params)
            self.p_nets.append(p_net)
                
            p_optimizer = torch.optim.Adam(p_net.parameters(),
                                           lr = self.p_lr)
            self.p_optimizers.append(p_optimizer)
            
        # load networks' weights 
        if alg_params.load_weights:
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
        v_inputs, v_tags, p_inputs, \
        acts, act_logprobs, advs = replay_buffer.get_training_data(self.v_net)
                                    
        self.train_value_net(v_inputs, v_tags)
        
        for i in range(self.device_num):
            self.train_policy_net(i, p_inputs[:, i], acts[:, i], act_logprobs[:, i], advs)
        
        if self.use_lr_decay:
            self.decay_lr()
       
    def train_value_net(self, v_inputs, v_tags):
        total_size = self.train_freq * self.train_time_slots
        for e in range(self.v_epochs):
            for ids in BatchSampler(SubsetRandomSampler(range(total_size)),
                                    self.train_batch_size, False):
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
        total_size = self.train_freq * self.train_time_slots
        for e in range(self.p_epochs):
            for ids in BatchSampler(SubsetRandomSampler(range(total_size)),
                                    self.train_batch_size, False):
                # mean: [train_batch_size, p_out_dim]
                # std: [train_batch_size, p_out_dim]
                mean, std = self.p_nets[agent_id](p_inputs[ids])
                dist = Normal(mean, std)
                # [train_batch_size]
                enty = dist.entropy().sum(-1)
                
                # [train_batch_size]
                new_act_logprobs = dist.log_prob(acts[ids]).sum(-1)
                # [train_batch_size]
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
       if self.v_lr > self.min_v_lr:  
           self.v_lr *= self.decay_fac
           for params in self.v_optimizer.param_groups:
               params['lr'] = self.v_lr
       
       if self.p_lr > self.min_p_lr:  
           self.p_lr *= self.decay_fac
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
            
class MaddpgEdgeAgent():
    def __init__(self, gen_params, alg_params):
       self.device_num = gen_params.device_num
       self.action_dim = alg_params.action_dim
       
       # training
       self.warm_time_slots = alg_params.warm_time_slots
       self.train_batch_size = alg_params.train_batch_size
       self.v_epochs = alg_params.v_epochs
       self.p_epochs = alg_params.p_epochs
       self.buffer_size = alg_params.buffer_size
       self.gamma = alg_params.gamma
       self.v_lr = alg_params.v_lr
       self.p_lr = alg_params.p_lr
       # gradient clip
       self.use_grad_clip = alg_params.use_grad_clip
       self.v_grad_clip = alg_params.v_grad_clip
       self.p_grad_clip = alg_params.p_grad_clip
       self.weights_dir = alg_params.weights_dir
       # learning-rate decay
       self.use_lr_decay = alg_params.use_lr_decay
       self.min_v_lr = alg_params.min_v_lr
       self.min_p_lr = alg_params.min_p_lr
       self.decay_intl = alg_params.decay_intl
       self.decay_fac = alg_params.decay_fac
       
       # value network
       self.v_net = MaddpgValueNet(alg_params)
       # target value network 
       self.target_v_net = MaddpgValueNet(alg_params) 
       self.target_v_net.load_state_dict(self.v_net.state_dict())
       # optimizer
       self.v_optimizer = torch.optim.Adam(self.v_net.parameters(),
                                           lr = self.v_lr)
       
       self.p_nets = []
       self.target_p_nets = []
       self.p_optimizers = []
       for i in range(self.device_num):
           # policy network
           p_net = MaddpgPolicyNet(alg_params)
           self.p_nets.append(p_net)
           # target policy network
           target_p_net = MaddpgPolicyNet(alg_params)
           target_p_net.load_state_dict(p_net.state_dict())
           self.target_p_nets.append(target_p_net)
           # optimizer
           p_optimizer = torch.optim.Adam(p_net.parameters(),
                                          lr = self.p_lr)
           self.p_optimizers.append(p_optimizer)
           
       # load networks' weights
       if alg_params.load_weights:
           v_path = self.weights_dir + "v_net_params.pkl"
           self.v_net.load_state_dict(torch.load(v_path))
           target_v_path = self.weights_dir + "target_v_net_params.pkl"
           self.target_v_net.load_state_dict(torch.load(target_v_path))
           
           for i in range(self.device_num):
               p_path = self.weights_dir + "p_net_params_" + str(i) + ".pkl"
               self.p_nets[i].load_state_dict(torch.load(p_path))
               target_p_path = self.weights_dir + "target_p_net_params_" + str(i) + ".pkl"
               self.target_p_nets[i].load_state_dict(torch.load(target_p_path))
    
    def train_nets(self, total_time_slots, replay_buffer):
        if total_time_slots >= self.warm_time_slots:
            if total_time_slots < self.buffer_size:
                batch_ids = np.random.choice(range(total_time_slots),
                                             self.train_batch_size, replace = False)
            else:
                batch_ids = np.random.choice(range(self.buffer_size),
                                             self.train_batch_size, replace = False)
            
            '''training data'''
            # batch_states: [batch_size, state_dim]
            # batch_device_obss: [batch_size, device_num, obs_dim]
            # batch_joint_acts: [batch_size, joint_act_dim]
            # batch_joint_rewards: [batch_size, 1]
            # batch_next_states: [batch_size, state_dim]
            # batch_next_device_obss: [batch_size, device_num, obs_dim]
            batch_states, batch_device_obss, \
            batch_joint_acts, batch_joint_rewards, \
            batch_next_states, batch_next_device_obss = replay_buffer.sample(batch_ids)
            
            self.train_value_net(batch_states, batch_joint_acts, 
                                 batch_joint_rewards, 
                                 batch_next_states, batch_next_device_obss)
            
            for i in range(self.device_num):
                self.train_policy_net(i, batch_states, batch_device_obss[:, i], 
                                      batch_joint_acts)
            
            if self.use_lr_decay:
                self.decay_lr(total_time_slots)
    
    def train_value_net(self, batch_states, batch_joint_acts, 
                              batch_joint_rewards, 
                              batch_next_states, batch_next_device_obss):
        with torch.no_grad():
            batch_next_joint_acts = []
            for i in range(self.device_num):
                batch_next_acts = self.target_p_nets[i](batch_next_device_obss[:, i])
                batch_next_joint_acts.append(batch_next_acts)
            # [batch_size, joint_act_dim]
            batch_next_joint_acts = torch.concat(batch_next_joint_acts, dim = -1)
            # [batch_size, 1]
            next_qs = self.target_v_net(batch_next_states, batch_next_joint_acts)
            target_qs = batch_joint_rewards + self.gamma * next_qs
            # normalization
            target_qs = (target_qs - target_qs.mean()) / (target_qs.std() + 1e-5)
            
        for i in range(self.v_epochs):
            # [batch_size, 1]
            qs = self.v_net(batch_states, batch_joint_acts)
            
            v_loss = F.mse_loss(target_qs, qs)
            
            self.v_optimizer.zero_grad()
            v_loss.backward()
            # gradient clip
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.v_net.parameters(), 
                                               self.v_grad_clip)
            self.v_optimizer.step()
            
    def train_policy_net(self, agent_id, batch_states, batch_device_obss, batch_joint_acts):
        for i in range(self.p_epochs):
            batch_joint_acts_ = batch_joint_acts.clone()
            batch_acts = self.p_nets[agent_id](batch_device_obss)
            batch_joint_acts_[:, agent_id * self.action_dim:
                                (agent_id + 1) * self.action_dim] = batch_acts
            
            p_loss = (-self.v_net(batch_states, batch_joint_acts_)).mean()
            
            self.p_optimizers[agent_id].zero_grad()
            p_loss.backward()
            # gradient clip
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.p_nets[agent_id].parameters(), 
                                               self.p_grad_clip)
            self.p_optimizers[agent_id].step()
            
    def update_target_nets(self, total_time_slots):
        if total_time_slots >= self.warm_time_slots:
            self.target_v_net.load_state_dict(self.v_net.state_dict())
            
            for i in range(self.device_num):
                self.target_p_nets[i].load_state_dict(self.p_nets[i].state_dict())
                
    def decay_lr(self, total_time_slots):
        if total_time_slots % self.decay_intl == 0:
            if self.v_lr > self.min_v_lr:
                self.v_lr -= self.decay_fac
                for params in self.v_optimizer.param_groups:
                    params['lr'] = self.v_lr
            
            if self.p_lr > self.min_p_lr:
                self.p_lr -= self.decay_fac
                for i in range(self.device_num):
                    for params in self.p_optimizers[i].param_groups:
                        params['lr'] = self.p_lr
        
    def save_nets(self, total_time_slots):
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
            
        torch.save(self.v_net.state_dict(),
                   self.weights_dir + "v_net_params_" + 
                   str(total_time_slots) + ".pkl")
        torch.save(self.target_v_net.state_dict(),
                   self.weights_dir + "target_v_net_params_" + 
                   str(total_time_slots) + ".pkl")
        
        for i in range(self.device_num):
            torch.save(self.p_nets[i].state_dict(),
                       self.weights_dir + "p_net_params_" + 
                       str(i) + "_" + str(total_time_slots) + ".pkl")
            torch.save(self.target_p_nets[i].state_dict(),
                       self.weights_dir + "target_p_net_params_" + 
                       str(i) + "_" + str(total_time_slots) + ".pkl")
import copy
import numpy as np
import torch
from env.mec_env import MECEnv
from agent.device_agent import MappoDeviceAgent, MaddpgDeviceAgent, \
LocalComputingDeviceAgent, EdgeComputingDeviceAgent, RandomComputingDeviceAgent
from agent.edge_agent import MappoEdgeAgent, MaddpgEdgeAgent
from util.replay_buffer import MappoReplayBuffer, MaddpgReplayBuffer
from util.utils import ObsScaling, RewardScaling

class Rollout:
    def __init__(self, gen_params, alg_params):
        self.device_num = gen_params.device_num
        self.task_num = gen_params.task_num
        
        self.evaluate = gen_params.evaluate
        self.train_mode = gen_params.train_mode
        self.eval_mode = gen_params.eval_mode
        
        # MEC env
        self.mec_env = MECEnv(gen_params)
        
        # device agents
        self.device_agents = []
        for i in range(self.device_num):
            if (not self.evaluate and self.train_mode == "mappo") or \
               (self.evaluate and self.eval_mode == "mappo"):
                self.device_agents.append(MappoDeviceAgent(i, gen_params, alg_params))
            if (not self.evaluate and self.train_mode == "maddpg") or \
               (self.evaluate and self.eval_mode == "maddpg"):
                self.device_agents.append(MaddpgDeviceAgent(i, gen_params, alg_params))
            if self.evaluate and self.eval_mode == "local_comp":
                self.device_agents.append(LocalComputingDeviceAgent(i, gen_params))
            if self.evaluate and self.eval_mode == "edge_comp":
                self.device_agents.append(EdgeComputingDeviceAgent(i, gen_params))
            if self.evaluate and self.eval_mode == "random_comp":
                self.device_agents.append(RandomComputingDeviceAgent(i, gen_params))
        
        # edge agent and replay buffer
        if not self.evaluate and self.train_mode == "mappo":
           self.edge_agent = MappoEdgeAgent(gen_params, alg_params)
           self.replay_buffer = MappoReplayBuffer(gen_params, alg_params)
        if not self.evaluate and self.train_mode == "maddpg":
           self.edge_agent = MaddpgEdgeAgent(gen_params, alg_params)
           self.replay_buffer = MaddpgReplayBuffer(gen_params, alg_params)
        
        # obs scaling
        if not self.evaluate or (self.evaluate and self.eval_mode[0] == "m"):
            if alg_params.use_obs_scaling:
                self.obs_scaling = ObsScaling(gen_params.task_num, 
                                              gen_params.max_data_size,
                                              gen_params.max_comp_dens,
                                              gen_params.std_comp_freq)
        
        # training
        if not self.evaluate:
            # fix random seed
            torch.manual_seed(alg_params.train_seed)
            np.random.seed(alg_params.train_seed)
            
            self.train_mode = gen_params.train_mode
            self.train_time_slots = alg_params.train_time_slots
            self.train_freq = alg_params.train_freq
            if self.train_mode == "maddpg":
                self.target_update_freq = alg_params.target_update_freq
            self.save_freq = alg_params.save_freq
            
            # reward scaling
            if alg_params.use_reward_scaling:
                self.reward_scaling = RewardScaling(alg_params.gamma)
            
            # initialize agents' policy networks
            for i in range(self.device_num):
                self.device_agents[i].update_net(self.edge_agent.p_nets[i].state_dict())
        # evaluation
        else:
            # fix random seed
            torch.manual_seed(gen_params.eval_seed)
            np.random.seed(gen_params.eval_seed)
            
            self.eval_time_slots = gen_params.eval_time_slots
            
            # initialize agents' policy networks
            if self.eval_mode[0] == "m":
                for i in range(self.device_num):
                    path = alg_params.weights_dir + "p_net_params_" + str(i) + ".pkl"
                    self.device_agents[i].load_net(path)
        
        self.joint_reward = None
        self.device_rewards = None
        self.joint_cost = None
        self.device_costs = None
        self.edge_comp_ql = None
        self.device_comp_qls = None
        self.device_comp_dlys = None
        self.device_csum_engys = None
        self.device_comp_expns = None
        self.device_overtime_nums = None
        
    def reset(self):
        if hasattr(self, "reward_scaling"):
            self.reward_scaling.reset()
        
        self.joint_reward = 0
        self.device_rewards = np.zeros([self.device_num], dtype = np.float32)
        self.joint_cost = 0
        self.device_costs = np.zeros([self.device_num], dtype = np.float32)
        self.edge_comp_ql = 0
        self.device_comp_qls = np.zeros([self.device_num], dtype = np.float32)
        self.device_comp_dlys = np.zeros([self.device_num], dtype = np.float32)
        self.device_csum_engys = np.zeros([self.device_num], dtype = np.float32)
        self.device_comp_expns = np.zeros([self.device_num], dtype = np.float32)
        self.device_overtime_nums = np.zeros([self.device_num], dtype = np.float32)
        
    def run(self, e_id):
        # reset
        self.reset()
        
        edge_obs, device_obss = self.mec_env.reset()
        edge_comp_ql = edge_obs[0]
        device_comp_qls = [obs[0] for obs in device_obss]
        # obs scaling
        if hasattr(self, "obs_scaling"):
            self.obs_scaling(edge_obs, device_obss)
        
        # rollout
        time_slots = self.train_time_slots + 1 if not self.evaluate else self.eval_time_slots
        for t_id in range(1, time_slots + 1):
            print("-------------time slot: " + str(t_id) + "-------------")
            
            # choose action (use deterministic strategy during evaluation)
            device_acts = [None for i in range(self.device_num)]
            if "Mappo" in type(self.device_agents[0]).__name__:
                # store actions used for interacting with the MEC env 
                device_acts_ = [[] for i in range(self.device_num)]
                if not self.evaluate:
                    device_act_logprobs = [None for i in range(self.device_num)]
                for i in range(self.device_num):
                    act, act_logprob = self.device_agents[i].choose_action(device_obss[i])
                    device_acts[i] = act
                    for j in range(self.task_num + 1):
                        device_acts_[i].append(act[j] / 10)
                    if not (act_logprob == None):
                        device_act_logprobs[i] = act_logprob
            if "Maddpg" in type(self.device_agents[0]).__name__:
                # store actions used for interacting with the MEC env
                device_acts_ = [[] for i in range(self.device_num)]
                for i in range(self.device_num):
                    act = self.device_agents[i].choose_action(device_obss[i])
                    device_acts[i] = act
                    for j in range(self.task_num + 1):
                        device_acts_[i].append((act[j * 10] + act[j * 10 + 1] + act[j * 10 + 2] + 
                                                act[j * 10 + 3] + act[j * 10 + 4] + act[j * 10 + 5] +
                                                act[j * 10 + 6] + act[j * 10 + 7] + act[j * 10 + 8] +
                                                act[j * 10 + 9]) / 20)
            if "Computing" in type(self.device_agents[0]).__name__:
                for i in range(self.device_num):
                    act = self.device_agents[i].choose_action()
                    device_acts[i] = act
                device_acts_ = device_acts
            
            # step
            joint_reward, device_rewards, \
            joint_cost, device_costs, \
            device_comp_dlys, device_csum_engys, \
            device_comp_expns, device_overtime_nums, \
            next_edge_obs, next_device_obss = self.mec_env.step(device_acts_)
            
            self.average(t_id, joint_reward, device_rewards,
                               joint_cost, device_costs,
                               edge_comp_ql, device_comp_qls,
                               device_comp_dlys, device_csum_engys,
                               device_comp_expns, device_overtime_nums)
            
            if hasattr(self, "reward_scaling"):
                joint_reward = self.reward_scaling(joint_reward)
                
            # update computing-queue lengths
            edge_comp_ql = next_edge_obs[0]
            device_comp_qls = [obs[0] for obs in next_device_obss]
            # obs scaling
            if hasattr(self, "obs_scaling"):
                self.obs_scaling(next_edge_obs, next_device_obss)
            
            if not self.evaluate and self.train_mode == "mappo":
                self.replay_buffer.store(edge_obs, device_obss,
                                         device_acts, device_act_logprobs,
                                         joint_reward)
            if not self.evaluate and self.train_mode == "maddpg":
                self.replay_buffer.store(edge_obs, device_obss, 
                                         device_acts, joint_reward,
                                         next_edge_obs, next_device_obss)
            
            # update obs
            edge_obs = next_edge_obs
            device_obss = next_device_obss
                    
            if not self.evaluate and self.train_mode == "maddpg":
                total_time_slots = (e_id - 1) * self.train_time_slots + t_id
                
                # train networks
                if total_time_slots % self.train_freq == 0:
                    self.edge_agent.train_nets(total_time_slots, self.replay_buffer)
                    # update agents' policy networks
                    for i in range(self.device_num):
                        self.device_agents[i].update_net(self.edge_agent.p_nets[i].state_dict())
                
                # update target networks
                if total_time_slots % self.target_update_freq == 0:
                    self.edge_agent.update_target_nets(total_time_slots)
                
                # save networks
                if total_time_slots % self.save_freq == 0:
                    self.edge_agent.save_nets(total_time_slots)
                    
        if not self.evaluate and self.train_mode == "mappo":
            # train networks
            if e_id % self.train_freq == 0:
                self.edge_agent.train_nets(self.replay_buffer)
                # update agents' policy networks
                for i in range(self.device_num):
                    self.device_agents[i].update_net(self.edge_agent.p_nets[i].state_dict())
                    
            if e_id % self.save_freq == 0:
                self.edge_agent.save_nets(e_id)
        
        joint_reward = copy.copy(self.joint_reward)
        device_rewards = copy.copy(self.device_rewards)
        joint_cost = copy.copy(self.joint_cost)
        device_costs = copy.copy(self.device_costs)
        edge_comp_ql =  copy.copy(self.edge_comp_ql)
        device_comp_qls = copy.copy(self.device_comp_qls)
        device_comp_dlys = copy.copy(self.device_comp_dlys)
        device_csum_engys = copy.copy(self.device_csum_engys)
        device_comp_expns = copy.copy(self.device_comp_expns)
        device_overtime_nums = copy.copy(self.device_overtime_nums)
        
        return joint_reward, device_rewards, \
               joint_cost, device_costs, \
               edge_comp_ql, device_comp_qls, \
               device_comp_dlys, device_csum_engys, \
               device_comp_expns, device_overtime_nums
    
    def average(self, t_id, joint_reward, device_rewards, 
                            joint_cost, device_costs, 
                            edge_comp_ql, device_comp_qls, 
                            device_comp_dlys, device_csum_engys, 
                            device_comp_expns, device_overtime_nums):
        self.joint_reward += 1 / t_id * (joint_reward - self.joint_reward)
        self.device_rewards += 1 / t_id * (device_rewards - self.device_rewards)
        self.joint_cost += 1 / t_id * (joint_cost - self.joint_cost)
        self.device_costs += 1 / t_id * (device_costs - self.device_costs)
        self.edge_comp_ql += 1 / t_id * (edge_comp_ql - self.edge_comp_ql)
        self.device_comp_qls += 1 / t_id * (device_comp_qls - self.device_comp_qls)
        self.device_comp_dlys += 1 / t_id * (device_comp_dlys - self.device_comp_dlys)
        self.device_csum_engys += 1 / t_id * (device_csum_engys - self.device_csum_engys)
        self.device_comp_expns += 1 / t_id * (device_comp_expns - self.device_comp_expns)
        self.device_overtime_nums += device_overtime_nums
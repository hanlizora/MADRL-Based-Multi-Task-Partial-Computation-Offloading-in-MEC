import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from config.params import get_mappo_params, get_maddpg_params
from rollout import Rollout

class Controller:
    def __init__(self, gen_params):
        self.device_num = gen_params.device_num
        
        # algorithm params
        alg_params = None
        if (not gen_params.evaluate and gen_params.train_mode == "mappo") or \
           (gen_params.evaluate and gen_params.eval_mode) == "mappo":
            alg_params = get_mappo_params()
        if (not gen_params.evaluate and gen_params.train_mode == "maddpg") or \
           (gen_params.evaluate and gen_params.eval_mode) == "maddpg":
            alg_params = get_maddpg_params()
        
        # rollout
        self.rollout = Rollout(gen_params, alg_params)
        
        # training
        if not gen_params.evaluate:
            self.train_episodes = alg_params.train_episodes
            self.results_dir = alg_params.results_dir
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            self.joint_reward_col = []
            self.device_rewards_col = []
            self.joint_cost_col = []
            self.device_costs_col = []
            self.edge_comp_ql_col = []
            self.device_comp_qls_col = []
            self.device_comp_dlys_col = []
            self.device_csum_engys_col = []
            self.device_comp_expns_col = []
            self.device_overtime_nums_col = []
        # evaluation
        else:
            self.eval_episodes = gen_params.eval_episodes
    
    def train(self):
        for e_id in range(1, self.train_episodes + 1):
            print("------------------train episode: " + str(e_id) + "------------------")
            
            joint_reward, device_rewards, \
            joint_cost, device_costs, \
            edge_comp_ql, device_comp_qls, \
            device_comp_dlys, device_csum_engys, \
            device_comp_expns, device_overtime_nums = self.rollout.run(e_id)
            
            # collection
            self.joint_reward_col.append(joint_reward)
            self.device_rewards_col.append(device_rewards)
            self.joint_cost_col.append(joint_cost)
            self.device_costs_col.append(device_costs)
            self.edge_comp_ql_col.append(edge_comp_ql)
            self.device_comp_qls_col.append(device_comp_qls)
            self.device_comp_dlys_col.append(device_comp_dlys)
            self.device_csum_engys_col.append(device_csum_engys)
            self.device_comp_expns_col.append(device_comp_expns)
            self.device_overtime_nums_col.append(device_overtime_nums)
            
            if e_id % 50 == 0:
                self.visualize()
            
            if e_id % 1000 == 0:
                with open(self.results_dir + "joint_rewards_" + str(e_id) + ".pkl", "wb") as f:
                    pickle.dump(self.joint_reward_col, f)
                with open(self.results_dir + "device_rewards_" + str(e_id) + ".pkl", "wb") as f:
                    pickle.dump(self.device_rewards_col, f)
                with open(self.results_dir + "joint_costs_" + str(e_id) + ".pkl", "wb") as f:
                    pickle.dump(self.joint_cost_col, f)
                with open(self.results_dir + "device_costs_" + str(e_id) + ".pkl", "wb") as f:
                    pickle.dump(self.device_costs_col, f)
                with open(self.results_dir + "edge_comp_qls_" + str(e_id) + ".pkl", "wb") as f:
                    pickle.dump(self.edge_comp_ql_col, f)
                with open(self.results_dir + "device_comp_qls_" + str(e_id) + ".pkl", "wb") as f:
                    pickle.dump(self.device_comp_qls_col, f)
                with open(self.results_dir + "device_comp_dlys_" + str(e_id) + ".pkl", "wb") as f:
                    pickle.dump(self.device_comp_dlys_col, f)
                with open(self.results_dir + "device_csum_engys_" + str(e_id) + ".pkl", "wb") as f:
                    pickle.dump(self.device_csum_engys_col, f)
                with open(self.results_dir + "device_comp_expns_" + str(e_id) + ".pkl", "wb") as f:
                    pickle.dump(self.device_comp_expns_col, f)
                with open(self.results_dir + "device_overtime_nums_" + str(e_id) + ".pkl", "wb") as f:
                    pickle.dump(self.device_overtime_nums_col, f)
                
    def evaluate(self):
        joint_reward = 0
        device_rewards = np.zeros([self.device_num], dtype = np.float32)
        joint_cost = 0
        device_costs = np.zeros([self.device_num], dtype = np.float32)
        edge_comp_ql = 0
        device_comp_qls = np.zeros([self.device_num], dtype = np.float32)
        device_comp_dlys = np.zeros([self.device_num], dtype = np.float32)
        device_csum_engys = np.zeros([self.device_num], dtype = np.float32)
        device_comp_expns = np.zeros([self.device_num], dtype = np.float32)
        device_overtime_nums = np.zeros([self.device_num], dtype = np.float32)
        
        for e_id in range(1, self.eval_episodes + 1):
            joint_reward_, device_rewards_, \
            joint_cost_, device_costs_, \
            edge_comp_ql_, device_comp_qls_, \
            device_comp_dlys_, device_csum_engys_, \
            device_comp_expns_, device_overtime_nums_ = self.rollout.run(e_id)
            
            joint_reward += joint_reward_
            device_rewards += device_rewards_
            joint_cost += joint_cost_
            device_costs += device_costs_
            edge_comp_ql += edge_comp_ql_
            device_comp_qls += device_comp_qls_
            device_comp_dlys += device_comp_dlys_
            device_csum_engys += device_csum_engys_
            device_comp_expns += device_comp_expns_
            device_overtime_nums += device_overtime_nums_
            
        # averages
        joint_reward /= self.eval_episodes
        device_rewards /= self.eval_episodes
        joint_cost /= self.eval_episodes
        device_costs /= self.eval_episodes
        edge_comp_ql /= self.eval_episodes
        device_comp_qls /= self.eval_episodes
        device_comp_dlys /= self.eval_episodes
        device_csum_engys /= self.eval_episodes
        device_comp_expns /= self.eval_episodes
        device_overtime_nums /= self.eval_episodes
        
        return joint_reward, device_rewards, \
               joint_cost, device_costs, \
               edge_comp_ql, device_comp_qls, \
               device_comp_dlys, device_csum_engys, \
               device_comp_expns, device_overtime_nums
    
    def visualize(self):
        episode_num = len(self.joint_reward_col)
        
        # x-axis
        x = np.array(range(episode_num))
        
        # joint reward
        fig1 = plt.figure(figsize = (12, 8))
        ax1 = fig1.add_subplot(1, 1, 1)
        # y = [episode_num]
        y = np.array(self.joint_reward_col)
        ax1.plot(x, y, marker = ".", linewidth = 3)
        ax1.grid(True)
        ax1.set_xlabel(xlabel = "Episodes", fontsize = 18)
        ax1.set_ylabel(ylabel = "Joint Reward", fontsize = 18)
        ax1.tick_params(axis = 'both', labelsize = 18)
        
        # device reward
        fig2 = plt.figure(figsize = (12, 8))
        ax2 = fig2.add_subplot(1, 1, 1)
        # y = [episode_num, device_num]
        y = np.array(self.device_rewards_col)
        for i in range(self.device_num):
            ax2.plot(x, y[:, i], marker = ".", linewidth = 3)
        ax2.grid(True)
        ax2.set_xlabel(xlabel = "Episodes", fontsize = 18)
        ax2.set_ylabel(ylabel = "Device Reward", fontsize = 18)
        ax2.tick_params(axis = 'both', labelsize = 18)
        
        '''
        # joint cost
        fig3 = plt.figure(figsize = (12, 8))
        ax3 = fig3.add_subplot(1, 1, 1)
        # y = [episode_num]
        y = np.array(self.joint_cost_col)
        ax3.plot(x, y, marker = ".", linewidth = 3)
        ax3.grid(True)
        ax3.set_xlabel(xlabel = "Episodes", fontsize = 18)
        ax3.set_ylabel(ylabel = "Joint Cost", fontsize = 18)
        ax3.tick_params(axis = 'both', labelsize = 18)
        '''
        
        '''
        # device cost
        fig4 = plt.figure(figsize = (12, 8))
        ax4 = fig4.add_subplot(1, 1, 1)
        # y = [episode_num, device_num]
        y = np.array(self.device_costs_col)
        for i in range(self.device_num):
            ax4.plot(x, y[:, i], marker = ".", linewidth = 3)
        ax4.grid(True)
        ax4.set_xlabel(xlabel = "Episodes", fontsize = 18)
        ax4.set_ylabel(ylabel = "Device Cost", fontsize = 18)
        ax4.tick_params(axis = 'both', labelsize = 18)
        '''
        
        '''
        # edge computing-queue length
        fig5 = plt.figure(figsize = (12, 8))
        ax5 = fig5.add_subplot(1, 1, 1)
        # y = [episode_num]
        y = np.array(self.edge_comp_ql_col)
        ax5.plot(x, y, marker = ".", linewidth = 3)
        ax5.grid(True)
        ax5.set_xlabel(xlabel = "Episodes", fontsize = 18)
        ax5.set_ylabel(ylabel = "Edge Computation-Queue Length (Gcycles)", fontsize = 18)
        ax5.tick_params(axis = 'both', labelsize = 18)
        '''
        
        '''
        # device computing-queue length
        fig6 = plt.figure(figsize = (12, 8))
        ax6 = fig6.add_subplot(1, 1, 1)
        # y = [episode_num, device_num]
        y = np.array(self.device_comp_qls_col)
        for i in range(self.device_num):
            ax6.plot(x, y[:, i], marker = ".", linewidth = 3)
        ax6.grid(True)
        ax6.set_xlabel(xlabel = "Episodes", fontsize = 18)
        ax6.set_ylabel(ylabel = "Device Computation-Queue Length (Gcycles)", fontsize = 18)
        ax6.tick_params(axis = 'both', labelsize = 18)
        '''
        
        '''
        # device cmputation-delay
        fig7 = plt.figure(figsize = (12, 8))
        ax7 = fig7.add_subplot(1, 1, 1)
        # y = [episode_num, device_num]
        y = np.array(self.device_comp_dlys_col) * pow(10, 3)
        for i in range(self.device_num):
            ax7.plot(x, y[:, i], marker = ".", linewidth = 3)
        ax7.grid(True)
        ax7.set_xlabel(xlabel = "Episodes", fontsize = 18)
        ax7.set_ylabel(ylabel = "Task Computation-Delay (ms)", fontsize = 18)
        ax7.tick_params(axis = 'both', labelsize = 18)
        '''
        
        '''
        # device consumed-energy
        fig8 = plt.figure(figsize = (12, 8))
        ax8 = fig8.add_subplot(1, 1, 1)
        # y = [episode_num, device_num]
        y = np.array(self.device_csum_engys_col) * pow(10, 3)
        for i in range(self.device_num):
            ax8.plot(x, y[:, i], marker = ".", linewidth = 3)
        ax8.grid(True)
        ax8.set_xlabel(xlabel = "Episodes", fontsize = 18)
        ax8.set_ylabel(ylabel = "Task Energy-Consumption (mJ)", fontsize = 18)
        ax8.tick_params(axis = 'both', labelsize = 18)
        '''
        
        '''
        # device computation-expense
        fig9 = plt.figure(figsize = (12, 8))
        ax9 = fig9.add_subplot(1, 1, 1)
        # y = [episode_num, device_num]
        y = np.array(self.device_comp_expns_col)
        for i in range(self.device_num):
            ax9.plot(x, y[:, i], marker = ".", linewidth = 3)
        ax9.grid(True)
        ax9.set_xlabel(xlabel = "Episodes", fontsize = 18)
        ax9.set_ylabel(ylabel = "Task Computation-Expense", fontsize = 18)
        ax9.tick_params(axis = 'both', labelsize = 18)
        '''
        
        '''
        # device overtime-number
        fig10 = plt.figure(figsize = (12, 8))
        ax10 = fig10.add_subplot(1, 1, 1)
        # y = [episode_num, device_num]
        y = np.array(self.device_overtime_nums_col)
        for i in range(self.device_num):
            ax10.plot(x, y[:, i], marker = ".", linewidth = 3)
        ax10.grid(True)
        ax10.set_xlabel(xlabel = "Episodes", fontsize = 18)
        ax10.set_ylabel(ylabel = "Device Overtime-Number", fontsize = 18)
        ax10.tick_params(axis = 'both', labelsize = 18)
        '''
        
        plt.show()
        '''
        fig1.savefig(self.results_dir 
                     + "/joint_reward_" + str(episode_num) + ".png")
        fig2.savefig(self.results_dir 
                     + "/device_rewards_" + str(episode_num) + ".png")
        fig3.savefig(self.results_dir 
                     + "/joint_cost_" + str(episode_num) + ".png")
        fig4.savefig(self.results_dir 
                     + "/device_costs_" + str(episode_num) + ".png")
        fig5.savefig(self.results_dir 
                     + "/edge_comp_ql_" + str(episode_num) + ".png")
        fig6.savefig(self.results_dir 
                     + "/device_comp_qls_" + str(episode_num) + ".png")
        fig7.savefig(self.results_dir 
                     + "/device_comp_dlys_" + str(episode_num) + ".png")
        fig8.savefig(self.results_dir 
                     + "/device_csum_engys_" + str(episode_num) + ".png")
        fig9.savefig(self.results_dir 
                     + "/device_comp_expns_" + str(episode_num) + ".png")
        fig10.savefig(self.results_dir 
                     + "/device_overtime_nums_" + str(episode_num) + ".png")
        '''
        plt.close()
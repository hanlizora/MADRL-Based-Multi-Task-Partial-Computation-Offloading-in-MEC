from env.device_env import DeviceEnv
from env.edge_env import EdgeEnv

class MECEnv():
    def __init__(self, params):
        self.device_num = params.device_num
        
        # edge env
        self.edge_env = EdgeEnv(params)
        # device envs
        self.device_envs = []
        for i in range(self.device_num):
            self.device_envs.append(DeviceEnv(i, params))
        
        # reward
        self.expense_weights = params.expense_weights
        self.energy_weights = params.energy_weights
        
    def reset(self):
        edge_obs = self.edge_env.reset()
        
        device_obss = [None for i in range(self.device_num)]
        for i in range(self.device_num):
            device_obss[i] = self.device_envs[i].reset()
        
        return edge_obs, device_obss
                
    def step(self, device_acts):
        device_sched_tasks = [None for i in range(self.device_num)]
        for i in range(self.device_num):
            sched_tasks = self.device_envs[i].compute(device_acts[i])
            device_sched_tasks[i] = sched_tasks
                
        self.edge_env.compute(device_sched_tasks)
        
        # reward
        device_rewards = [0 for i in range(self.device_num)]
        device_costs = [0 for i in range(self.device_num)]
        device_comp_dlys = [0 for i in range(self.device_num)]
        device_csum_engys = [0 for i in range(self.device_num)]
        device_comp_expns = [0 for i in range(self.device_num)]
        device_overtime_nums = [0 for i in range(self.device_num)]
        for i in range(self.device_num):
            sched_tasks = device_sched_tasks[i]
            task_num = len(sched_tasks)
            for j in range(task_num):
                task = sched_tasks[j]
                
                comp_dly = max(task.l_comp_dly, task.e_comp_dly)
                device_comp_dlys[i] += 1 / (j + 1) * (comp_dly - device_comp_dlys[i])
                
                csum_engy = task.l_csum_engy + task.e_csum_engy
                device_csum_engys[i] += 1 / (j + 1) * (csum_engy - device_csum_engys[i])
                
                comp_expn = task.comp_expn
                device_comp_expns[i] += 1 / (j + 1) * (comp_expn - device_comp_expns[i])
                
                device_costs[i] += self.energy_weights[i] * csum_engy + \
                                   self.expense_weights[i] * comp_expn
                
                if comp_dly > task.dly_cons:
                    device_rewards[i] += -5000
                    device_overtime_nums[i] += 1
                else:
                    norm_csum_engy = task.norm_csum_engy
                    norm_comp_expn = task.norm_comp_expn
                    device_rewards[i] += -1000 * (self.energy_weights[i] * 
                                                  csum_engy / norm_csum_engy +
                                                  self.expense_weights[i] * 
                                                  comp_expn / norm_comp_expn)
        joint_reward = sum(device_rewards)
        joint_cost = sum(device_costs)
        
        # next obs
        next_edge_obs = self.edge_env.get_obs()
        next_device_obss = [None for i in range(self.device_num)]
        for i in range(self.device_num):
            next_device_obss[i] = self.device_envs[i].get_obs()
                    
        return joint_reward, device_rewards, \
               joint_cost, device_costs, \
               device_comp_dlys, device_csum_engys, \
               device_comp_expns, device_overtime_nums, \
               next_edge_obs, next_device_obss
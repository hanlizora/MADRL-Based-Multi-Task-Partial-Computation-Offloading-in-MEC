import copy
import math
import numpy as np

class Task():
    def __init__(self, data_size, comp_dens):
        '''attributes'''
        # unit: Mb
        self.data_size = data_size
        # unit: Gcycles/bit
        self.comp_dens = comp_dens
        # unit: s
        self.dly_cons = None
        '''local subtask'''
        # computing delay
        self.l_comp_dly = None
        # energy consumption
        self.l_csum_engy = None
        '''offloading subtask'''
        # offloading data-size
        self.offl_dz = None
        # transmission time
        self.trans_time = None
        # computing delay 
        self.e_comp_dly = None
        # consumed energy
        self.e_csum_engy = None
        # service expense
        self.comp_expn = None
        '''normalization'''
        self.norm_csum_engy = None
        self.norm_comp_expn = None

    def __str__(self):
        return "data_size: " + str(self.data_size) + \
               "\ncomp_dens: " + str(self.comp_dens) + \
               "\ndly_cons: " + str(self.dly_cons) + \
               "\nl_comp_dly: " + str(self.l_comp_dly) + \
               "\nl_csum_engy: " + str(self.l_csum_engy) + \
               "\noffl_data_size: " + str(self.offl_dz) + \
               "\ntrans_time: " + str(self.trans_time) + \
               "\ne_comp_dly: " + str(self.e_comp_dly) + \
               "\ne_csum_engy: " + str(self.e_csum_engy) + \
               "\ncomp_expn: " + str(self.comp_expn) + \
               "\nnorm_csum_engy: " + str(self.norm_csum_engy) + \
               "\nnorm_comp_expn: " + str(self.norm_comp_expn)

class DeviceEnv():
    def __init__(self, env_id, gen_params):
        # env id
        self.env_id = env_id
        # unit: s
        self.delta = gen_params.delta
        self.task_num = gen_params.task_num
        # unit: Hz
        self.bandwidth = gen_params.total_bandwidth / gen_params.device_num
        # unit: mW
        self.trans_power = gen_params.device_trans_powers[env_id]
        self.path_loss = gen_params.device_path_loss[env_id]
        self.channel_gain = None
        # unit: mW
        self.noise_power = gen_params.spec_dens * self.bandwidth
        # unit: Mb/s
        self.trans_rate = None
        # unit: Gcycles/s
        self.device_comp_freq = gen_params.device_comp_freqs[env_id]
        # unit: Gcycles/s
        self.std_comp_freq = gen_params.std_comp_freq
        # unit: J/Gcycles
        self.engy_fac = gen_params.device_engy_facs[env_id]
        # unit: KB
        self.data_size_inl = gen_params.data_size_inls[env_id]
        # unit: cycles/bit
        self.comp_dens_inl = gen_params.comp_dens_inls[env_id]
        # unit: $/Gcycles
        self.service_price = gen_params.service_price
        
        # unit: Gcycles
        self.comp_ql = 0
        self.sched_tasks = []
    
    def reset(self):
        # reset computation-queue length
        self.comp_ql = 0
        
        # reset channel gain
        self.channel_gain = self.path_loss * np.random.exponential(1)
        
        # reset scheduling tasks
        self.sched_tasks.clear()
        for i in range(self.task_num):
            # unit: Mb
            data_size = np.random.uniform(self.data_size_inl[0],
                                          self.data_size_inl[1])
            data_size = data_size * 1024 * 8 * pow(10, -6)
            # unit: Gcycles/bit
            comp_dens = np.random.uniform(self.comp_dens_inl[0],
                                          self.comp_dens_inl[1])
            comp_dens = comp_dens * pow(10, -9)
            
            task = Task(data_size, comp_dens)
            
            comp = data_size * pow(10, 6) * comp_dens
            task.dly_cons = comp / self.std_comp_freq
            task.norm_csum_engy = comp * self.engy_fac
            task.norm_comp_expn = comp * self.service_price
            
            self.sched_tasks.append(task)
        
        # obs
        obs = self.get_obs()
        
        return obs
    
    def get_obs(self):
        comp_ql = self.comp_ql
        cgnp_rto = self.channel_gain / self.noise_power
        task_msgs = []
        for i in range(self.task_num):
            data_size = self.sched_tasks[i].data_size
            comp_dens = self.sched_tasks[i].comp_dens
            dly_cons = self.sched_tasks[i].dly_cons
            task_msgs += [data_size, comp_dens, dly_cons]
        obs = [comp_ql, cgnp_rto] + task_msgs
        
        return obs
            
    def compute(self, act):
        '''offloading'''
        # offloading data-size
        offl_dzs = {}
        for i in range(self.task_num):
            # offloading ratio
            offl_rto = act[i]
            offl_dz = self.sched_tasks[i].data_size * offl_rto
            offl_dzs[i] = offl_dz
        # ascending order
        offl_dzs = sorted(offl_dzs.items(), key = lambda x: x[1])
        # transmission-power ratio
        tspw_rto = act[-1]
        trans_power = self.trans_power * tspw_rto
        # unit: Mb/s
        trans_rate = self.bandwidth * math.log(1 + trans_power * self.channel_gain / 
                                               self.noise_power, 2) * pow(10, -6)
        total_trans_dz = trans_rate * self.delta
        total_offl_dz = 0
        # local computation
        local_comps = {}
        for task_id, offl_dz in offl_dzs:
            offl_dz = min(offl_dz, total_trans_dz)
            total_trans_dz -= offl_dz
            total_offl_dz += offl_dz
            
            task = self.sched_tasks[task_id]
            task.offl_dz = offl_dz
            # if offl_dz = 0, there is no need to queue
            if task.offl_dz == 0:
                task.trans_time = 0
                task.e_csum_engy = 0
                task.comp_expn = 0
            else:
                task.trans_time = total_offl_dz / trans_rate
                task.e_csum_engy = trans_power * pow(10, -3) * \
                                   task.offl_dz / trans_rate
                task.comp_expn = task.offl_dz * pow(10, 6) * task.comp_dens * \
                                 self.service_price
            
            local_comps[task_id] = (task.data_size - task.offl_dz) * pow(10, 6) * \
                                    task.comp_dens
            
        '''local computing'''
        # ascending order
        local_comps = sorted(local_comps.items(), key = lambda x: x[1])
        # computation-frequency ratio
        total_local_comp = self.comp_ql
        for task_id, local_comp in local_comps:
            task = self.sched_tasks[task_id]
            total_local_comp += local_comp
            
            # if local_comp = 0, there is no need to queue
            if local_comp == 0:
                task.l_comp_dly = 0
                task.l_csum_engy = 0
            else:
                task.l_comp_dly = total_local_comp / self.device_comp_freq
                task.l_csum_engy = self.engy_fac * local_comp
                                
        # update computation-queue length
        self.comp_ql = max(0, total_local_comp - self.device_comp_freq * self.delta)
        
        # update channel gain
        self.channel_gain = self.path_loss * np.random.exponential(1)
            
        # update scheduling tasks
        sched_tasks = copy.copy(self.sched_tasks)
        self.sched_tasks.clear()
        for i in range(self.task_num):
            # unit: Mb
            data_size = np.random.uniform(self.data_size_inl[0],
                                          self.data_size_inl[1])
            data_size = data_size * 1024 * 8 * pow(10, -6)
            # unit: Gcycles/bit
            comp_dens = np.random.uniform(self.comp_dens_inl[0],
                                          self.comp_dens_inl[1])
            comp_dens = comp_dens * pow(10, -9)
            
            task = Task(data_size, comp_dens)
            
            comp = data_size * pow(10, 6) * comp_dens
            task.dly_cons = comp / self.std_comp_freq
            task.norm_csum_engy = comp * self.engy_fac
            task.norm_comp_expn = comp * self.service_price
            
            self.sched_tasks.append(task)
        
        return sched_tasks
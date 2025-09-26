class EdgeEnv():
    def __init__(self, general_params):
        # unit: s
        self.delta = general_params.delta
        # unit: Gcycles/s
        self.edge_comp_freq = general_params.edge_comp_freq
        
        # unit: Gcycles
        self.comp_ql = None
        
    def reset(self):
        # reset computation-queue length
        self.comp_ql = 0
        
        # obs
        obs = self.get_obs()
        
        return obs
        
    def get_obs(self):
        obs = [self.comp_ql]
        
        return obs
    
    def compute(self, device_sched_tasks):
        device_sched_tasks_ = []
        for sched_tasks in device_sched_tasks:
            device_sched_tasks_ += sched_tasks
        device_sched_tasks_ = sorted(device_sched_tasks_, 
                                     key = lambda x: x.trans_time)
        comp_dly = self.comp_ql / self.edge_comp_freq
        self.comp_ql = max(0, self.comp_ql - self.edge_comp_freq * self.delta)
        for task in device_sched_tasks_:
            if task.trans_time == 0:
                task.e_comp_dly = 0
            else:
                task.e_comp_dly = max(comp_dly, task.trans_time) + task.offl_dz * \
                                  pow(10, 6) * task.comp_dens / self.edge_comp_freq
                self.comp_ql += max(0, task.offl_dz * pow(10, 6) * task.comp_dens - 
                                    self.edge_comp_freq * max(0, self.delta - 
                                                              max(comp_dly, task.trans_time)))
                comp_dly = task.e_comp_dly
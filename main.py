from config.params import get_params
from controller import Controller

if __name__ == '__main__':
    # params
    params = get_params()
    
    ctr = Controller(params)
    if params.evaluate == False:
        ctr.train()
    else:
        '''
        if eval_mode = "mappo", open the switches (load_weights = True, load_scales = True)
        '''
        joint_reward, device_rewards, \
        joint_cost, device_costs, \
        edge_comp_ql, device_comp_qls, \
        device_comp_dlys, device_csum_engys, \
        device_comp_expns, device_overtime_nums = ctr.evaluate()
        
        print("joint_reward:\n", joint_reward)
        print("device_rewards:\n", device_rewards)
        print("joint_cost:\n", joint_cost)
        print("device_costs:\n", device_costs)
        print("edge_comp_ql:\n", edge_comp_ql)
        print("device_comp_qls:\n", device_comp_qls)
        print("device_comp_dlys:\n", device_comp_dlys)
        print("device_csum_engys:\n", device_csum_engys)
        print("device_comp_expns:\n", device_comp_expns)
        print("device_overtime_nums:\n", device_overtime_nums)
import argparse

"""
general params
"""
def get_general_params():
    parser = argparse.ArgumentParser(description = "general params")
    
    parser.add_argument("--evaluate", type = bool, default = False,
                        help = "evaluate or train")
    
    # choices: mappo or maddpg
    parser.add_argument("--train_mode", type = str, default = "mappo",
                        help = "training mode")
    
    parser.add_argument("--eval_seed", type = int, default = 2345,
                        help = "evaluation random-seed")
    
    # choices: mappo or maddpg or local_comp or edge_comp or random_comp
    parser.add_argument("--eval_mode", type = str, default = "mappo",
                        help = "evaluation mode")
    
    parser.add_argument("--eval_episodes", type = int, default = 500,
                        help = "the number of sample-episodes for evaluation")
    
    parser.add_argument("--eval_time_slots", type = int, default = 200,
                        help = "the number of time-slots for evaluation")
    
    # environment
    parser.add_argument("--delta", type = float, default = 0.5, 
                        help = "the duration of each time-slot (s)")
    
    parser.add_argument("--device_num", type = int, default = 5,
                        help = "the number of devices")
    
    parser.add_argument("--task_num", type = int, default = 3,
                        help = "the number of arrival tasks at each time-slot")
    
    parser.add_argument("--total_bandwidth", type = float, 
                        default = 10 * pow(10, 6),
                        help = "total bandwidth (Hz)")
    
    parser.add_argument("--device_trans_powers", type = list, 
                        default = [251, 206, 126, 227, 186], 
                        help = "the transmission powers of devices (mW)")
    
    parser.add_argument("--device_path_loss", type = list, 
                        default = [2.4e-10, 7.6e-11, 8.0e-11, 4.0e-10, 5.3e-10], 
                        help = "the path loss of devices")
    
    parser.add_argument("--spec_dens", type = float, 
                        default = pow(10, -174 / 10),
                        help = "the spectral density of noise power (mW/Hz)")
    
    parser.add_argument("--device_comp_freqs", type = list, 
                        default = [2.1, 2.5, 2.8, 2.2, 2.4], 
                        help = "the computation frequencies of devices (Gcycles/s)")
    
    parser.add_argument("--std_comp_freq", type = float, default = 2, 
                        help = "standard computation frequency (Gcycles/s)")
    
    parser.add_argument("--device_engy_facs", type = list, 
                        default = [1, 1, 1, 1, 1], 
                        help = "the energy factors of devices (J/Gcycles)")
    
    parser.add_argument("--data_size_inls", type = list, 
                        default = [[20, 200], [20, 200], [20, 200], 
                                   [20, 200], [20, 200]], 
                        help = "the data-size intervals of tasks (KB)")
    
    parser.add_argument("--comp_dens_inls", type = list, 
                        default = [[200, 2000], [200, 2000], [200, 2000],  
                                   [200, 2000], [200, 2000]], 
                        help = "the computation-density intervals of tasks (cycles/bit)")
    
    parser.add_argument("--edge_comp_freq", type = float, default = 25, 
                        help = "the computation frequency of MEC server (Gcycles/s)")
    
    parser.add_argument("--service_price", type = float, default = 0.1, 
                        help = "the service price of MEC server ($/Gcycles)")
    
    parser.add_argument("--energy_weights", type = list, 
                        default = [0.8, 0.8, 0.8, 0.8, 0.8],
                        help = "the weights of tasks' energy consumption")
    
    parser.add_argument("--expense_weights", type = list, 
                        default = [0.2, 0.2, 0.2, 0.2, 0.2], 
                        help = "the weights of tasks' edge computation expense")
    
    parser.add_argument("--max_data_size", type = float, 
                        default = 200 * 1024 * 8 * pow(10, -6),
                        help = "maximum data-size (Mb)")
    
    parser.add_argument("--max_comp_dens", type = float,
                        default = 2000 * pow(10, -9),
                        help = "maximum computation density (Gcycles/bit)")
    
    params = parser.parse_args()
    
    return params

"""
mappo params
"""
def get_mappo_params():
    parser = argparse.ArgumentParser(description = "mappo params")
    
    # networks
    parser.add_argument("--obs_dim", type = int, default = 11,
                        help = "the dimension of agents' observations")
    
    parser.add_argument("--state_dim", type = int, default = 56,
                        help = "the dimension of global states")
    
    parser.add_argument("--action_dim", type = int, default = 4,
                        help = "the dimension of agents' actions")
    
    parser.add_argument("--v_hid_dims", type = list, default = [200, 200],   
                        help = "the dimension of value network's hidden layers")
    
    parser.add_argument("--p_hid_dims", type = list, default = [200, 200],   
                        help = "the dimension of policy network's hidden layers")
    
    parser.add_argument("--use_orthogonal_init", type = bool, default = True,
                        help = "whether to use orthogonal-initialization")
    
    # training
    parser.add_argument("--train_seed", type = int, default = 1234,
                        help = "training random-seed")
    
    parser.add_argument("--train_episodes", type = int, default = 4000,
                        help = "the number of training episodes")
    
    parser.add_argument("--train_time_slots", type = int, default = 200,
                        help = "the number of training time-slots")
    
    parser.add_argument("--train_freq", type = int, default = 4,     
                        help = "training frequency")
    
    parser.add_argument("--train_batch_size", type = int, default = 800,
                        help = "training batch-size")
    
    parser.add_argument("--v_epochs", type = int, default = 4,
                        help = "the number of training epoches of value network")
    
    parser.add_argument("--p_epochs", type = int, default = 4,
                        help = "the number of training epoches of policy networks")
    
    parser.add_argument("--gamma", type = float, default = 0.99,
                        help = "the discount factor of rewards")
    
    parser.add_argument("--lamda", type = float, default = 0.95,
                        help = "the parameter about GAE")
    
    parser.add_argument("--v_lr", type = float, default = 4e-4,
                        help = "the learning-rate of value network")
    
    parser.add_argument("--p_lr", type = float, default = 4e-4,
                        help = "the learning-rate of policy networks")
    
    parser.add_argument("--use_lr_decay", type = bool, default = False,
                        help = "whether to use learning-rate decay")
    
    parser.add_argument("--min_v_lr", type = float, default = 1e-4,        
                        help = "the minimal learning-rate of value network")
    
    parser.add_argument("--min_p_lr", type = float, default = 1e-4,        
                        help = "the minimal learning-rate of policy networks")
    
    parser.add_argument("--decay_fac", type = float, default = 0.999, 
                        help = "the parameter about learning-rate decay")
    
    parser.add_argument("--use_obs_scaling", type = bool, default = True, 
                        help = "whether to use observation scaling")
    
    parser.add_argument("--use_reward_scaling", type = bool, default = True, 
                        help = "whether to use reward scaling")
    
    parser.add_argument("--use_grad_clip", type = bool, default = True, 
                        help = "whether to use gradient clip")
    
    parser.add_argument("--v_grad_clip", type = float, default = 2,  
                        help = "the parameter about value network's gradient clip")
    
    parser.add_argument("--p_grad_clip", type = float, default = 2,  
                        help = "the parameter about policy networks' gradient clip")
    
    parser.add_argument("--p_clip", type = float, default = 0.1,
                        help = "the parameter about ppo clip")
    
    parser.add_argument("--enty_coef", type = float, default = 0.01,   
                        help = "the coefficient about policy's entropy")
    
    parser.add_argument("--save_freq", type = int, default = 500, 
                        help = "the saving frequency of networks")
    
    parser.add_argument("--load_weights", type = bool, default = False, 
                        help = "whether to load network parameters")
    
    parser.add_argument("--weights_dir", type = str, default = "weight/mappo/", 
                        help = "the directory for saving network parameters")
    
    parser.add_argument("--results_dir", type = str, default = "result/mappo/",
                        help = "the directory for saving training results")
    
    params = parser.parse_args()
    
    return params

"""
mddpg params
"""
def get_maddpg_params():
    parser = argparse.ArgumentParser(description = "maddpg params")
    
    # networks
    parser.add_argument("--obs_dim", type = int, default = 11,
                        help = "the dimension of agents' observations")
    
    parser.add_argument("--state_action_dim", type = int, default = 256,
                        help = "the dimension of global states")
    
    parser.add_argument("--action_dim", type = int, default = 40,
                        help = "the dimension of agents' actions")
    
    parser.add_argument("--v_hid_dims", type = list, default = [400, 400],
                        help = "the dimension of value network's hidden layers")
    
    parser.add_argument("--p_hid_dims", type = list, default = [400, 400], 
                        help = "the dimension of policy network's hidden layers")
    
    parser.add_argument("--use_orthogonal_init", type = bool, default = True,
                        help = "whether to use orthogonal-initialization")
    
    # training
    parser.add_argument("--train_seed", type = int, default = 1234,
                        help = "training random-seed")
    
    parser.add_argument("--train_episodes", type = int, default = 4000,
                        help = "the number of training episodes")
    
    parser.add_argument("--train_time_slots", type = int, default = 200,
                        help = "the number of training time-slots")
    
    parser.add_argument("--warm_time_slots", type = int, default = 4000,
                        help = "the number of warming time-slots")
    
    parser.add_argument("--train_freq", type = int, default = 400,
                        help = "training frequency")
    
    parser.add_argument("--target_update_freq", type = int, default = 8000,
                        help = "the updating frequency of target networks")
    
    parser.add_argument("--train_batch_size", type = int, default = 1000,
                        help = "training batch-size")
    
    parser.add_argument("--v_epochs", type = int, default = 4,
                        help = "the number of training epochs of value network")
    
    parser.add_argument("--p_epochs", type = int, default = 1,
                        help = "the number of training epochs of policy networks")
    
    parser.add_argument("--buffer_size", type = int, default = 40000,       
                        help = "the size of replay buffer")
    
    parser.add_argument("--gamma", type = float, default = 0.99,
                        help = "the discount factor of rewards")
    
    parser.add_argument("--v_lr", type = float, default = 1e-5,           
                        help = "the learning-rate of value network")
    
    parser.add_argument("--p_lr", type = float, default = 1e-5,          
                        help = "the learning-rate of policy networks")
    
    parser.add_argument("--use_lr_decay", type = bool, default = True,
                        help = "whether to use learning-rate decay")
    
    parser.add_argument("--min_v_lr", type = float, default = 1e-7,       
                        help = "the minimal learning-rate of value network")
    
    parser.add_argument("--min_p_lr", type = float, default = 1e-7,   
                        help = "the minimal learning-rate of policy networks")
    
    parser.add_argument("--decay_intl", type = int, default = 300000,  
                        help = "the interval of learning-rate decay")
    
    parser.add_argument("--decay_fac", type = float, default = 4.95e-06,
                        help = "the parameter about learning-rate decay")
    
    parser.add_argument("--use_obs_scaling", type = bool, default = True, 
                        help = "whether to use observation scaling")
    
    parser.add_argument("--use_reward_scaling", type = bool, default = True, 
                        help = "whether to use reward scaling")
    
    parser.add_argument("--use_action_noise", type = bool, default = True,  
                        help = "whether to add noise in agents' actions")
    
    parser.add_argument("--use_grad_clip", type = bool, default = True, 
                        help = "whether to use gradient clip")
    
    parser.add_argument("--v_grad_clip", type = float, default = 2,
                        help = "the parameter about value network's gradient clip")
    
    parser.add_argument("--p_grad_clip", type = float, default = 2,   
                        help = "the parameter about policy networks' gradient clip")
    
    parser.add_argument("--save_freq", type = int, default = 100000,
                        help = "the saving frequency of networks")
    
    parser.add_argument("--load_weights", type = bool, default = False, 
                        help = "whether to load network parameters")
    
    parser.add_argument("--weights_dir", type = str, default = "weight/maddpg/", 
                        help = "the directory for saving network parameters")
    
    parser.add_argument("--results_dir", type = str, default = "result/maddpg/",
                        help = "the directory for saving training results")
    
    params = parser.parse_args()
    
    return params
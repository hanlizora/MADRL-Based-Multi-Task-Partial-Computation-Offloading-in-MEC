from config.params import get_params

# params
params = get_params()

'''communication'''
# distance (sampled uniformly from [100, 200] m)
dis = [148.92, 198.14, 183.25, 130.30, 174.58]
# bandwidth (Hz)
W = params.bandwidth / params.device_num
# noise power (mW)
sigma2 = params.spec_dens * W
# path loss
def path_loss(dis):
    return pow(10, -128.1 / 10) * pow((dis / 1000), -3.76)

# [1] A Joint Uplink/Downlink Resource Allocation Algorithm in OFDMA Wireless Networks
# [2] Energy-Efficient Resource Allocation for Mobile Edge Computing-Based Augmented Reality Applications
# [3] On the Application of Uplink/Downlink Decoupled Access in Heterogeneous Mobile Edge Computing
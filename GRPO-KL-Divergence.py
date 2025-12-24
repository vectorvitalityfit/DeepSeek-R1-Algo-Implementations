import numpy as np

def grpo_kl_divergence(pi_theta,pi_ref):
    # Calculate the importance ratio r
    r=pi_ref/pi_theta
    
    # Apply the GRPO KL divergence formula: r-log(r)-1
    kl_estimate=r-np.log(r)-1
    
    return kl_estimate
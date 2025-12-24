import numpy as np

def grpo_objective(rhos,A,pi_theta_old,pi_theta_ref,epsilon=0.2,beta=0.01) -> float:
    """
    Compute the GRPO objective function.

    Args:
        rhos: List of likelihood ratios (rho_i).
        A: List of advantage estimates (A_i).
        pi_theta_old: List of old policy probabilities (current policy).
        pi_theta_ref: List of reference policy probabilities.
        epsilon: Clipping parameter.
        beta: KL divergence penalty coefficient.

    Returns:
        The computed GRPO objective value.
    """
    rhos=np.array(rhos,dtype=np.float64)
    A=np.array(A,dtype=np.float64)
    pi_theta_old=np.array(pi_theta_old,dtype=np.float64)
    pi_theta_ref=np.array(pi_theta_ref,dtype=np.float64)

    clipped_rho=np.clip(rhos,1-epsilon,1+epsilon)
    surrogate1=rhos*A
    surrogate2=clipped_rho*A
    surrogate_clipped=np.minimum(surrogate1, surrogate2)

    kl_rho=pi_theta_old/pi_theta_ref
    kl_penalty=kl_rho-np.log(kl_rho)-1

    combined=surrogate_clipped-beta*kl_penalty
    objective_value=np.mean(combined)

    return objective_value
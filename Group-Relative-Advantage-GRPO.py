import numpy as np

def compute_group_relative_advantage(rewards):
    # Convert list of rewards to a NumPy array
    np_rewards=np.array(rewards)

    # Calculate the mean and standard deviation
    mean=np.mean(np_rewards)
    std=np.std(np_rewards)

    # Handle the edge case where standard deviation is close to zero
    if np.isclose(std,0.0):
        return [0.0]*len(rewards)

    # Calculate the advantage for each reward
    advantages=(np_rewards-mean)/std

    # Convert the result back to a list
    return advantages.tolist()
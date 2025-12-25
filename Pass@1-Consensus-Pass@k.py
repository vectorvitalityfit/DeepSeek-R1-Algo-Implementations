import numpy as np
from collections import Counter

def pass_at_1(responses_correct):
    arrayy=np.array(responses_correct)
    return arrayy.mean()

def majority_voting(responses):
    counts=Counter(responses)
    return counts.most_common(1)[0][0]

def pass_at_k(n:int,c:int,k:int)->float:
    """
    Compute unbiased pass@k from n samples with c correct.
    
    Formula: pass@k=1-C(n-c,k)/C(n,k)
    
    Arguments:
        n: Total samples
        c: Correct samples
        k: k in pass@k
        
    Returns:
        Estimated pass@k
    """
    if c==0:
        return 0.0
    if k>n:
        return 1.0
    product=1.0
    for i in range(k):
        product*=(n-c-i)/(n-i)
    return 1-product
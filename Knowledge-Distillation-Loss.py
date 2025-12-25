import numpy as np

def distillation_loss(student_logits:np.ndarray,teacher_logits:np.ndarray,temperature:float=1.0) -> float:
    """
    L=T^2*KL(softmax(teacher/T)||softmax(student/T))
    Arguments:
        student_logits: Logits from student model
        teacher_logits: Logits from teacher model
        temperature:    Softmax temperature
    Returns:
        Distillation loss value
    """
    def softmax(logits,t):
        logits=logits-np.max(logits)
        exponential_logits=np.exp(logits/t)
        return exponential_logits/np.sum(exponential_logits)

    def kl_divergence(p,q,epsilon=1e-12):
        p=np.clip(p,epsilon,1.0)
        q=np.clip(q,epsilon,1.0)
        return np.sum(p*np.log(p/q))
    
    teacher_probs=softmax(teacher_logits,temperature)
    student_probs=softmax(student_logits,temperature)
    kl_div=kl_divergence(teacher_probs,student_probs)
    return (temperature**2)*kl_div